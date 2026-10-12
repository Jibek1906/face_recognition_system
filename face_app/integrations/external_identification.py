"""
External Integration Module
----------------------------
Отправляет фото клиента на внешний сервер для идентификации.
Получает в ответ ФИО, ID и опциональный дополнительный ID (например, для семьи/группы).

Логика работы:
1. Проверяем, есть ли уже ФИО в базе → если да, пропускаем (нет дубликатов).
2. Декодируем photo_base64 в изображение OpenCV.
3. Проверяем размытость (Laplacian variance). Если фото размыто → пропускаем.
4. Проверяем позу лица (должно смотреть прямо — "frontal").
5. Асинхронно отправляем POST-запрос на внешний сервер.
6. Сохраняем ФИО, person_id (внешний) и additional_id в базу данных.

Наш локальный face.id НИКОГДА не меняется.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger("external_integration")

# ───────────────────────────────────────────────
#  Глобальный кэш: face_id → флаг "уже отправлено"
#  Защищает от дублирования при перезапуске пайплайна
#  (хранится только в памяти, после рестарта очищается,
#   но при рестарте проверка по полю name в БД защищает повторно)
# ───────────────────────────────────────────────
_sent_cache: set[int] = set()


def _get_settings():
    """Читаем настройки через settings_manager (живой reload без рестарта)."""
    try:
        from face_app.core.settings.manager import settings
        return settings
    except Exception:
        return None


def _check_blur(img: np.ndarray, threshold: float) -> tuple[bool, float]:
    """
    Проверка резкости изображения через Laplacian variance.
    Чем выше variance, тем чётче изображение.
    Returns: (is_sharp, variance_value)
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    is_sharp = variance >= threshold
    return is_sharp, variance


def _check_pose_frontal(img: np.ndarray) -> tuple[bool, str]:
    """
    Проверяет, что лицо смотрит прямо (frontal).
    Использует существующий FaceRecognizer.detect_head_pose.
    Returns: (is_frontal, angle_category)
    """
    try:
        from face_app.vision.recognition.face_recognition import FaceRecognizer
        # FaceRecognizer — тяжёлая модель, используем синглтон если он уже загружен
        recognizer = _get_recognizer()
        if recognizer is None:
            # Не можем проверить позу, разрешаем отправку
            return True, "unknown"

        pose = recognizer.detect_head_pose(img)
        angle = pose.get("angle_category", "unknown")
        is_frontal = angle in ("frontal", "unknown")
        return is_frontal, angle
    except Exception as e:
        logger.debug(f"Pose check error: {e}")
        return True, "unknown"  # При ошибке не блокируем


def _get_recognizer():
    """Получаем существующий FaceRecognizer из глобального состояния (не создаём новый)."""
    try:
        from face_app.state import video_state
        system = getattr(video_state, "global_face_system", None)
        if system and hasattr(system, "recognizer"):
            return system.recognizer
        if system and hasattr(system, "face_recognizer"):
            return system.face_recognizer
    except Exception:
        pass
    return None


async def process_and_send_face(face_id: int) -> dict:
    """
    Основная функция. Выполняется асинхронно, не блокирует видеопоток.

    Шаги:
    1. Проверка in-memory кэша
    2. Загрузка Face из БД + проверка ФИО
    3. Фильтр по размытости
    4. Фильтр по позе
    5. HTTP-запрос к внешнему серверу
    6. Сохранение результата

    Returns: dict со статусом операции
    """
    # ── Шаг 0: кэш in-memory ──────────────────────────────────────
    if face_id in _sent_cache:
        logger.debug(f"[ExtInt] face_id={face_id} уже отправлен (кэш), пропускаем.")
        return {"status": "skipped", "reason": "already_sent_cache"}

    settings = _get_settings()
    if settings and not settings.get("external_integration_enabled", True):
        return {"status": "skipped", "reason": "integration_disabled"}

    external_url = settings.get("external_api_url", "") if settings else ""
    if not external_url or external_url.startswith("http://127.0.0.1:8000/dummy_external"):
        logger.warning("[ExtInt] external_api_url не настроен. Укажите реальный URL в settings.json.")
        return {"status": "skipped", "reason": "url_not_configured"}

    blur_threshold = float(settings.get("blur_threshold", 100.0)) if settings else 100.0

    # ── Шаг 1: Загружаем Face из БД ──────────────────────────────
    from face_app.db.session import get_session
    from face_app.db.models import Face

    session = get_session()
    try:
        face: Optional[Face] = session.query(Face).filter(Face.id == face_id).first()
        if face is None:
            logger.warning(f"[ExtInt] face_id={face_id} не найден в БД.")
            return {"status": "error", "reason": "face_not_found"}

        # ── Шаг 2: Если ФИО уже есть → дубликат, пропускаем ─────
        # «ФИО уже есть» = name существует И не начинается с «Client» (дефолтное имя)
        if face.name and not face.name.startswith("Client "):
            logger.info(f"[ExtInt] face_id={face_id} уже имеет ФИО='{face.name}'. Пропускаем.")
            _sent_cache.add(face_id)
            return {"status": "skipped", "reason": "fio_already_present"}

        photo_b64 = face.photo_base64
        local_face_db_id = face.id  # наш локальный ID (не меняется)
    finally:
        session.close()

    if not photo_b64:
        logger.warning(f"[ExtInt] face_id={face_id} — нет фото.")
        return {"status": "skipped", "reason": "no_photo"}

    # ── Шаг 3: Декодируем фото ────────────────────────────────────
    try:
        img_data = base64.b64decode(photo_b64)
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None or img.size == 0:
            raise ValueError("Декодирование вернуло пустое изображение")
    except Exception as e:
        logger.error(f"[ExtInt] face_id={face_id} — ошибка декодирования фото: {e}")
        return {"status": "error", "reason": f"decode_error: {e}"}

    # ── Шаг 4: Проверка размытости ────────────────────────────────
    is_sharp, variance = _check_blur(img, blur_threshold)
    if not is_sharp:
        logger.info(f"[ExtInt] face_id={face_id} — фото размыто (variance={variance:.1f} < {blur_threshold}). Пропускаем.")
        return {"status": "skipped", "reason": "blurry_photo", "blur_variance": variance}

    # ── Шаг 5: Проверка позы ──────────────────────────────────────
    # Выполняем в отдельном потоке, чтобы не блокировать event loop
    is_frontal, angle = await asyncio.get_event_loop().run_in_executor(
        None, _check_pose_frontal, img
    )
    if not is_frontal:
        logger.info(f"[ExtInt] face_id={face_id} — лицо не фронтальное (angle='{angle}'). Пропускаем.")
        return {"status": "skipped", "reason": "not_frontal", "angle": angle}

    # ── Шаг 6: HTTP-запрос к внешнему серверу ────────────────────
    logger.info(f"[ExtInt] face_id={face_id} — отправляем на {external_url} (blur={variance:.1f}, angle={angle})")

    try:
        import aiohttp
    except ImportError:
        logger.error("[ExtInt] aiohttp не установлен. Установите: pip install aiohttp")
        return {"status": "error", "reason": "aiohttp_not_installed"}

    payload = {
        "face_id": local_face_db_id,   # наш локальный ID — сторонний сервер знает этот ключ
        "photo_base64": photo_b64,
    }

    try:
        async with aiohttp.ClientSession() as http_session:
            async with http_session.post(
                external_url,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=30),
            ) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    logger.error(f"[ExtInt] face_id={face_id} — сервер вернул {resp.status}: {body[:200]}")
                    return {"status": "error", "reason": f"http_{resp.status}", "body": body[:200]}

                data = await resp.json()
                logger.info(f"[ExtInt] face_id={face_id} — ответ: {data}")

    except asyncio.TimeoutError:
        logger.error(f"[ExtInt] face_id={face_id} — таймаут запроса.")
        return {"status": "error", "reason": "timeout"}
    except Exception as e:
        logger.error(f"[ExtInt] face_id={face_id} — ошибка HTTP: {e}")
        return {"status": "error", "reason": str(e)}

    # ── Шаг 7: Парсим ответ и сохраняем в БД ─────────────────────
    #
    # Ожидаемый формат ответа:
    # {
    #   "id": <int|str>,          # внешний ID персоны (person_id на их стороне)
    #   "fio": "<Имя Фамилия>",   # ФИО
    #   "additional_id": <str>    # ОПЦИОНАЛЬНО — ID семьи/группы
    # }
    #
    fio = data.get("fio") or data.get("name") or data.get("full_name")
    external_id = data.get("id") or data.get("person_id")
    additional_id = data.get("additional_id") or data.get("family_id") or data.get("group_id")

    if not fio:
        logger.warning(f"[ExtInt] face_id={face_id} — ответ не содержит ФИО: {data}")
        return {"status": "error", "reason": "no_fio_in_response", "response": data}

    # Сохраняем в базу данных
    session = get_session()
    try:
        face = session.query(Face).filter(Face.id == face_id).first()
        if face:
            face.name = str(fio).strip()

            # person_id обновляем ТОЛЬКО если пришёл внешний ID
            if external_id is not None:
                face.person_id = int(external_id)

            # additional_id — дополнительный ID для группировки (семья и т.д.)
            if additional_id is not None:
                face.additional_id = str(additional_id).strip()

            if face.status == "new":
                face.status = "known"

            session.commit()
            logger.info(
                f"[ExtInt] ✅ face_id={face_id} обновлён: "
                f"fio='{fio}', person_id={external_id}, additional_id={additional_id}"
            )

            # Добавляем в кэш, чтобы не отправлять повторно
            _sent_cache.add(face_id)

            # Принудительно перезагружаем эмбеддинги в IdentityManager
            try:
                from face_app.state import video_state
                system = getattr(video_state, "global_face_system", None)
                if system and hasattr(system, "identity_manager"):
                    system.identity_manager.refresh_definitions()
            except Exception:
                pass

            return {
                "status": "success",
                "face_id": face_id,
                "fio": fio,
                "external_id": external_id,
                "additional_id": additional_id,
            }
        else:
            session.rollback()
            return {"status": "error", "reason": "face_disappeared_from_db"}

    except Exception as e:
        session.rollback()
        logger.error(f"[ExtInt] face_id={face_id} — ошибка при сохранении в БД: {e}")
        return {"status": "error", "reason": f"db_save_error: {e}"}
    finally:
        session.close()


def schedule_send(face_id: int):
    """
    Безопасный запуск process_and_send_face из синхронного кода.
    Если event loop запущен — создаём задачу в нём.
    Если нет — запускаем в отдельном потоке.
    """
    try:
        loop = asyncio.get_running_loop()
        loop.create_task(_send_with_log(face_id))
    except RuntimeError:
        # Нет запущенного event loop — запускаем в потоке
        import threading
        t = threading.Thread(
            target=asyncio.run,
            args=(_send_with_log(face_id),),
            daemon=True,
            name=f"ext_integration_{face_id}"
        )
        t.start()


async def _send_with_log(face_id: int):
    """Обёртка для логирования результата фоновой задачи."""
    try:
        result = await process_and_send_face(face_id)
        status = result.get("status")
        reason = result.get("reason", "")
        if status == "success":
            logger.info(f"[ExtInt BG] face_id={face_id} → Успешно: {result.get('fio')}")
        elif status == "skipped":
            logger.debug(f"[ExtInt BG] face_id={face_id} → Пропущено: {reason}")
        else:
            logger.warning(f"[ExtInt BG] face_id={face_id} → Ошибка: {reason}")
    except Exception as e:
        logger.error(f"[ExtInt BG] face_id={face_id} → Необработанное исключение: {e}", exc_info=True)
