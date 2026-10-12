"""
Mock External Identification Server
=====================================
Имитирует сторонний сервер идентификации.
Принимает фото + face_id, возвращает ФИО и ID.

Запуск:
    cd /var/www/face_recognition/face_system
    venv/bin/python -m face_app.cli.tools.mock_external_server

Сервер запустится на http://127.0.0.1:5050
"""

import base64
import json
import logging
from http.server import BaseHTTPRequestHandler, HTTPServer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [MOCK] %(message)s")
logger = logging.getLogger("mock_server")

# ─── Тестовые данные ──────────────────────────────────────────────────────────
# face_id → что вернуть
MOCK_DB = {
    # Если face_id совпадает — вернём конкретные данные
    # Добавьте сюда реальные face_id из вашей базы для тестирования
    1:   {"id": 1001, "fio": "Иванов Иван Иванович", "additional_id": None},
    2:   {"id": 1002, "fio": "Петрова Мария Сергеевна", "additional_id": "FAM-001"},
    3:   {"id": 1001, "fio": "Иванов Иван Иванович", "additional_id": "FAM-001"},  # ребёнок
}

# Если face_id нет в MOCK_DB — вернём это (дефолтный ответ)
DEFAULT_RESPONSE = {
    "id": 9999,
    "fio": "Тест Тестов",
    "additional_id": "GROUP-TEST"
}
# ─────────────────────────────────────────────────────────────────────────────


class MockHandler(BaseHTTPRequestHandler):

    def log_message(self, format, *args):
        # Переопределяем чтобы использовать наш логгер
        logger.info("%s - %s" % (self.address_string(), format % args))

    def do_POST(self):
        if self.path != "/identify":
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b'{"error": "Not found"}')
            return

        # Читаем тело запроса
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length)

        try:
            data = json.loads(body)
        except Exception:
            self.send_response(400)
            self.end_headers()
            self.wfile.write(b'{"error": "Invalid JSON"}')
            return

        face_id = data.get("face_id")
        photo_b64 = data.get("photo_base64", "")

        # Краткий лог (не выводим весь base64)
        photo_len = len(photo_b64)
        logger.info(f"📥 Получен запрос: face_id={face_id}, photo_size={photo_len} chars")

        # Сохраняем фото для просмотра (опционально)
        if photo_b64 and face_id:
            try:
                import os
                save_dir = "/tmp/mock_received_photos"
                os.makedirs(save_dir, exist_ok=True)
                photo_bytes = base64.b64decode(photo_b64)
                photo_path = os.path.join(save_dir, f"face_{face_id}.jpg")
                with open(photo_path, "wb") as f:
                    f.write(photo_bytes)
                logger.info(f"💾 Фото сохранено: {photo_path}")
            except Exception as e:
                logger.warning(f"Не удалось сохранить фото: {e}")

        # Формируем ответ
        response = MOCK_DB.get(face_id, DEFAULT_RESPONSE)
        logger.info(f"📤 Ответ: {response}")

        resp_bytes = json.dumps(response, ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(resp_bytes)))
        self.end_headers()
        self.wfile.write(resp_bytes)


if __name__ == "__main__":
    PORT = 5050
    server = HTTPServer(("127.0.0.1", PORT), MockHandler)
    logger.info(f"🚀 Mock сервер запущен на http://127.0.0.1:{PORT}/identify")
    logger.info("📋 Настройте в settings.json:")
    logger.info(f'   "external_api_url": "http://127.0.0.1:{PORT}/identify"')
    logger.info("   Затем вызовите POST /api/external/identify/{{face_id}}")
    logger.info("   Фото будут сохраняться в /tmp/mock_received_photos/")
    logger.info("   Нажмите Ctrl+C для остановки")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Сервер остановлен")
