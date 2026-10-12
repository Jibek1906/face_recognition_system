"""SQLAlchemy models.

This is a move of the legacy `face_system/models.py` to a more conventional
package location.
"""

from __future__ import annotations

from datetime import date, datetime

import json
import numpy as np
import pytz
from sqlalchemy import Column, Date, DateTime, Float, ForeignKey, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship


Base = declarative_base()
bishkek_tz = pytz.timezone("Asia/Bishkek")


def get_bishkek_time():
    """Получить текущее время в Бишкеке (без timezone)."""

    utc_now = datetime.utcnow()
    bishkek_now = pytz.utc.localize(utc_now).astimezone(bishkek_tz)
    return bishkek_now.replace(tzinfo=None)


class Face(Base):
    __tablename__ = "faces"

    id = Column(Integer, primary_key=True, index=True)
    person_id = Column(Integer, index=True, nullable=True)
    name = Column(String(255), nullable=True)  # ✅ NEW: Person name
    additional_id = Column(String(255), nullable=True)  # ✅ NEW: Family ID
    photo_base64 = Column(String(length=None), nullable=True)
    embedding = Column(String(length=None), nullable=False)
    body_embedding = Column(String(length=None), nullable=True)  # ✅ NEW: body embedding
    status = Column(String(50), default="new", index=True)
    total_time_seconds = Column(Float, default=0.0)
    first_seen = Column(DateTime, default=get_bishkek_time, index=True)
    last_seen = Column(DateTime, default=get_bishkek_time)
    detection_count = Column(Integer, default=1)
    additional_id = Column(String(255), nullable=True) # ✅ NEW: External family/group ID

    def __repr__(self):
        return f"<Face id={self.id} person_id={self.person_id} status={self.status}>"

    def to_dict(self):
        return {
            "id": self.id,
            "person_id": self.person_id,
            "additional_id": self.additional_id,
            "name": self.name,  # ✅ NEW: Include name
            "status": self.status,
            "total_time_seconds": round(self.total_time_seconds, 2),
            "detection_count": self.detection_count,
            "first_seen": self.first_seen.isoformat() if self.first_seen else None,
            "last_seen": self.last_seen.isoformat() if self.last_seen else None,
            "photo_base64": self.photo_base64[:100] + "..." if self.photo_base64 else None,
            "is_employee": self.status == "employee",
        }

    @staticmethod
    def embedding_to_list(embedding_json: str) -> list:
        return json.loads(embedding_json)

    @staticmethod
    def embedding_from_list(embedding_list) -> str:
        if hasattr(embedding_list, "tolist"):
            embedding_list = embedding_list.tolist()
        return json.dumps(embedding_list)


class Visit(Base):
    """✅ Каждая запись = одна сессия (непрерывное присутствие от входа до выхода)"""

    __tablename__ = "visits"
    id = Column(Integer, primary_key=True)
    face_id = Column(Integer, ForeignKey("faces.id"), index=True)
    timestamp = Column(DateTime, default=get_bishkek_time, index=True)  # Время начала сессии
    end_time = Column(DateTime, nullable=True)  # Время окончания (NULL = активная сессия)
    face = relationship("Face", backref="visits", lazy="joined")


class PersonDailyVisits(Base):
    """Точное количество визитов (сессий) по каждому человеку за каждый день"""

    __tablename__ = "person_daily_visits"
    id = Column(Integer, primary_key=True)
    face_id = Column(Integer, ForeignKey("faces.id"), index=True)
    date = Column(Date, default=date.today, index=True)
    session_count = Column(Integer, default=0)

    face = relationship("Face", backref="person_daily_visits")


class DailyStats(Base):
    """Статистика по дням для каждого посетителя"""

    __tablename__ = "daily_stats"
    id = Column(Integer, primary_key=True)
    date = Column(Date, default=date.today, index=True)
    face_id = Column(Integer, ForeignKey("faces.id"), index=True)
    visit_count = Column(Integer, default=0)
    total_minutes = Column(Float, default=0.0)

    face = relationship("Face", backref="daily_stats")


class PersonPhoto(Base):
    """✅ ИСПРАВЛЕНО: Таблица дополнительных фотографий"""

    __tablename__ = "person_photos"

    id = Column(Integer, primary_key=True, index=True)
    face_id = Column(Integer, ForeignKey("faces.id"), nullable=False, index=True)
    photo_base64 = Column(String(length=None), nullable=False)
    embedding = Column(String(length=None), nullable=False)  # ✅ JSON string
    body_embedding = Column(String(length=None), nullable=True)  # ✅ NEW: body embedding (JSON string)
    angle = Column(String(50), nullable=True)  # frontal, left_profile, right_profile, etc.
    angle_manual = Column(String(50), nullable=True)  # Ручное переопределение
    pose_pitch = Column(Float, default=0.0)
    pose_yaw = Column(Float, default=0.0)
    pose_roll = Column(Float, default=0.0)
    created_at = Column(DateTime, default=get_bishkek_time)

    face = relationship("Face", backref="additional_photos")

    def __repr__(self):
        return f"<PersonPhoto id={self.id} face_id={self.face_id} angle={self.angle}>"

    def to_dict(self):
        return {
            "id": self.id,
            "face_id": self.face_id,
            "photo_base64": self.photo_base64,
            "angle": self.angle,
            "pose_pitch": self.pose_pitch,
            "pose_yaw": self.pose_yaw,
            "pose_roll": self.pose_roll,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


def group_visits_to_sessions(visits, session_gap_minutes=10):
    """Group visits into sessions.

    This function is used by API responses; logic kept 1:1 with legacy
    implementation.
    """

    if not visits:
        return [], 0

    visits_sorted = sorted(visits, key=lambda v: v.timestamp)

    sessions = []
    session_start = visits_sorted[0].timestamp
    session_end = visits_sorted[0].timestamp
    total_seconds = 0

    for v in visits_sorted[1:]:
        delta_minutes = (v.timestamp - session_end).total_seconds() / 60

        if delta_minutes > session_gap_minutes:
            session_duration = (session_end - session_start).total_seconds()

            if session_duration < 10:
                if len([vi for vi in visits_sorted if session_start <= vi.timestamp <= session_end]) > 1:
                    session_duration = session_duration
                else:
                    session_duration = 10
            elif session_duration > 3600:
                session_duration = 3600

            total_seconds += session_duration
            sessions.append((session_start, session_end))

            session_start = v.timestamp

        session_end = v.timestamp

    session_duration = (session_end - session_start).total_seconds()
    if session_duration < 10:
        if len([v for v in visits_sorted if session_start <= v.timestamp <= session_end]) > 1:
            session_duration = session_duration
        else:
            session_duration = 10
    elif session_duration > 3600:
        session_duration = 3600

    total_seconds += session_duration
    sessions.append((session_start, session_end))

    # Legacy contract: return integer seconds
    return sessions, int(round(total_seconds))


def calculate_real_session_time(visits_in_session, min_seconds=10, max_seconds=3600):
    """✅ НОВАЯ ФУНКЦИЯ: Рассчитывает реалистичное время сессии.

    Copied from legacy `models.py` for compatibility.
    """

    if not visits_in_session:
        return min_seconds

    times = [v.timestamp for v in visits_in_session]
    times.sort()

    # Общее время от первого до последнего визита
    total_span = (times[-1] - times[0]).total_seconds()

    # Плотность визитов (сколько визитов в секунду)
    if total_span > 0:
        density = len(visits_in_session) / total_span
    else:
        density = 1.0  # Все визиты в одну секунду

    if density < 0.2:
        # Низкая плотность - человек появлялся ненадолго
        realistic_time = total_span * density * 3
    elif density > 0.5:
        # Высокая плотность - человек был постоянно
        realistic_time = total_span * 0.9
    else:
        # Средняя плотность
        realistic_time = total_span * 0.7

    realistic_time = max(min_seconds, min(realistic_time, max_seconds))
    return int(round(realistic_time))


def group_visits_to_sessions_improved(visits, session_gap_minutes=10):
    """✅ УЛУЧШЕННАЯ ВЕРСИЯ: Использует реалистичную оценку времени."""

    if not visits:
        return [], 0

    visits_sorted = sorted(visits, key=lambda v: v.timestamp)

    sessions = []
    current_session_visits = [visits_sorted[0]]
    total_seconds = 0

    for i in range(1, len(visits_sorted)):
        current_visit = visits_sorted[i]
        last_visit = current_session_visits[-1]

        delta_minutes = (current_visit.timestamp - last_visit.timestamp).total_seconds() / 60

        if delta_minutes > session_gap_minutes:
            session_time = calculate_real_session_time(current_session_visits)
            total_seconds += session_time

            sessions.append(
                (
                    current_session_visits[0].timestamp,
                    current_session_visits[-1].timestamp,
                    session_time,
                )
            )

            current_session_visits = [current_visit]
        else:
            current_session_visits.append(current_visit)

    if current_session_visits:
        session_time = calculate_real_session_time(current_session_visits)
        total_seconds += session_time

        sessions.append(
            (
                current_session_visits[0].timestamp,
                current_session_visits[-1].timestamp,
                session_time,
            )
        )

    # Возвращаем в том же формате для совместимости
    simple_sessions = [(start, end) for start, end, _ in sessions]
    return simple_sessions, total_seconds


def get_visit_stats_by_period(face_id, db_session, start_date=None, end_date=None):
    """✅ ИСПРАВЛЕНО: Получить детальную статистику визитов за период."""

    from datetime import timedelta

    face = db_session.query(Face).filter(Face.id == face_id).first()
    if not face:
        return 0, 0, 0, {}

    if start_date and start_date.tzinfo is None:
        start_date = pytz.timezone("Asia/Bishkek").localize(start_date).replace(tzinfo=None)
    if end_date and end_date.tzinfo is None:
        end_date = pytz.timezone("Asia/Bishkek").localize(end_date).replace(tzinfo=None)

    visits = face.visits
    if start_date:
        visits = [v for v in visits if v.timestamp >= start_date]
    if end_date:
        visits = [v for v in visits if v.timestamp <= end_date]

    if not visits:
        return 0, 0, 0, {}

    from collections import defaultdict

    visits_by_day = defaultdict(list)
    for v in visits:
        day = v.timestamp.date()
        visits_by_day[day].append(v)

    day_stats = {}
    total_sessions = 0
    total_minutes = 0

    for day, day_visits in visits_by_day.items():
        sessions, seconds = group_visits_to_sessions(day_visits)
        minutes = seconds / 60

        day_stats[day] = {
            "sessions": len(sessions),
            "minutes": round(minutes, 1),
            "first_seen": min(v.timestamp for v in day_visits),
            "last_seen": max(v.timestamp for v in day_visits),
            "real_seconds": seconds,
        }
        total_sessions += len(sessions)
        total_minutes += minutes

    return total_sessions, round(total_minutes, 1), len(visits_by_day), day_stats


def get_visit_stats(face_id, db_session, period_days=None):
    """✅ ИСПРАВЛЕНО: Получить статистику посещений для лица."""

    from datetime import timedelta

    face = db_session.query(Face).filter(Face.id == face_id).first()
    if not face:
        return 0, 0

    now = get_bishkek_time()

    if period_days:
        start_date = now - timedelta(days=period_days)
        visits = [v for v in face.visits if v.timestamp >= start_date]
    else:
        visits = face.visits

    if not visits:
        return 0, 0

    sessions, total_seconds = group_visits_to_sessions(visits)
    total_minutes = total_seconds / 60

    return len(sessions), round(total_minutes, 1)
