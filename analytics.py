import threading
import time
from visit_manager import VisitState

class Analytics:
    def __init__(self, visit_manager):
        self.visit_manager = visit_manager
        self.lock = threading.Lock()
        self.closed_visits = []  # Сохраняем завершённые визиты для аналитики

    def record_closed_visit(self, visit):
        with self.lock:
            self.closed_visits.append({
                "visit_id": visit.visit_id,
                "person_id": visit.person_id,
                "start_time": visit.start_time,
                "end_time": visit.last_seen,
                "duration": visit.last_seen - visit.start_time,
            })

    def get_active_count(self):
        return len(self.visit_manager.get_active_visits())

    def get_total_visits(self):
        with self.lock:
            return len(self.closed_visits)

    def get_unique_persons(self):
        with self.lock:
            return len(set(v["person_id"] for v in self.closed_visits))

    def get_visits_by_time(self, since_ts):
        with self.lock:
            return [
                v for v in self.closed_visits
                if v["end_time"] >= since_ts
            ]

    def get_stats(self):
        with self.lock:
            total = len(self.closed_visits)
            unique = len(set(v["person_id"] for v in self.closed_visits))
            durations = [v["duration"] for v in self.closed_visits]
            avg_duration = sum(durations) / total if total else 0
            return {
                "total_visits": total,
                "unique_persons": unique,
                "avg_duration": avg_duration,
            }
