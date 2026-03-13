"""Админ-панель для управления системой распознавания лиц
Минималистичный дизайн + Аналитика + Управление сотрудниками
"""

from flask import Flask, render_template_string, request, jsonify, redirect, url_for
from sqlalchemy import func, and_, extract
from datetime import datetime, timedelta
from db import get_session
from models import Face, Visit, get_bishkek_time
import logging

logger = logging.getLogger(__name__)

ADMIN_TEMPLATE = """
<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Face Recognition - Admin Panel</title>
    <style>
    </style>
</head>
<body>
    <div class="header">
        <h1>🎯 Face Recognition - Admin Panel</h1>
    </div>
    <div class="container">
        <!-- Статистика -->
        <div class="stats-grid" id="stats-grid">
            <div class="stat-card">
                <div class="stat-value" id="stat-today">0</div>
                <div class="stat-label">Сегодня</div>
            </div>
            <div class="stat-card success">
                <div class="stat-value" id="stat-week">0</div>
                <div class="stat-label">За неделю</div>
            </div>
            <div class="stat-card warning">
                <div class="stat-value" id="stat-month">0</div>
                <div class="stat-label">За месяц</div>
            </div>
            <div class="stat-card danger">
                <div class="stat-value" id="stat-employees">0</div>
                <div class="stat-label">Сотрудники (игнор.)</div>
            </div>
        </div>

        <!-- Календарь и фильтр -->
        <div class="section">
            <h2 class="section-title">Календарь и фильтр посещений</h2>
            <div style="margin-bottom:1em;">
                <input type="date" id="date-start">
                <input type="date" id="date-end">
                <button class="btn btn-primary btn-sm" onclick="setPeriod('day')">День</button>
                <button class="btn btn-primary btn-sm" onclick="setPeriod('week')">Неделя</button>
                <button class="btn btn-primary btn-sm" onclick="setPeriod('month')">Месяц</button>
                <button class="btn btn-success btn-sm" onclick="loadVisitStats()">Показать</button>
            </div>
            <div id="calendar-block">
                <div style="color:#95a5a6;">Загрузка...</div>
            </div>
        </div>

        <!-- Управление людьми (оставить как есть, можно доработать позже) -->
        <!-- ... (rest of template unchanged for brevity) ... -->
    </div>
    <script>
        function setPeriod(period) {
            const now = new Date();
            let start, end;
            if (period === 'day') {
                start = new Date(now.getFullYear(), now.getMonth(), now.getDate());
                end = new Date(start);
            } else if (period === 'week') {
                end = new Date(now.getFullYear(), now.getMonth(), now.getDate());
                start = new Date(end);
                start.setDate(end.getDate() - 6);
            } else if (period === 'month') {
                end = new Date(now.getFullYear(), now.getMonth(), now.getDate());
                start = new Date(end);
                start.setDate(end.getDate() - 29);
            }
            document.getElementById('date-start').value = start.toISOString().slice(0,10);
            document.getElementById('date-end').value = end.toISOString().slice(0,10);
        }

        async function loadVisitStats() {
            const start = document.getElementById('date-start').value;
            const end = document.getElementById('date-end').value;
            const block = document.getElementById('calendar-block');
            if (!start || !end) {
                block.innerHTML = '<div style="color:#e74c3c;">Выберите даты</div>';
                return;
            }
            block.innerHTML = '<div style="color:#95a5a6;">Загрузка...</div>';
            try {
                const resp = await fetch(`/admin/visit_stats?start_date=${start}&end_date=${end}`);
                if (!resp.ok) throw new Error('Ошибка загрузки статистики');
                const data = await resp.json();
                let html = `<div style="margin-bottom:1em;">
                    <b>Уникальных пользователей:</b> ${data.unique_users} &nbsp; 
                    <b>Всего визитов:</b> ${data.total_visits}
                </div>`;
                html += '<table style="width:100%;border-collapse:collapse;"><thead><tr><th>Person ID</th><th>Визитов</th><th>Последний визит</th></tr></thead><tbody>';
                for (const user of data.users) {
                    html += `<tr>
                        <td>${user.person_id ?? user.face_id}</td>
                        <td>${user.visit_count}</td>
                        <td>${user.last_visit ? new Date(user.last_visit).toLocaleString() : ''}</td>
                    </tr>`;
                }
                html += '</tbody></table>';
                block.innerHTML = html;
            } catch (e) {
                block.innerHTML = '<div style="color:#e74c3c;">Ошибка загрузки статистики</div>';
            }
        }

        window.addEventListener('DOMContentLoaded', () => {
            setPeriod('month');
            loadVisitStats();
        });
    </script>
</body>
</html>
"""

from flask import Blueprint

stats_api = Blueprint('stats_api', __name__)

@stats_api.route('/admin/visit_stats', methods=['GET'])
def visit_stats():
    """
    API: Получить статистику посещений за произвольный период.
    Параметры: start_date, end_date (YYYY-MM-DD)
    """
    from models import Face, Visit, group_visits_to_sessions
    db_session = get_session()
    try:
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        if not start_date or not end_date:
            return jsonify({'error': 'start_date and end_date required'}), 400
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=1)

        visits = db_session.query(Visit).filter(Visit.timestamp >= start, Visit.timestamp < end).all()

        from collections import defaultdict
        visits_by_face = defaultdict(list)
        for v in visits:
            visits_by_face[v.face_id].append(v)

        users = []
        for face_id, vlist in visits_by_face.items():
            visits_by_day = defaultdict(list)
            for v in vlist:
                visits_by_day[v.timestamp.date()].append(v)
            total_sessions = 0
            last_visit = None
            for day, day_visits in visits_by_day.items():
                sessions, _ = group_visits_to_sessions(day_visits)
                total_sessions += len(sessions)
                day_last = max(v.timestamp for v in day_visits)
                if not last_visit or day_last > last_visit:
                    last_visit = day_last
            face = db_session.query(Face).filter(Face.id == face_id).first()
            users.append({
                "face_id": face_id,
                "person_id": face.person_id if face else None,
                "status": face.status if face else None,
                "visit_count": total_sessions,
                "last_visit": last_visit.isoformat() if last_visit else None,
                "photo_base64": (face.photo_base64[:100] + "..." if face and face.photo_base64 else None)
            })

        unique_users = len(users)
        total_visits = sum(u["visit_count"] for u in users)

        return jsonify({
            "unique_users": unique_users,
            "total_visits": total_visits,
            "users": users,
        })
    finally:
        db_session.close()

admin_app = Flask(__name__)
admin_app.secret_key = "face-recognition-admin-2025"
admin_app.register_blueprint(stats_api)

