from face_app.db.session import SessionLocal
from face_app.db.models import Face, PersonPhoto, Visit

def clear_database():
    db = SessionLocal()
    try:
        # Удаляем все записи из таблиц
        db.query(Visit).delete()
        db.query(PersonPhoto).delete()
        db.query(Face).delete()

        # Сброс счетчиков ID (работает для PostgreSQL)
        from sqlalchemy import text
        db.execute(text("TRUNCATE TABLE visits RESTART IDENTITY CASCADE;"))
        db.execute(text("TRUNCATE TABLE person_photos RESTART IDENTITY CASCADE;"))
        db.execute(text("TRUNCATE TABLE faces RESTART IDENTITY CASCADE;"))

        db.commit()
        print("База данных успешно очищена!")
    except Exception as e:
        db.rollback()
        print(f"Ошибка при очистке: {e}")
    finally:
        db.close()

if __name__ == "__main__":
    clear_database()
