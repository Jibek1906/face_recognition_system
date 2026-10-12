# init_db.py
from face_app.db.session import engine
from face_app.db.models import Base

def create_tables():
    Base.metadata.create_all(bind=engine)
    print("✅ Все таблицы созданы")

if __name__ == "__main__":
    create_tables()
