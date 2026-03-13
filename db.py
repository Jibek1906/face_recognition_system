from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy import text
from config import DATABASE_URL
import logging


logger = logging.getLogger(__name__)


engine = create_engine(
   DATABASE_URL,
   pool_size=10,
   max_overflow=20,
   pool_pre_ping=True,
   echo=False
)


SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)




def get_db() -> Session:
   db = SessionLocal()
   try:
       yield db
   finally:
       db.close()




def init_db():
   """Инициализация базы данных с автоматической миграцией"""
   try:
       from models import Base
      
       Base.metadata.create_all(bind=engine)
      
       with engine.begin() as conn:
           result = conn.execute(
               text("SELECT column_name FROM information_schema.columns WHERE table_name='faces' AND column_name='person_id';")
           )
           if result.fetchone() is None:
               logger.info("Migrating: Adding person_id column...")
               conn.execute(text("ALTER TABLE faces ADD COLUMN person_id INTEGER;"))
               conn.execute(text("UPDATE faces SET person_id = id WHERE person_id IS NULL;"))
               conn.execute(text("CREATE INDEX IF NOT EXISTS idx_faces_person_id ON faces(person_id);"))
               logger.info("✓ Added person_id column")
          
           result = conn.execute(
               text("SELECT column_name FROM information_schema.columns WHERE table_name='faces' AND column_name='is_employee';")
           )
           if result.fetchone() is None:
               logger.info("Migrating: Adding is_employee column...")
               conn.execute(text("ALTER TABLE faces ADD COLUMN is_employee BOOLEAN DEFAULT FALSE;"))
               conn.execute(text("UPDATE faces SET is_employee = FALSE WHERE is_employee IS NULL;"))
               conn.execute(text("CREATE INDEX IF NOT EXISTS idx_faces_is_employee ON faces(is_employee);"))
               logger.info("✓ Added is_employee column")
          
           result = conn.execute(
               text("SELECT column_name FROM information_schema.columns WHERE table_name='faces' AND column_name='employee_name';")
           )
           if result.fetchone() is None:
               logger.info("Migrating: Adding employee_name column...")
               conn.execute(text("ALTER TABLE faces ADD COLUMN employee_name VARCHAR(200);"))
               logger.info("✓ Added employee_name column")
          
           result = conn.execute(
               text("SELECT column_name FROM information_schema.columns WHERE table_name='faces' AND column_name='employee_position';")
           )
           if result.fetchone() is None:
               logger.info("Migrating: Adding employee_position column...")
               conn.execute(text("ALTER TABLE faces ADD COLUMN employee_position VARCHAR(200);"))
               logger.info("✓ Added employee_position column")


           result = conn.execute(
               text("SELECT column_name FROM information_schema.columns WHERE table_name='daily_stats' AND column_name='face_id';")
           )
           if result.fetchone() is None:
               logger.info("Migrating: Adding face_id column to daily_stats...")
               conn.execute(text("ALTER TABLE daily_stats ADD COLUMN face_id INTEGER REFERENCES faces(id);"))
               conn.execute(text("CREATE INDEX IF NOT EXISTS idx_daily_stats_face_id ON daily_stats(face_id);"))
               logger.info("✓ Added face_id column to daily_stats")


           result = conn.execute(
               text("SELECT column_name FROM information_schema.columns WHERE table_name='daily_stats' AND column_name='visit_count';")
           )
           if result.fetchone() is None:
               logger.info("Migrating: Adding visit_count column to daily_stats...")
               conn.execute(text("ALTER TABLE daily_stats ADD COLUMN visit_count INTEGER DEFAULT 0;"))
               logger.info("✓ Added visit_count column to daily_stats")


           result = conn.execute(
               text("SELECT column_name FROM information_schema.columns WHERE table_name='daily_stats' AND column_name='total_minutes';")
           )
           if result.fetchone() is None:
               logger.info("Migrating: Adding total_minutes column to daily_stats...")
               conn.execute(text("ALTER TABLE daily_stats ADD COLUMN total_minutes FLOAT DEFAULT 0.0;"))
               logger.info("✓ Added total_minutes column to daily_stats")
          
           # ✅ NEW: Body Embedding for Face table
           result = conn.execute(
               text("SELECT column_name FROM information_schema.columns WHERE table_name='faces' AND column_name='body_embedding';")
           )
           if result.fetchone() is None:
               logger.info("Migrating: Adding body_embedding column to faces...")
               conn.execute(text("ALTER TABLE faces ADD COLUMN body_embedding TEXT;"))
               logger.info("✓ Added body_embedding column to faces")
          
           # Connection auto-commits on exit of engine.begin() block
      
       logger.info("✓ Database tables initialized and migrated successfully")
      
   except Exception as e:
       logger.error(f"✗ Failed to initialize database: {e}")
       raise




def get_session() -> Session:
   return SessionLocal()


def reset_visits_table():
   """Удалить и пересоздать таблицу visits"""
   from models import Visit, Base
   try:
       Visit.__table__.drop(bind=engine, checkfirst=True)
       Visit.__table__.create(bind=engine, checkfirst=True)
       logger.info("✓ visits table dropped and recreated")
   except Exception as e:
       logger.error(f"✗ Failed to reset visits table: {e}")


if __name__ == "__main__":
   print("Resetting visits table...")
   reset_visits_table()
   print("Done.")