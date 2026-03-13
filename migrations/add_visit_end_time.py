"""
Database migration: Add end_time column to visits table

This migration adds the end_time field to track when a visit session ends.
Previously, each Visit record was just a timestamp. Now each Visit represents
a complete session with start (timestamp) and end (end_time).

Usage:
    cd /Users/zibekakylbekkyzy/Desktop/face_recognition/face_system
    python3 migrations/add_visit_end_time.py
"""
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import text
from db import get_session

def migrate():
    """Add end_time column to visits table"""
    session = get_session()
    try:
        # Add end_time column (nullable, as existing records won't have it)
        session.execute(text("""
            ALTER TABLE visits 
            ADD COLUMN IF NOT EXISTS end_time TIMESTAMP
        """))
        session.commit()
        print("✅ Migration complete: added end_time to visits table")
        print("📊 Existing Visit records will have NULL end_time (treated as legacy)")
    except Exception as e:
        session.rollback()
        print(f"❌ Migration failed: {e}")
        raise
    finally:
        session.close()

if __name__ == "__main__":
    print("🔄 Running migration: add_visit_end_time")
    migrate()
