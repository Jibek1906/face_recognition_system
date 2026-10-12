"""
Database Migration: Add Performance Indexes

This script adds composite indexes to improve query performance:
- idx_visits_face_timestamp: For queries filtering by face_id and timestamp
- idx_faces_status_lastseen: For queries filtering by status and ordering by last_seen
"""

import sys
import logging
from sqlalchemy import create_engine, text, inspect
from face_app.core.config.env import DATABASE_URL

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_index_exists(engine, table_name, index_name):
    """Check if an index already exists"""
    inspector = inspect(engine)
    indexes = inspector.get_indexes(table_name)
    return any(idx['name'] == index_name for idx in indexes)

def add_performance_indexes():
    """Add performance indexes to the database"""
    engine = create_engine(DATABASE_URL)
    
    indexes_to_create = [
        {
            'table': 'visits',
            'name': 'idx_visits_face_timestamp',
            'sql': 'CREATE INDEX IF NOT EXISTS idx_visits_face_timestamp ON visits(face_id, timestamp)'
        },
        {
            'table': 'faces',
            'name': 'idx_faces_status_lastseen',
            'sql': 'CREATE INDEX IF NOT EXISTS idx_faces_status_lastseen ON faces(status, last_seen)'
        }
    ]
    
    with engine.begin() as conn:
        for idx in indexes_to_create:
            try:
                if check_index_exists(engine, idx['table'], idx['name']):
                    logger.info(f"✓ Index {idx['name']} already exists")
                else:
                    logger.info(f"Creating index {idx['name']} on {idx['table']}...")
                    conn.execute(text(idx['sql']))
                    logger.info(f"✓ Index {idx['name']} created successfully")
            except Exception as e:
                logger.error(f"Error creating index {idx['name']}: {e}")
                raise
    
    # Verify indexes
    logger.info("\n=== Verifying Indexes ===")
    with engine.connect() as conn:
        for table in ['visits', 'faces']:
            result = conn.execute(text(f"""
                SELECT indexname, indexdef 
                FROM pg_indexes 
                WHERE tablename = '{table}' 
                AND indexname LIKE 'idx_%'
                ORDER BY indexname
            """))
            logger.info(f"\n{table} indexes:")
            for row in result:
                logger.info(f"  - {row[0]}")
    
    logger.info("\n✅ Migration complete!")

if __name__ == "__main__":
    try:
        add_performance_indexes()
    except Exception as e:
        logger.error(f"Migration failed: {e}", exc_info=True)
        sys.exit(1)