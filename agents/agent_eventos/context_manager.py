import sqlite3
import logging
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

class SQLiteContextManager:
    """Manages synthesized context storage in SQLite to prevent LLM overflow."""
    
    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            # Store in the same directory as the agent
            base_dir = Path(__file__).parent
            self.db_path = base_dir / "context_cache.db"
        else:
            self.db_path = Path(db_path)
        
        self.init_db()

    def init_db(self):
        """Initialize the SQLite database and tables."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS synthesized_context (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        category TEXT,
                        target_date TEXT,
                        content TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(category, target_date)
                    )
                """)
                conn.commit()
                logger.info(f"SQLite Context DB initialized at {self.db_path}")
        except Exception as e:
            logger.error(f"Failed to initialize SQLite DB: {e}")

    def get_synthesized_context(self, category: str, target_date: str) -> Optional[str]:
        """Retrieve synthesized context for a given category and date."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT content FROM synthesized_context WHERE category = ? AND target_date = ?",
                    (category, target_date)
                )
                result = cursor.fetchone()
                if result:
                    logger.info(f"✅ Found synthesized context in DB for {category} on {target_date}")
                    return result[0]
        except Exception as e:
            logger.error(f"Error reading from SQLite: {e}")
        return None

    def save_synthesized_context(self, category: str, target_date: str, content: str):
        """Save or update synthesized context."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO synthesized_context (category, target_date, content)
                    VALUES (?, ?, ?)
                    ON CONFLICT(category, target_date) DO UPDATE SET
                        content = excluded.content,
                        created_at = CURRENT_TIMESTAMP
                """, (category, target_date, content))
                conn.commit()
                logger.info(f"💾 Saved synthesized context to DB for {category} on {target_date}")
        except Exception as e:
            logger.error(f"Error saving to SQLite: {e}")

    def clear_all_context(self):
        """Delete all cached synthesized contexts."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM synthesized_context")
                conn.commit()
                logger.info("🧹 SQLite Context Cache cleared.")
        except Exception as e:
            logger.error(f"Error clearing SQLite: {e}")
