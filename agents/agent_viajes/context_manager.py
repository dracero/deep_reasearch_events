import sqlite3
import logging
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

class SQLiteContextManager:
    """Manages synthesized context storage in SQLite to prevent LLM overflow for Travel Agent."""
    
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
                        origin TEXT,
                        destination TEXT,
                        travel_dates TEXT,
                        content TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(origin, destination, travel_dates)
                    )
                """)
                conn.commit()
                logger.info(f"SQLite Context DB initialized at {self.db_path}")
        except Exception as e:
            logger.error(f"Failed to initialize SQLite DB: {e}")

    def get_synthesized_context(self, origin: str, destination: str, travel_dates: str) -> Optional[str]:
        """Retrieve synthesized context for a given route and dates."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT content FROM synthesized_context WHERE origin = ? AND destination = ? AND travel_dates = ?",
                    (origin, destination, travel_dates)
                )
                result = cursor.fetchone()
                if result:
                    logger.info(f"✅ Found synthesized context in DB for {origin} -> {destination} on {travel_dates}")
                    return result[0]
        except Exception as e:
            logger.error(f"Error reading from SQLite: {e}")
        return None

    def save_synthesized_context(self, origin: str, destination: str, travel_dates: str, content: str):
        """Save or update synthesized context."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO synthesized_context (origin, destination, travel_dates, content)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(origin, destination, travel_dates) DO UPDATE SET
                        content = excluded.content,
                        created_at = CURRENT_TIMESTAMP
                """, (origin, destination, travel_dates, content))
                conn.commit()
                logger.info(f"💾 Saved synthesized context to DB for {origin} -> {destination} on {travel_dates}")
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
