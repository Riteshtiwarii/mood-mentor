"""
Mood Mentor — SQLite Persistent Storage Manager
Handles interactions, recommendations, feedback telemetry, and data privacy operations.
"""

import sqlite3
import os
import json
from datetime import datetime
from typing import List, Dict, Any, Optional

DEFAULT_DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "moodmentor.db")

class DatabaseManager:
    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        self.db_path = db_path
        self._memory_conn = None
        if self.db_path == ":memory:":
            self._memory_conn = sqlite3.connect(":memory:", check_same_thread=False)
            self._memory_conn.row_factory = sqlite3.Row
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        if self._memory_conn is not None:
            return self._memory_conn
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        dir_name = os.path.dirname(self.db_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # 1. Interactions Table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS interactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                raw_text TEXT NOT NULL,
                primary_emotion TEXT NOT NULL,
                confidence REAL NOT NULL,
                intensity_score REAL NOT NULL,
                tier TEXT NOT NULL,
                triage_level TEXT NOT NULL,
                vader_compound REAL NOT NULL
            )
            """)

            # 2. Recommendations Table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS recommendations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                interaction_id INTEGER,
                user_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                activity_id TEXT NOT NULL,
                title TEXT NOT NULL,
                modality TEXT NOT NULL,
                score REAL NOT NULL,
                rationale TEXT NOT NULL,
                FOREIGN KEY (interaction_id) REFERENCES interactions(id) ON DELETE CASCADE
            )
            """)

            # 3. Feedback Table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                activity_id TEXT NOT NULL,
                activity_title TEXT,
                action TEXT NOT NULL,  -- viewed, accepted, rejected
                rating REAL DEFAULT 3.0,
                comment TEXT,
                timestamp TEXT NOT NULL
            )
            """)

            # 4. User Preferences Table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_preferences (
                user_id TEXT PRIMARY KEY,
                preferred_modalities TEXT,
                time_limit_mins INTEGER DEFAULT 15,
                updated_at TEXT NOT NULL
            )
            """)
            conn.commit()

    def save_interaction(self, user_id: str, raw_text: str, primary_emotion: str,
                         confidence: float, intensity_score: float, tier: str,
                         triage_level: str, vader_compound: float = 0.0,
                         timestamp: Optional[str] = None) -> int:
        ts = timestamp or datetime.now().isoformat()
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            INSERT INTO interactions (user_id, timestamp, raw_text, primary_emotion,
                                      confidence, intensity_score, tier, triage_level, vader_compound)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (user_id, ts, raw_text, primary_emotion, confidence, intensity_score, tier, triage_level, vader_compound))
            conn.commit()
            return cursor.lastrowid

    def save_recommendations(self, interaction_id: int, user_id: str,
                             recommendations: List[Dict[str, Any]],
                             timestamp: Optional[str] = None):
        ts = timestamp or datetime.now().isoformat()
        with self._get_connection() as conn:
            cursor = conn.cursor()
            for rec in recommendations:
                cursor.execute("""
                INSERT INTO recommendations (interaction_id, user_id, timestamp, activity_id,
                                             title, modality, score, rationale)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (interaction_id, user_id, ts, rec["id"], rec["title"],
                      rec["modality"], rec["score"], rec.get("rationale", "")))
            conn.commit()

    def save_feedback(self, user_id: str, activity_id: str, activity_title: str,
                      action: str, rating: float = 3.0, comment: str = "",
                      timestamp: Optional[str] = None) -> int:
        ts = timestamp or datetime.now().isoformat()
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            INSERT INTO feedback (user_id, activity_id, activity_title, action, rating, comment, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (user_id, activity_id, activity_title, action, rating, comment, ts))
            conn.commit()
            return cursor.lastrowid

    def get_user_interactions(self, user_id: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            if user_id:
                cursor.execute("SELECT * FROM interactions WHERE user_id = ? ORDER BY timestamp DESC LIMIT ?", (user_id, limit))
            else:
                cursor.execute("SELECT * FROM interactions ORDER BY timestamp DESC LIMIT ?", (limit,))
            return [dict(row) for row in cursor.fetchall()]

    def get_user_recommendations(self, user_id: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            if user_id:
                cursor.execute("SELECT * FROM recommendations WHERE user_id = ? ORDER BY timestamp DESC LIMIT ?", (user_id, limit))
            else:
                cursor.execute("SELECT * FROM recommendations ORDER BY timestamp DESC LIMIT ?", (limit,))
            return [dict(row) for row in cursor.fetchall()]

    def get_user_feedback(self, user_id: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            if user_id:
                cursor.execute("SELECT * FROM feedback WHERE user_id = ? ORDER BY timestamp DESC LIMIT ?", (user_id, limit))
            else:
                cursor.execute("SELECT * FROM feedback ORDER BY timestamp DESC LIMIT ?", (limit,))
            return [dict(row) for row in cursor.fetchall()]

    def purge_user_data(self, user_id: str) -> Dict[str, int]:
        """GDPR Right-to-be-Forgotten data deletion"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM recommendations WHERE user_id = ?", (user_id,))
            recs_deleted = cursor.rowcount
            cursor.execute("DELETE FROM interactions WHERE user_id = ?", (user_id,))
            ints_deleted = cursor.rowcount
            cursor.execute("DELETE FROM feedback WHERE user_id = ?", (user_id,))
            feedbacks_deleted = cursor.rowcount
            cursor.execute("DELETE FROM user_preferences WHERE user_id = ?", (user_id,))
            prefs_deleted = cursor.rowcount
            conn.commit()

        return {
            "interactions_deleted": ints_deleted,
            "recommendations_deleted": recs_deleted,
            "feedback_deleted": feedbacks_deleted,
            "preferences_deleted": prefs_deleted
        }

    def get_database_stats(self) -> Dict[str, int]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM interactions")
            c_int = cursor.fetchone()[0]
            cursor.execute("SELECT COUNT(*) FROM recommendations")
            c_rec = cursor.fetchone()[0]
            cursor.execute("SELECT COUNT(*) FROM feedback")
            c_fb = cursor.fetchone()[0]
            return {
                "total_interactions": c_int,
                "total_recommendations": c_rec,
                "total_feedback": c_fb
            }
