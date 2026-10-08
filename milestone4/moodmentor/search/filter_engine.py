"""
Mood Mentor — Advanced Search & Filter Engine
Performs parameterized multi-criteria searching and filtering across interactions, recommendations, and wellness content.
"""

from typing import List, Dict, Any, Optional
from datetime import datetime
from moodmentor.storage.db import DatabaseManager

class FilterEngine:
    def __init__(self, db: Optional[DatabaseManager] = None):
        self.db = db

    def search_interactions(self, db: DatabaseManager,
                            user_id: Optional[str] = None,
                            keyword: Optional[str] = None,
                            emotion: Optional[str] = None,
                            tier: Optional[str] = None,
                            triage_level: Optional[str] = None,
                            from_date: Optional[str] = None,
                            to_date: Optional[str] = None,
                            limit: int = 100) -> List[Dict[str, Any]]:
        """
        Filters user interactions by date range, emotion, intensity tier, triage, and text keyword.
        """
        query = "SELECT * FROM interactions WHERE 1=1"
        params: List[Any] = []

        if user_id:
            query += " AND user_id = ?"
            params.append(user_id)

        if keyword:
            query += " AND raw_text LIKE ?"
            params.append(f"%{keyword}%")

        if emotion and emotion.lower() != "all":
            query += " AND LOWER(primary_emotion) = ?"
            params.append(emotion.lower())

        if tier and tier.lower() != "all":
            query += " AND LOWER(tier) = ?"
            params.append(tier.lower())

        if triage_level and triage_level.lower() != "all":
            query += " AND LOWER(triage_level) = ?"
            params.append(triage_level.lower())

        if from_date:
            query += " AND timestamp >= ?"
            params.append(from_date)

        if to_date:
            query += " AND timestamp <= ?"
            params.append(to_date)

        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        with db._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, tuple(params))
            return [dict(row) for row in cursor.fetchall()]

    def search_recommendations(self, db: DatabaseManager,
                               user_id: Optional[str] = None,
                               modality: Optional[str] = None,
                               min_score: float = 0.0,
                               keyword: Optional[str] = None,
                               limit: int = 100) -> List[Dict[str, Any]]:
        query = "SELECT * FROM recommendations WHERE score >= ?"
        params: List[Any] = [min_score]

        if user_id:
            query += " AND user_id = ?"
            params.append(user_id)

        if modality and modality.lower() != "all":
            query += " AND LOWER(modality) = ?"
            params.append(modality.lower())

        if keyword:
            query += " AND (title LIKE ? OR rationale LIKE ?)"
            params.extend([f"%{keyword}%", f"%{keyword}%"])

        query += " ORDER BY timestamp DESC, score DESC LIMIT ?"
        params.append(limit)

        with db._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, tuple(params))
            return [dict(row) for row in cursor.fetchall()]

    def search_feedback(self, db: DatabaseManager,
                        user_id: Optional[str] = None,
                        action: Optional[str] = None,
                        min_rating: float = 1.0,
                        limit: int = 100) -> List[Dict[str, Any]]:
        query = "SELECT * FROM feedback WHERE rating >= ?"
        params: List[Any] = [min_rating]

        if user_id:
            query += " AND user_id = ?"
            params.append(user_id)

        if action and action.lower() != "all":
            query += " AND LOWER(action) = ?"
            params.append(action.lower())

        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        with db._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, tuple(params))
            return [dict(row) for row in cursor.fetchall()]
