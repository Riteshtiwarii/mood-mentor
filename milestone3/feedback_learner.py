"""Recommendation Feedback Learning & Online Adaptation Module (Milestone 3 - Task 7).

Captures recommendation views, acceptances, rejections, and ratings to dynamically
update user preference weights and item affinity in real time.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from config import (
    FEEDBACK_ACCEPT_BOOST,
    FEEDBACK_FILE_PATH,
    FEEDBACK_REJECT_PENALTY,
)
from exceptions import FeedbackError
from user_profile import UserProfile, UserProfileRegistry
from wellness_catalog import WellnessCatalogManager, get_wellness_catalog

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FeedbackRecord:
    """Immutable data model for user telemetry and feedback interactions."""

    user_id: str
    item_id: str
    action: str  # 'viewed', 'accepted', 'rejected'
    rating: Optional[float]
    timestamp: str
    modality: Optional[str] = None
    category: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "user_id": self.user_id,
            "item_id": self.item_id,
            "action": self.action,
            "rating": self.rating,
            "timestamp": self.timestamp,
            "modality": self.modality,
            "category": self.category,
        }


class FeedbackLearner:
    """Manages feedback loops, storage, and online preference weight adaptation."""

    def __init__(
        self,
        feedback_file: Path = FEEDBACK_FILE_PATH,
        catalog_manager: WellnessCatalogManager | None = None,
    ) -> None:
        self.feedback_file = feedback_file
        self.catalog = catalog_manager or get_wellness_catalog()
        self._history: List[FeedbackRecord] = []
        self._load_feedback_history()

    def _load_feedback_history(self) -> None:
        """Load stored feedback logs from disk."""
        if not self.feedback_file.exists():
            logger.debug("No existing feedback file found at %s. Starting fresh.", self.feedback_file)
            return

        try:
            with self.feedback_file.open("r", encoding="utf-8") as f:
                data = json.load(f)

            self._history.clear()
            for entry in data:
                record = FeedbackRecord(
                    user_id=entry["user_id"],
                    item_id=entry["item_id"],
                    action=entry["action"],
                    rating=entry.get("rating"),
                    timestamp=entry.get("timestamp", datetime.now().isoformat()),
                    modality=entry.get("modality"),
                    category=entry.get("category"),
                )
                self._history.append(record)
            logger.info("Loaded %d historical feedback records from disk.", len(self._history))
        except Exception as exc:
            logger.warning("Failed to load feedback history: %s", exc)

    def _save_feedback_history(self) -> None:
        """Persist feedback logs to disk."""
        try:
            self.feedback_file.parent.mkdir(parents=True, exist_ok=True)
            with self.feedback_file.open("w", encoding="utf-8") as f:
                json.dump([r.to_dict() for r in self._history], f, indent=2)
        except Exception as exc:
            raise FeedbackError(f"Failed to persist feedback record to {self.feedback_file}: {exc}") from exc

    def record_feedback(
        self,
        user_profile: UserProfile,
        item_id: str,
        action: str,
        rating: Optional[float] = None,
    ) -> FeedbackRecord:
        """Process user feedback action and adapt user weights dynamically (Task 7).

        Args:
            user_profile: The active user profile being updated.
            item_id: ID of the wellness intervention.
            action: 'viewed', 'accepted', or 'rejected'.
            rating: Optional user satisfaction rating (1.0 to 5.0).
        """
        clean_action = action.lower().strip()
        if clean_action not in ("viewed", "accepted", "rejected"):
            raise FeedbackError(
                f"Invalid action '{action}'. Must be 'viewed', 'accepted', or 'rejected'."
            )

        item = self.catalog.get_by_id(item_id)
        modality = item.modality if item else None
        category = item.category if item else None

        record = FeedbackRecord(
            user_id=user_profile.user_id,
            item_id=item_id,
            action=clean_action,
            rating=float(rating) if rating is not None else None,
            timestamp=datetime.now().isoformat(),
            modality=modality,
            category=category,
        )

        self._history.append(record)
        self._save_feedback_history()

        # Online Learning: Update User Preference & Modality Weights
        if modality:
            current_weight = user_profile.modality_weights.get(modality, 1.0)

            if clean_action == "accepted":
                # Boost preferred modality
                boost = FEEDBACK_ACCEPT_BOOST
                if rating and rating >= 4.0:
                    boost += 0.10
                new_weight = min(current_weight + boost, 1.8)
                user_profile.modality_weights[modality] = new_weight
                user_profile.preferred_modalities.add(modality)
                logger.info(
                    "User '%s' accepted %s -> Modality '%s' boosted: %.2f -> %.2f",
                    user_profile.user_id, item_id, modality, current_weight, new_weight
                )

            elif clean_action == "rejected":
                # Dampen rejected modality
                penalty = FEEDBACK_REJECT_PENALTY
                new_weight = max(current_weight - penalty, 0.2)
                user_profile.modality_weights[modality] = new_weight
                logger.info(
                    "User '%s' rejected %s -> Modality '%s' dampened: %.2f -> %.2f",
                    user_profile.user_id, item_id, modality, current_weight, new_weight
                )

        # Log into profile interaction history if rating or completed
        if clean_action == "accepted":
            user_profile.record_interaction(
                item_id=item_id,
                rating=rating if rating is not None else 4.5,
                completed=True,
            )

        return record

    def compute_feedback_modifier(self, user_id: str, item_id: str) -> float:
        """Compute multiplier penalty or boost based on user's past feedback on this item."""
        user_records = [r for r in self._history if r.user_id == user_id and r.item_id == item_id]
        if not user_records:
            return 0.0

        last_record = user_records[-1]
        if last_record.action == "rejected":
            return -0.25  # Substantial penalty for explicitly rejected items
        elif last_record.action == "accepted":
            return 0.15   # Boost for previously accepted items
        return 0.0

    def get_user_feedback_summary(self, user_id: str) -> dict[str, Any]:
        """Return analytics summary of user's feedback activity."""
        records = [r for r in self._history if r.user_id == user_id]
        accepted = [r for r in records if r.action == "accepted"]
        rejected = [r for r in records if r.action == "rejected"]
        ratings = [r.rating for r in records if r.rating is not None]

        return {
            "user_id": user_id,
            "total_feedback_events": len(records),
            "accepted_count": len(accepted),
            "rejected_count": len(rejected),
            "avg_rating": round(sum(ratings) / len(ratings), 2) if ratings else None,
            "acceptance_rate_pct": round((len(accepted) / len(records) * 100.0), 1) if records else 0.0,
        }
