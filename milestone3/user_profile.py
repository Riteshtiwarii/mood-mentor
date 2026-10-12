"""User Profile, Personalization & Longitudinal Trend Module (Milestone 3 - Task 2).

Manages individual employee preferences, interaction histories, recommendation
logs, and emotional trajectory trend analysis over time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import pandas as pd

from config import DATA_DIR
from exceptions import UserProfileError

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class UserInteraction:
    """Historical record of an employee's engagement with a wellness activity."""

    user_id: str
    item_id: str
    rating: float
    completed: bool
    timestamp: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "user_id": self.user_id,
            "item_id": self.item_id,
            "rating": self.rating,
            "completed": self.completed,
            "timestamp": self.timestamp,
        }


@dataclass(frozen=True)
class MoodSnapshot:
    """Timestamped snapshot of user's past emotional state for longitudinal tracking."""

    dominant_emotion: str
    intensity_score: float
    polarity: str
    severity: str
    timestamp: str


@dataclass(frozen=True)
class EmotionalTrendReport:
    """Detailed longitudinal trend analytics report (Task 6)."""

    user_id: str
    total_snapshots: int
    emotion_frequencies: dict[str, int]
    dominant_emotion_overall: str
    recent_dominant_emotion: str
    avg_recent_intensity: float
    intensity_trajectory_slope: float
    positive_ratio: float
    negative_ratio: float
    repeated_patterns: list[str]
    trend_label: str
    recommendation_bias: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "user_id": self.user_id,
            "total_snapshots": self.total_snapshots,
            "emotion_frequencies": self.emotion_frequencies,
            "dominant_emotion_overall": self.dominant_emotion_overall.capitalize(),
            "recent_dominant_emotion": self.recent_dominant_emotion.capitalize(),
            "avg_recent_intensity": round(self.avg_recent_intensity, 4),
            "intensity_trajectory_slope": round(self.intensity_trajectory_slope, 4),
            "positive_ratio": round(self.positive_ratio, 2),
            "negative_ratio": round(self.negative_ratio, 2),
            "repeated_patterns": self.repeated_patterns,
            "trend_label": self.trend_label,
            "recommendation_bias": {k: round(v, 3) for k, v in self.recommendation_bias.items()},
        }


@dataclass
class UserProfile:
    """Mutable profile storing user preferences, dynamic weights, interaction logs, and trends."""

    user_id: str
    preferred_modalities: Set[str] = field(default_factory=lambda: {"breathing", "mindfulness", "micro_break"})
    modality_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "breathing": 1.0,
            "mindfulness": 1.0,
            "micro_break": 1.0,
            "cognitive": 0.8,
            "exercise": 0.8,
            "journaling": 0.8,
            "audio": 0.8,
            "interactive": 0.7,
            "reading": 0.7,
        }
    )
    max_preferred_duration: int = 15
    interaction_history: List[UserInteraction] = field(default_factory=list)
    recent_recommendation_ids: List[str] = field(default_factory=list)
    mood_history: List[MoodSnapshot] = field(default_factory=list)

    def record_interaction(
        self, item_id: str, rating: float, completed: bool = True
    ) -> None:
        """Log a new interaction rating and completion."""
        interaction = UserInteraction(
            user_id=self.user_id,
            item_id=item_id,
            rating=float(rating),
            completed=completed,
            timestamp=datetime.now().isoformat(),
        )
        self.interaction_history.append(interaction)
        logger.info("Recorded interaction for user '%s' with item '%s' (Rating: %.1f)", self.user_id, item_id, rating)

    def record_recommendation(self, item_id: str) -> None:
        """Track recommendation presentation to prevent repetitive fatigue."""
        self.recent_recommendation_ids.append(item_id)
        if len(self.recent_recommendation_ids) > 15:
            self.recent_recommendation_ids.pop(0)

    def record_mood(
        self, dominant_emotion: str, intensity_score: float, polarity: str, severity: str
    ) -> None:
        """Append emotional state for longitudinal trend analysis (Task 6)."""
        snapshot = MoodSnapshot(
            dominant_emotion=dominant_emotion.lower(),
            intensity_score=intensity_score,
            polarity=polarity,
            severity=severity,
            timestamp=datetime.now().isoformat(),
        )
        self.mood_history.append(snapshot)
        if len(self.mood_history) > 30:
            self.mood_history.pop(0)

    def has_recently_received(self, item_id: str, recency_limit: int = 3) -> bool:
        """Check if the user was recently recommended this activity."""
        recent_window = self.recent_recommendation_ids[-recency_limit:]
        return item_id in recent_window

    def get_trend_report(self) -> EmotionalTrendReport:
        """Compute full longitudinal trend analytics and bias vectors (Task 6)."""
        if not self.mood_history:
            return EmotionalTrendReport(
                user_id=self.user_id,
                total_snapshots=0,
                emotion_frequencies={},
                dominant_emotion_overall="neutral",
                recent_dominant_emotion="neutral",
                avg_recent_intensity=0.3,
                intensity_trajectory_slope=0.0,
                positive_ratio=0.0,
                negative_ratio=0.0,
                repeated_patterns=["No historical baseline"],
                trend_label="Baseline Establishment",
                recommendation_bias={},
            )

        # 1. Emotion Frequency Tracking
        freqs: Dict[str, int] = {}
        for snap in self.mood_history:
            freqs[snap.dominant_emotion] = freqs.get(snap.dominant_emotion, 0) + 1

        overall_dominant = max(freqs.items(), key=lambda x: x[1])[0]

        # 2. Recent Window Analysis (Last 5 snapshots)
        window = self.mood_history[-5:]
        recent_scores = [m.intensity_score for m in window]
        recent_polarities = [m.polarity for m in window]
        recent_dominant = window[-1].dominant_emotion

        avg_recent = sum(recent_scores) / len(recent_scores)
        pos_count = recent_polarities.count("Positive")
        neg_count = recent_polarities.count("Negative")
        total_w = len(recent_polarities)

        pos_ratio = pos_count / total_w
        neg_ratio = neg_count / total_w

        # 3. Trajectory Slope (delta between halves)
        if len(recent_scores) >= 2:
            mid = len(recent_scores) // 2
            h1 = recent_scores[:mid]
            h2 = recent_scores[mid:]
            slope = (sum(h2) / len(h2)) - (sum(h1) / len(h1))
        else:
            slope = 0.0

        # 4. Repeated Patterns Detection
        patterns: List[str] = []
        if neg_count >= 3:
            patterns.append("Chronic Negative Polarity Loop")
        if freqs.get("fear", 0) >= 3:
            patterns.append("Recurrent Anxiety/Panic Vulnerability")
        if freqs.get("anger", 0) >= 3:
            patterns.append("Recurring Workplace Conflict/Frustration")
        if freqs.get("sadness", 0) >= 3:
            patterns.append("Persistent Low Mood & Depressive Lethargy")
        if slope > 0.15:
            patterns.append("Accelerating Emotional Distress Curve")
        elif slope < -0.15:
            patterns.append("De-escalating Stress & Recovery Pattern")

        if not patterns:
            patterns.append("Balanced Emotional Variance")

        # 5. Trend Label Classification & Recommendation Bias
        rec_bias: Dict[str, float] = {}

        if (slope > 0.12 and neg_count >= 2) or (freqs.get("fear", 0) >= 2 and avg_recent >= 0.70):
            trend_label = "Escalating Distress (Rising Stress Pattern)"
            # Boost somatic calming, breathing, and clinical support
            rec_bias["breathing"] = 0.25
            rec_bias["clinical"] = 0.25
            rec_bias["mindfulness"] = 0.20
        elif slope < -0.12 and neg_count <= 1:
            trend_label = "Improving Resilience (De-escalating Trend)"
            rec_bias["cognitive"] = 0.15
            rec_bias["social"] = 0.15
        elif neg_count >= 3:
            trend_label = "Chronic Strain (Sustained Low Mood Pattern)"
            rec_bias["clinical"] = 0.25
            rec_bias["breathing"] = 0.20
            rec_bias["mindfulness"] = 0.15
            rec_bias["social"] = 0.10
            rec_bias["cognitive"] = 0.10
        elif pos_ratio >= 0.60:
            trend_label = "Positive Stability (Flourishing State)"
            rec_bias["social"] = 0.20
            rec_bias["cognitive"] = 0.15
        else:
            trend_label = "Stable Homeostasis"

        return EmotionalTrendReport(
            user_id=self.user_id,
            total_snapshots=len(self.mood_history),
            emotion_frequencies=freqs,
            dominant_emotion_overall=overall_dominant,
            recent_dominant_emotion=recent_dominant,
            avg_recent_intensity=avg_recent,
            intensity_trajectory_slope=slope,
            positive_ratio=pos_ratio,
            negative_ratio=neg_ratio,
            repeated_patterns=patterns,
            trend_label=trend_label,
            recommendation_bias=rec_bias,
        )

    def analyze_emotional_trend(self) -> str:
        """Shorthand accessor for trend label string."""
        return self.get_trend_report().trend_label

    def to_dict(self) -> dict[str, Any]:
        report = self.get_trend_report()
        return {
            "user_id": self.user_id,
            "preferred_modalities": sorted(list(self.preferred_modalities)),
            "max_preferred_duration": self.max_preferred_duration,
            "total_interactions": len(self.interaction_history),
            "recent_recommendation_count": len(self.recent_recommendation_ids),
            "trend_report": report.to_dict(),
            "emotional_trend": self.analyze_emotional_trend(),
        }


class UserProfileRegistry:
    """Manages active user profiles and loads historical interaction matrices."""

    def __init__(self, interaction_csv: Path = DATA_DIR / "user_interactions.csv") -> None:
        self._profiles: Dict[str, UserProfile] = {}
        self.interaction_csv = interaction_csv
        self._load_historical_interactions()

    def _load_historical_interactions(self) -> None:
        """Populate initial profiles from interaction dataset."""
        if not self.interaction_csv.exists():
            logger.debug("No historical interaction CSV found at %s", self.interaction_csv)
            return

        try:
            df = pd.read_csv(self.interaction_csv)
            for _, row in df.iterrows():
                uid = str(row["user_id"]).strip()
                item_id = str(row["item_id"]).strip()
                rating = float(row["rating"])
                completed = bool(row["completed"])
                ts = str(row.get("timestamp", datetime.now().isoformat()))

                profile = self.get_or_create(uid)
                profile.interaction_history.append(
                    UserInteraction(user_id=uid, item_id=item_id, rating=rating, completed=completed, timestamp=ts)
                )
            logger.info("Loaded interactions for %d user profile(s).", len(self._profiles))
        except Exception as exc:
            logger.warning("Failed loading interaction CSV: %s", exc)

    def get_or_create(
        self,
        user_id: str,
        preferred_modalities: Optional[Set[str]] = None,
        max_duration: int = 15,
    ) -> UserProfile:
        """Retrieve existing user profile or instantiate a new personalized profile."""
        clean_id = user_id.strip() if user_id else "anonymous_user"
        if clean_id not in self._profiles:
            modalities = preferred_modalities or {"breathing", "mindfulness", "micro_break"}
            self._profiles[clean_id] = UserProfile(
                user_id=clean_id,
                preferred_modalities=modalities,
                max_preferred_duration=max_duration,
            )
        return self._profiles[clean_id]


_GLOBAL_REGISTRY = UserProfileRegistry()


def get_user_profile(user_id: str | None = None) -> UserProfile:
    return _GLOBAL_REGISTRY.get_or_create(user_id or "default_employee")
