"""Wellness Content Catalog Manager (Milestone 3 - Task 5).

Manages evidence-based workplace wellness interventions, metadata indexing,
intensity band filtering, and clinical triage discovery.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from config import DATA_DIR
from exceptions import WellnessCatalogError

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class WellnessIntervention:
    """Immutable data model for a curated clinical/wellness intervention."""

    id: str
    title: str
    category: str
    modality: str
    duration_minutes: int
    target_emotions: list[str]
    min_intensity: float
    max_intensity: float
    description: str
    instructions: str
    is_crisis_resource: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "category": self.category,
            "modality": self.modality,
            "duration_minutes": self.duration_minutes,
            "target_emotions": self.target_emotions,
            "min_intensity": self.min_intensity,
            "max_intensity": self.max_intensity,
            "description": self.description,
            "instructions": self.instructions,
            "is_crisis_resource": self.is_crisis_resource,
        }


class WellnessCatalogManager:
    """In-memory indexer and retrieval manager for wellness content."""

    def __init__(self, catalog_json_path: Path = DATA_DIR / "wellness_catalog.json") -> None:
        self.catalog_path = catalog_json_path
        self._items: Dict[str, WellnessIntervention] = {}
        self.load_catalog()

    def load_catalog(self) -> None:
        """Load and parse interventions from JSON."""
        if not self.catalog_path.exists():
            raise WellnessCatalogError(f"Catalog file not found: {self.catalog_path}")

        try:
            with self.catalog_path.open("r", encoding="utf-8") as f:
                data = json.load(f)

            self._items.clear()
            for entry in data:
                item = WellnessIntervention(
                    id=entry["id"],
                    title=entry["title"],
                    category=entry["category"],
                    modality=entry["modality"],
                    duration_minutes=int(entry["duration_minutes"]),
                    target_emotions=[e.lower() for e in entry.get("target_emotions", [])],
                    min_intensity=float(entry.get("min_intensity", 0.0)),
                    max_intensity=float(entry.get("max_intensity", 1.0)),
                    description=entry["description"],
                    instructions=entry["instructions"],
                    is_crisis_resource=bool(entry.get("is_crisis_resource", False)),
                )
                self._items[item.id] = item

            logger.info("Successfully loaded %d wellness interventions into catalog.", len(self._items))
        except Exception as exc:
            raise WellnessCatalogError(f"Failed to load wellness catalog: {exc}") from exc

    def get_all(self) -> list[WellnessIntervention]:
        """Return all available interventions."""
        return list(self._items.values())

    def get_by_id(self, item_id: str) -> Optional[WellnessIntervention]:
        """Retrieve a specific intervention by unique ID."""
        return self._items.get(item_id)

    def get_crisis_interventions(self) -> list[WellnessIntervention]:
        """Retrieve priority interventions flagged for acute crisis triage."""
        return [item for item in self._items.values() if item.is_crisis_resource]

    def filter_by_intensity(self, intensity_score: float) -> list[WellnessIntervention]:
        """Return interventions whose intensity band encompasses the user's score."""
        return [
            item for item in self._items.values()
            if item.min_intensity <= intensity_score <= item.max_intensity
        ]

    def filter_by_emotion(self, emotion: str) -> list[WellnessIntervention]:
        """Return interventions that specifically target a given emotion."""
        clean_emo = emotion.lower().strip()
        return [item for item in self._items.values() if clean_emo in item.target_emotions]


_GLOBAL_CATALOG = WellnessCatalogManager()


def get_wellness_catalog() -> WellnessCatalogManager:
    return _GLOBAL_CATALOG
