"""Semantic Wellness Content Matching Module (Milestone 3 - Task 5).

Computes dense text embeddings and cosine semantic similarity between the user's
emotional text/state and curated wellness catalog intervention descriptions.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from exceptions import SemanticMatchingError
from wellness_catalog import WellnessCatalogManager, WellnessIntervention, get_wellness_catalog

logger = logging.getLogger(__name__)


class SemanticWellnessMatcher:
    """Computes semantic similarity between emotional text and wellness content."""

    def __init__(self, catalog_manager: WellnessCatalogManager | None = None) -> None:
        self.catalog = catalog_manager or get_wellness_catalog()
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            sublinear_tf=True,
            stop_words="english",
        )
        self._index_catalog()

    def _index_catalog(self) -> None:
        """Build semantic index from catalog intervention descriptions and metadata."""
        items = self.catalog.get_all()
        if not items:
            raise SemanticMatchingError("Cannot index an empty wellness catalog.")

        self.item_ids = [item.id for item in items]
        self.corpus_docs = [
            f"{item.title}. {item.description} Target emotions: {' '.join(item.target_emotions)}. "
            f"Modality: {item.modality}. Instructions: {item.instructions}"
            for item in items
        ]

        try:
            self.catalog_matrix = self.vectorizer.fit_transform(self.corpus_docs)
            logger.info("Successfully indexed %d wellness interventions into semantic matrix.", len(self.item_ids))
        except Exception as exc:
            raise SemanticMatchingError(f"Failed to index wellness catalog: {exc}") from exc

    def compute_similarity(self, query_text: str) -> Dict[str, float]:
        """Compute cosine semantic similarity between query text and all catalog items.

        Args:
            query_text: Raw or emotional description text.

        Returns:
            Dictionary mapping item_id -> semantic similarity score (0.0 to 1.0).
        """
        if not query_text or not query_text.strip():
            # Return baseline uniform minimal scores for empty query
            return {item_id: 0.1 for item_id in self.item_ids}

        try:
            query_vec = self.vectorizer.transform([query_text.strip()])
            sim_scores = cosine_similarity(query_vec, self.catalog_matrix).flatten()

            max_sim = float(np.max(sim_scores)) if len(sim_scores) > 0 else 0.0
            results: Dict[str, float] = {}
            for idx, item_id in enumerate(self.item_ids):
                raw_score = float(sim_scores[idx])
                if max_sim > 0.05:
                    normalized_score = max(0.0, min(1.0, raw_score / max_sim))
                else:
                    normalized_score = max(0.0, min(1.0, raw_score))
                results[item_id] = normalized_score

            return results
        except Exception as exc:
            raise SemanticMatchingError(
                f"Failed computing semantic similarity for query '{query_text[:30]}...': {exc}"
            ) from exc

    def find_top_semantic_matches(
        self, query_text: str, top_k: int = 5
    ) -> List[Tuple[WellnessIntervention, float]]:
        """Return top-K wellness interventions ranked strictly by semantic similarity."""
        scores = self.compute_similarity(query_text)
        sorted_pairs = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        matched: List[Tuple[WellnessIntervention, float]] = []
        for item_id, score in sorted_pairs:
            item = self.catalog.get_by_id(item_id)
            if item:
                matched.append((item, score))

        return matched
