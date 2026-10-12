"""
Tests for Advanced Search and Filtering Engine (Task 4)
"""

import pytest
from moodmentor.storage.db import DatabaseManager
from moodmentor.search.filter_engine import FilterEngine

@pytest.fixture
def populated_db():
    db = DatabaseManager(db_path=":memory:")
    # Seed 3 distinct interactions
    db.save_interaction("user_1", "Acute anxiety attack at work", "fear", 0.9, 0.85, "Severe", "High", timestamp="2026-09-20T10:00:00")
    db.save_interaction("user_1", "Celebrated successful deployment", "joy", 0.95, 0.70, "Moderate", "Mild", timestamp="2026-09-21T11:00:00")
    db.save_interaction("user_2", "Frustrated with team communication", "anger", 0.80, 0.60, "Moderate", "Mild", timestamp="2026-09-22T12:00:00")
    return db

def test_filter_by_emotion(populated_db):
    engine = FilterEngine()
    results = engine.search_interactions(populated_db, emotion="fear")
    assert len(results) == 1
    assert results[0]["primary_emotion"] == "fear"

def test_filter_by_intensity_tier(populated_db):
    engine = FilterEngine()
    results = engine.search_interactions(populated_db, tier="severe")
    assert len(results) == 1
    assert results[0]["tier"] == "Severe"

def test_filter_by_keyword(populated_db):
    engine = FilterEngine()
    results = engine.search_interactions(populated_db, keyword="deployment")
    assert len(results) == 1
    assert "deployment" in results[0]["raw_text"]

def test_filter_by_date_range(populated_db):
    engine = FilterEngine()
    results = engine.search_interactions(populated_db, from_date="2026-09-21T00:00:00", to_date="2026-09-23T00:00:00")
    assert len(results) == 2
