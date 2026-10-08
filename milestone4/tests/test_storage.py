"""
Tests for Mood Mentor SQLite Storage Layer (Task 3 & 8)
"""

import pytest
from moodmentor.storage.db import DatabaseManager

@pytest.fixture
def memory_db():
    return DatabaseManager(db_path=":memory:")

def test_save_and_retrieve_interaction(memory_db):
    int_id = memory_db.save_interaction(
        user_id="user_test_1",
        raw_text="I am feeling great today!",
        primary_emotion="joy",
        confidence=0.92,
        intensity_score=0.75,
        tier="Moderate",
        triage_level="Mild",
        vader_compound=0.85
    )
    assert int_id == 1
    records = memory_db.get_user_interactions("user_test_1")
    assert len(records) == 1
    assert records[0]["primary_emotion"] == "joy"
    assert records[0]["raw_text"] == "I am feeling great today!"

def test_save_and_retrieve_recommendations(memory_db):
    int_id = memory_db.save_interaction("user_1", "text", "fear", 0.9, 0.8, "High", "High")
    recs = [
        {"id": "act_box_breathing", "title": "Box Breathing", "modality": "Breathing", "score": 0.88, "rationale": "High anxiety"}
    ]
    memory_db.save_recommendations(int_id, "user_1", recs)
    saved = memory_db.get_user_recommendations("user_1")
    assert len(saved) == 1
    assert saved[0]["title"] == "Box Breathing"
    assert saved[0]["score"] == 0.88

def test_save_and_retrieve_feedback(memory_db):
    memory_db.save_feedback("user_1", "act_box_breathing", "Box Breathing", "accepted", 5.0, "Very calming")
    feedbacks = memory_db.get_user_feedback("user_1")
    assert len(feedbacks) == 1
    assert feedbacks[0]["action"] == "accepted"
    assert feedbacks[0]["rating"] == 5.0

def test_purge_user_data_gdpr(memory_db):
    int_id = memory_db.save_interaction("user_purge", "text", "anger", 0.8, 0.7, "Moderate", "Mild")
    memory_db.save_recommendations(int_id, "user_purge", [{"id": "act_1", "title": "T", "modality": "M", "score": 0.5}])
    memory_db.save_feedback("user_purge", "act_1", "T", "viewed", 3.0)

    # Purge
    res = memory_db.purge_user_data("user_purge")
    assert res["interactions_deleted"] == 1
    assert res["recommendations_deleted"] == 1
    assert res["feedback_deleted"] == 1

    # Verify empty
    assert len(memory_db.get_user_interactions("user_purge")) == 0
    assert len(memory_db.get_user_recommendations("user_purge")) == 0
