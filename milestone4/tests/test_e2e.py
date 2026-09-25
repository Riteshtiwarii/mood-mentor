"""
Complete End-to-End Workflow Testing for Mood Mentor (Task 6)
Validates:
Text Input -> Sanitization -> Emotion Detection -> Intensity -> Recommendation -> Persistence -> Filter -> Export
"""

import os
import tempfile
try:
    import pytest
except ImportError:
    pytest = None

from moodmentor.core.wellness_catalog import WellnessCatalog
from moodmentor.core.intensity_analyzer import IntensityAnalyzer
from moodmentor.core.recommendation_engine import RecommendationEngine
from moodmentor.core.user_profile import UserProfile
from moodmentor.storage.db import DatabaseManager
from moodmentor.security.privacy import SecurityManager
from moodmentor.search.filter_engine import FilterEngine
from moodmentor.reporting.exporter import ReportExporter

def test_complete_end_to_end_workflow():
    # 1. Initialize Pipeline
    db = DatabaseManager(db_path=":memory:")
    catalog = WellnessCatalog()
    intensity_analyzer = IntensityAnalyzer()
    recommender = RecommendationEngine(catalog=catalog)
    security = SecurityManager()
    filter_engine = FilterEngine()
    exporter = ReportExporter(db)

    user_id = "e2e_employee_42"
    raw_input = "Contact my manager at alice@corp.com. I feel like the walls are closing in on me, my heart is pounding, and I can barely breathe!"

    # 2. Security Sanitization & PII Anonymization
    sanitized, warnings = security.sanitize_input(raw_input)
    clean_text, pii_counts = security.anonymize_pii(sanitized)
    assert "[REDACTED_EMAIL]" in clean_text
    assert "alice@corp.com" not in clean_text

    # 3. Emotion Detection (Simulated transformer inference for panic/fear)
    emotion_probs = {"fear": 0.90, "sadness": 0.04, "anger": 0.03, "joy": 0.01, "love": 0.01, "surprise": 0.01}
    vader_compound = -0.85

    # 4. Intensity & Clinical Triage Analysis
    state = intensity_analyzer.analyze(clean_text, emotion_probs, vader_compound)
    assert state.primary_emotion == "fear"
    assert state.tier in ["High", "Severe"]
    assert state.triage_level in ["High", "Critical"]

    # 5. Hybrid Recommendation Generation
    profile = UserProfile(user_id=user_id, preferred_modalities=["Somatic", "Breathing"])
    recs = recommender.get_recommendations(state, emotion_probs, user_profile=profile, query_text=clean_text, top_k=3)
    assert len(recs) >= 1
    # Somatic grounding or Box breathing should rank top
    top_ids = [r.activity.id for r in recs]
    assert any(act_id in top_ids for act_id in ["act_54321_grounding", "act_box_breathing"])

    # 6. Database Persistence
    int_id = db.save_interaction(
        user_id=user_id,
        raw_text=clean_text,
        primary_emotion=state.primary_emotion,
        confidence=state.confidence,
        intensity_score=state.intensity_score,
        tier=state.tier,
        triage_level=state.triage_level,
        vader_compound=vader_compound
    )
    assert int_id > 0
    db.save_recommendations(int_id, user_id, [r.to_dict() for r in recs])
    db.save_feedback(user_id, recs[0].activity.id, recs[0].activity.title, "accepted", 5.0, "Immediately grounded me.")

    # 7. Search & Filtering Verification
    search_res = filter_engine.search_interactions(db, user_id=user_id, emotion="fear")
    assert len(search_res) == 1
    assert search_res[0]["id"] == int_id

    # 8. Report Export Verification
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        tmp_csv = f.name
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        tmp_pdf = f.name

    try:
        csv_out = exporter.export_interactions_csv(user_id, tmp_csv)
        pdf_out = exporter.export_executive_pdf(user_id, tmp_pdf)

        assert os.path.exists(csv_out) and os.path.getsize(csv_out) > 0
        assert os.path.exists(pdf_out) and os.path.getsize(pdf_out) > 0
    finally:
        if os.path.exists(tmp_csv):
            os.remove(tmp_csv)
        if os.path.exists(tmp_pdf):
            os.remove(tmp_pdf)
