"""
Tests for Report Exporter (CSV & PDF) (Task 5)
"""

import os
import tempfile
import pytest
from moodmentor.storage.db import DatabaseManager
from moodmentor.reporting.exporter import ReportExporter

@pytest.fixture
def populated_db():
    db = DatabaseManager(db_path=":memory:")
    int_id = db.save_interaction("user_export", "Testing export functionality", "sadness", 0.85, 0.65, "Moderate", "Mild")
    db.save_recommendations(int_id, "user_export", [
        {"id": "act_gratitude", "title": "Gratitude Journaling", "modality": "Journaling", "score": 0.82, "rationale": "Sadness support"}
    ])
    return db

def test_export_csv_file_generation(populated_db):
    exporter = ReportExporter(populated_db)
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        tmp_csv = f.name
    try:
        path = exporter.export_interactions_csv("user_export", tmp_csv)
        assert os.path.exists(path)
        assert os.path.getsize(path) > 50
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
            assert "Testing export functionality" in content
            assert "sadness" in content
    finally:
        if os.path.exists(tmp_csv):
            os.remove(tmp_csv)

def test_export_pdf_file_generation(populated_db):
    exporter = ReportExporter(populated_db)
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        tmp_pdf = f.name
    try:
        path = exporter.export_executive_pdf("user_export", tmp_pdf)
        assert os.path.exists(path)
        assert os.path.getsize(path) > 100
        # Check PDF header
        with open(path, "rb") as f:
            header = f.read(5)
            assert header == b"%PDF-"
    finally:
        if os.path.exists(tmp_pdf):
            os.remove(tmp_pdf)
