"""
Tests for Mood Mentor Security and Data Privacy (Task 8)
"""

import pytest
from moodmentor.security.privacy import SecurityManager

@pytest.fixture
def security():
    return SecurityManager(secret_key="test-key-1234")

def test_sanitize_xss_scripts(security):
    malicious = "Feeling stressed <script>alert('hack')</script> please help"
    cleaned, warnings = security.sanitize_input(malicious)
    assert "<script>" not in cleaned
    assert "alert" not in cleaned
    assert len(warnings) > 0

def test_sanitize_dom_events(security):
    malicious = "<b onmouseover='stealData()'>Hover me</b>"
    cleaned, warnings = security.sanitize_input(malicious)
    assert "onmouseover=" not in cleaned

def test_anonymize_email_and_phone(security):
    text = "Contact john.doe@enterprise.com or call +1-555-234-5678 regarding EMP_98432."
    scrubbed, counts = security.anonymize_pii(text)
    assert "[REDACTED_EMAIL]" in scrubbed
    assert "[REDACTED_PHONE]" in scrubbed
    assert "[REDACTED_USER_ID]" in scrubbed
    assert "john.doe" not in scrubbed
    assert counts["emails"] == 1
    assert counts["phones"] == 1
    assert counts["emp_ids"] == 1

def test_token_generation_and_validation(security):
    user_id = "emp_ritesh_01"
    token = security.generate_token(user_id)
    assert security.validate_token(token) is True
    assert security.validate_token("tampered:token_signature") is False
    assert security.validate_token("") is False
