"""
Mood Mentor — Security & Data Privacy Manager
Provides input sanitization, PII anonymization, authentication, and GDPR data purge validation.
"""

import re
import hashlib
import hmac
import html
from typing import Dict, Any, List, Optional, Tuple

class SecurityManager:
    # PII Regex Patterns
    EMAIL_PATTERN = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")
    PHONE_PATTERN = re.compile(r"(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b")
    EMPLOYEE_ID_PATTERN = re.compile(r"\b(EMP|USR|ID)[-_]?\d{4,8}\b", re.IGNORECASE)
    SSN_PATTERN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")

    # XSS and Injection Patterns
    XSS_TAG_PATTERN = re.compile(r"<(script|iframe|object|embed|svg|style)[^>]*>.*?</\1>", re.IGNORECASE | re.DOTALL)
    HTML_ATTR_EVENT_PATTERN = re.compile(r"on\w+\s*=", re.IGNORECASE)
    SQL_INJECTION_PATTERN = re.compile(r"(--|;|\b(DROP|ALTER|TRUNCATE|DELETE\s+FROM)\b)", re.IGNORECASE)

    DEFAULT_SECRET_KEY = "moodmentor-production-security-salt-2026"

    def __init__(self, secret_key: str = DEFAULT_SECRET_KEY):
        self.secret_key = secret_key

    def sanitize_input(self, text: str) -> Tuple[str, List[str]]:
        """
        Strips XSS vectors and potential injection payloads.
        Returns: (sanitized_text, warnings)
        """
        if not text:
            return "", []

        warnings = []
        cleaned = text

        # Check & Strip malicious scripts/tags
        if self.XSS_TAG_PATTERN.search(cleaned):
            warnings.append("XSS script tags detected and neutralized.")
            cleaned = self.XSS_TAG_PATTERN.sub("", cleaned)

        if self.HTML_ATTR_EVENT_PATTERN.search(cleaned):
            warnings.append("DOM event handlers stripped.")
            cleaned = self.HTML_ATTR_EVENT_PATTERN.sub("blocked_event=", cleaned)

        # Check SQL dangerous keywords in free text
        if self.SQL_INJECTION_PATTERN.search(cleaned):
            warnings.append("SQL delimiter or destructive command pattern detected.")

        # HTML entity escape
        cleaned = html.escape(cleaned.strip())
        return cleaned, warnings

    def anonymize_pii(self, text: str) -> Tuple[str, Dict[str, int]]:
        """
        Scrubs personally identifiable information (emails, phone numbers, employee IDs)
        prior to ML inference and database persistence.
        """
        if not text:
            return "", {}

        counts = {"emails": 0, "phones": 0, "emp_ids": 0, "ssns": 0}

        def _sub_email(m):
            counts["emails"] += 1
            return "[REDACTED_EMAIL]"

        def _sub_phone(m):
            counts["phones"] += 1
            return "[REDACTED_PHONE]"

        def _sub_emp(m):
            counts["emp_ids"] += 1
            return "[REDACTED_USER_ID]"

        def _sub_ssn(m):
            counts["ssns"] += 1
            return "[REDACTED_SSN]"

        scrubbed = self.EMAIL_PATTERN.sub(_sub_email, text)
        scrubbed = self.PHONE_PATTERN.sub(_sub_phone, scrubbed)
        scrubbed = self.EMPLOYEE_ID_PATTERN.sub(_sub_emp, scrubbed)
        scrubbed = self.SSN_PATTERN.sub(_sub_ssn, scrubbed)

        return scrubbed, counts

    def generate_token(self, user_id: str) -> str:
        """Generates a secure HMAC authentication token for session validation"""
        signature = hmac.new(
            self.secret_key.encode("utf-8"),
            user_id.encode("utf-8"),
            hashlib.sha256
        ).hexdigest()
        return f"{user_id}:{signature[:16]}"

    def validate_token(self, token: str) -> bool:
        """Validates incoming session bearer tokens"""
        if not token or ":" not in token:
            return False
        parts = token.split(":", 1)
        if len(parts) != 2:
            return False
        user_id, expected_sig = parts
        real_sig = hmac.new(
            self.secret_key.encode("utf-8"),
            user_id.encode("utf-8"),
            hashlib.sha256
        ).hexdigest()[:16]
        return hmac.compare_digest(expected_sig, real_sig)

    def hash_identifier(self, identifier: str) -> str:
        """Hashes an employee ID using salted SHA-256 for anonymized metrics"""
        return hashlib.sha256((self.secret_key + identifier).encode("utf-8")).hexdigest()[:16]
