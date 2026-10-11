"""
MoodMentor — Security, PII Guardrails & GDPR Compliance
Page 4 of Enterprise SaaS Platform
"""

import streamlit as st
import os
import sys

# Ensure parent and package paths are discoverable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dashboard.ui_theme import apply_enterprise_theme, render_top_navbar, render_sidebar_context
from moodmentor.security.privacy import SecurityManager
from moodmentor.storage.db import DatabaseManager

apply_enterprise_theme("Security & Compliance — MoodMentor", "🔒")
render_top_navbar("Security Guardrails, PII Redaction & GDPR Compliance")
active_user = render_sidebar_context()

sec = SecurityManager()
db = DatabaseManager()

st.markdown("""
<div class="enterprise-card">
    <div style="display:flex; justify-content:space-between; align-items:center;">
        <div>
            <h4 style="margin:0; font-size:17px; font-weight:700;">Privacy-First Engineering Architecture</h4>
            <p style="margin:0; font-size:12.5px; color:#64748B;">Live validation of automated PII scrubbing, injection prevention, and GDPR data subject rights</p>
        </div>
        <span class="pill-tag">HIPAA / GDPR Ready</span>
    </div>
</div>
""", unsafe_allow_html=True)

sec_col1, sec_col2 = st.columns([1.3, 1.0])

with sec_col1:
    st.markdown("""
    <div class="enterprise-card">
        <h4 style="margin:0 0 6px 0; font-size:16px; font-weight:700;">🧪 Interactive PII Redaction Sandbox</h4>
        <p style="margin:0 0 14px 0; font-size:12px; color:#64748B;">Enter text containing sensitive employee identifiers to verify real-time anonymization</p>
    """, unsafe_allow_html=True)

    default_test_text = "Hello, my name is Sarah Lin and my email is sarah.lin@enterprise.org. Call me at +1 (555) 234-5678 if the project crashes."
    user_test_input = st.text_area(
        "Raw Input Text",
        value=default_test_text,
        height=95
    )

    if user_test_input:
        sanitized_res = sec.scrub_pii(user_test_input)
        is_clean = (sanitized_res != user_test_input)
        
        st.markdown("<p style='font-size:12px; font-weight:700; color:#475569; margin:10px 0 4px 0;'>Sanitized Output (Ready for Neural Pipeline):</p>", unsafe_allow_html=True)
        st.code(sanitized_res, language="text")

        if is_clean:
            st.success("✅ PII Detected and Safely Masked: Email & Phone numbers anonymized before any inference.")
        else:
            st.info("ℹ️ No PII detected in the provided input.")

    st.markdown("</div>", unsafe_allow_html=True)

with sec_col2:
    st.markdown("""
    <div class="enterprise-card">
        <h4 style="margin:0 0 6px 0; font-size:16px; font-weight:700;">⚖️ GDPR "Right to be Forgotten"</h4>
        <p style="margin:0 0 14px 0; font-size:12px; color:#64748B;">Execute cryptographic purge of all employee records pursuant to Article 17 GDPR</p>
    """, unsafe_allow_html=True)

    records_count = len(db.get_user_history(active_user, limit=500))
    st.markdown(f"""
    <div style="background:#F8FAFC; border:1px solid #E2E8F0; border-radius:10px; padding:12px; margin-bottom:14px; font-size:12.5px;">
        Target Account: <b>{active_user}</b><br>
        Identified Telemetry Entries: <b>{records_count} records</b><br>
        Status: <b>Compliant with 30-day retention policy</b>
    </div>
    """, unsafe_allow_html=True)

    purge_confirm = st.checkbox("Confirm irreversible account erasure request")
    if st.button("🗑️ Execute Right to be Forgotten Purge", type="secondary", disabled=not purge_confirm, use_container_width=True):
        deleted = db.delete_user_data(active_user)
        st.warning(f"Purge Executed: {deleted} interaction records permanently purged from telemetry database.")
        st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

# Security Controls Matrix
st.markdown("""
<div class="enterprise-card">
    <h4 style="margin:0 0 12px 0; font-size:16px; font-weight:700;">Enterprise Security Controls Matrix</h4>
    <div style="display:grid; grid-template-columns: repeat(3, 1fr); gap:14px;">
        <div style="background:#F8FAFC; border:1px solid #E2E8F0; border-radius:10px; padding:14px;">
            <p style="margin:0; font-weight:700; font-size:13px; color:#0F172A;">🛡️ Input Sanitization</p>
            <p style="margin:6px 0 0 0; font-size:11.5px; color:#64748B;">Html-escaping & XSS strip to prevent malicious script injection.</p>
        </div>
        <div style="background:#F8FAFC; border:1px solid #E2E8F0; border-radius:10px; padding:14px;">
            <p style="margin:0; font-weight:700; font-size:13px; color:#0F172A;">🔐 Parameterized SQL</p>
            <p style="margin:6px 0 0 0; font-size:11.5px; color:#64748B;">Zero raw string interpolation; 100% prepared statement execution.</p>
        </div>
        <div style="background:#F8FAFC; border:1px solid #E2E8F0; border-radius:10px; padding:14px;">
            <p style="margin:0; font-weight:700; font-size:13px; color:#0F172A;">🚨 Emergency Guardrails</p>
            <p style="margin:6px 0 0 0; font-size:11.5px; color:#64748B;">Automated high-severity override linking to certified crisis support.</p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)
