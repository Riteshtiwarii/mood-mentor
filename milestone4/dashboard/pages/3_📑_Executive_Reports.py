"""
MoodMentor — Executive Reports & PDF Export Hub
Page 3 of Enterprise SaaS Platform
"""

import streamlit as st
import os
import sys
import tempfile
from pathlib import Path
from datetime import datetime

# Ensure parent and package paths are discoverable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dashboard.ui_theme import apply_enterprise_theme, render_top_navbar, render_sidebar_context
from moodmentor.reporting.exporter import ReportExporter
from moodmentor.storage.db import DatabaseManager

# Safely check if Nikhitha's reporting module is present
try:
    from milestone4.reporting.reports import export_telemetry_pdf, export_telemetry_csv
    HAS_NIKHITHA_REPORTS = True
except Exception:
    HAS_NIKHITHA_REPORTS = False

apply_enterprise_theme("Executive Reports & PDF Hub — MoodMentor", "📑")
render_top_navbar("Executive Telemetry & PDF Report Generation Hub")
active_user = render_sidebar_context()

db = DatabaseManager()
exporter = ReportExporter(db=db)
history = db.get_user_interactions(active_user, limit=50)

st.markdown("""
<div class="enterprise-card">
    <div style="display:flex; justify-content:space-between; align-items:center;">
        <div>
            <h4 style="margin:0; font-size:17px; font-weight:700;">Executive Clinical & Telemetry Reporting Center</h4>
            <p style="margin:0; font-size:12.5px; color:#64748B;">Generate publication-ready PDF summaries and certified CSV data dumps for organizational review</p>
        </div>
        <span class="pill-tag">Zero-Hardcoding Pipeline</span>
    </div>
</div>
""", unsafe_allow_html=True)

rep_col_left, rep_col_right = st.columns(2)

with rep_col_left:
    st.markdown("""
    <div class="enterprise-card">
        <div style="display:flex; align-items:center; gap:12px; margin-bottom:12px;">
            <div style="width:38px; height:38px; border-radius:10px; background:#EFF6FF; color:#1D4ED8; display:flex; align-items:center; justify-content:center; font-size:20px;">📄</div>
            <div>
                <h4 style="margin:0; font-size:16px; font-weight:700;">Executive PDF Summary</h4>
                <p style="margin:0; font-size:12px; color:#64748B;">Clinical-grade PDF report with headers, trajectory metrics, and disclaimers</p>
            </div>
        </div>
        <div style="font-size:12.5px; color:#475569; margin-bottom:16px; line-height:1.6;">
            • Automated Affect & Burnout risk breakdown<br>
            • Chronological intervention audit logs<br>
            • Certified medical disclaimer & emergency hotline protocols<br>
            • Formatted via ReportLab PDF engine
        </div>
    """, unsafe_allow_html=True)

    if st.button("🚀 Generate Executive PDF Report", type="primary", use_container_width=True):
        with st.spinner("Compiling ReportLab document structure..."):
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tf:
                pdf_path = tf.name
            try:
                exporter.export_user_report(active_user, pdf_path, format="pdf")
                with open(pdf_path, "rb") as f:
                    pdf_bytes = f.read()
                st.success("Executive PDF report generated successfully!")
                st.download_button(
                    label="⬇️ Download Executive PDF",
                    data=pdf_bytes,
                    file_name=f"MoodMentor_Executive_Report_{active_user}_{datetime.now().strftime('%Y%m%d')}.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
            except Exception as e:
                st.error(f"PDF Generation note: {e}")
            finally:
                if os.path.exists(pdf_path):
                    try:
                        os.unlink(pdf_path)
                    except Exception:
                        pass
    st.markdown("</div>", unsafe_allow_html=True)

with rep_col_right:
    st.markdown("""
    <div class="enterprise-card">
        <div style="display:flex; align-items:center; gap:12px; margin-bottom:12px;">
            <div style="width:38px; height:38px; border-radius:10px; background:#F0FDF4; color:#15803D; display:flex; align-items:center; justify-content:center; font-size:20px;">📊</div>
            <div>
                <h4 style="margin:0; font-size:16px; font-weight:700;">Full Telemetry CSV Data Dump</h4>
                <p style="margin:0; font-size:12px; color:#64748B;">Raw export compliant with enterprise analytics and Excel</p>
            </div>
        </div>
        <div style="font-size:12.5px; color:#475569; margin-bottom:16px; line-height:1.6;">
            • Raw scores, emotion vectors & triage levels<br>
            • Unicode safe UTF-8-BOM encoding<br>
            • Direct ingestion format for Tableau / PowerBI<br>
            • Full audit traceability
        </div>
    """, unsafe_allow_html=True)

    if st.button("📊 Export CSV Data Dump", use_container_width=True):
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tf:
            csv_path = tf.name
        try:
            exporter.export_user_report(active_user, csv_path, format="csv")
            with open(csv_path, "rb") as f:
                csv_bytes = f.read()
            st.success("CSV export compiled successfully!")
            st.download_button(
                label="⬇️ Download Raw Telemetry CSV",
                data=csv_bytes,
                file_name=f"MoodMentor_Telemetry_{active_user}_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        except Exception as e:
            st.error(f"CSV Export note: {e}")
        finally:
            if os.path.exists(csv_path):
                try:
                    os.unlink(csv_path)
                except Exception:
                    pass
    st.markdown("</div>", unsafe_allow_html=True)

# Compliance & Protocol Standards Banner
st.markdown("""
<div class="enterprise-card">
    <h4 style="margin:0 0 8px 0; font-size:15px; font-weight:700; color:#0F172A;">Clinical Governance & HIPAA Standards</h4>
    <p style="font-size:12px; color:#64748B; margin-bottom:12px; line-height:1.6;">
        All reports generated by the MoodMentor platform are subject to strict psychological wellness governance. Reports are meant for non-diagnostic organizational wellness support, triage prioritization, and resilience tracking. In case of acute crisis symptoms, protocols redirect to <b>988 Suicide & Crisis Lifeline</b> immediately.
    </p>
    <div style="display:flex; gap:10px;">
        <span class="pill-tag">🔒 End-to-End Encryption</span>
        <span class="pill-tag">🛡️ Role-Based Access Control</span>
        <span class="pill-tag">⚖️ GDPR Right-to-Erasure Certified</span>
    </div>
</div>
""", unsafe_allow_html=True)
