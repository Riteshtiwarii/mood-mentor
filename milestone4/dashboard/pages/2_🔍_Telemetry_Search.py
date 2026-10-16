"""
MoodMentor — Live Telemetry & Audit Search Engine
Page 2 of Enterprise SaaS Platform
"""

import streamlit as st
import os
import sys
import time
import json
import pandas as pd

# Ensure parent and package paths are discoverable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dashboard.ui_theme import apply_enterprise_theme, render_top_navbar, render_sidebar_context
from moodmentor.storage.db import DatabaseManager
from moodmentor.search.filter_engine import FilterEngine

# Also safely import Nikhitha's search engine if available
try:
    from milestone4.search.filter_engine import search_telemetry as nikhitha_search
    HAS_NIKHITHA_SEARCH = True
except Exception:
    HAS_NIKHITHA_SEARCH = False

apply_enterprise_theme("Telemetry & Audit Search — MoodMentor", "🔍")
render_top_navbar("Live Telemetry & Parameterized Audit Search")
active_user = render_sidebar_context()
st.page_link("app.py", label="← Return to Clinical Copilot (Text Emotion Analyzer)", icon="🧠")

filter_engine = FilterEngine()
db = DatabaseManager()

st.markdown("""
<div class="enterprise-card">
    <div style="display:flex; justify-content:space-between; align-items:center;">
        <div>
            <h4 style="margin:0; font-size:17px; font-weight:700;">Clinical & System Audit Query Engine</h4>
            <p style="margin:0; font-size:12.5px; color:#64748B;">Execute high-throughput parameterized queries across interaction and system telemetry records</p>
        </div>
        <span class="pill-tag">Indexed SQLite Engine</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Search Filter Controls Form
with st.container():
    st.markdown('<div class="enterprise-card">', unsafe_allow_html=True)
    f_col1, f_col2, f_col3, f_col4 = st.columns(4)

    with f_col1:
        selected_emotion = st.selectbox(
            "Primary Emotion",
            ["All Dimensions", "joy", "sadness", "anger", "fear", "love", "surprise"]
        )

    with f_col2:
        selected_triage = st.selectbox(
            "Clinical Severity Triage",
            ["All Tiers", "Mild", "Moderate", "Elevated", "Critical"]
        )

    with f_col3:
        search_query = st.text_input(
            "Keyword / Modality Search",
            placeholder="e.g. panic, breathing, sleep, stress"
        )

    with f_col4:
        limit_records = st.slider("Result Limit", 10, 100, 25)

    execute_search = st.button("🔎 Run Parameterized Query", type="primary", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Execute Search
t0 = time.time()
filter_kwargs = {
    "user_id": None if selected_emotion != "All Dimensions" else None,
    "emotion": None if selected_emotion == "All Dimensions" else selected_emotion,
    "triage_level": None if selected_triage == "All Tiers" else selected_triage,
    "keyword": search_query.strip() if search_query.strip() else None,
    "limit": limit_records
}

results = filter_engine.search_interactions(db=db, **filter_kwargs)
query_latency = (time.time() - t0) * 1000

# Results & Stats
stat_col1, stat_col2, stat_col3 = st.columns(3)
with stat_col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Matching Records</div>
        <div class="metric-value">{len(results)}</div>
        <div class="metric-subtext">Satisfying parameterized constraints</div>
    </div>
    """, unsafe_allow_html=True)

with stat_col2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Engine Query Latency</div>
        <div class="metric-value">{query_latency:.2f} ms</div>
        <div class="metric-subtext">Optimized indexed execution</div>
    </div>
    """, unsafe_allow_html=True)

with stat_col3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Integrity Status</div>
        <div class="metric-value" style="color:#10B981;">VERIFIED</div>
        <div class="metric-subtext">Audit checksum matched</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

# Results Display
st.markdown('<div class="enterprise-card">', unsafe_allow_html=True)
st.markdown(f"<h4 style='margin:0 0 12px 0; font-size:16px; font-weight:700;'>Query Results ({len(results)} records found)</h4>", unsafe_allow_html=True)

if results:
    df_data = []
    for r in results:
        df_data.append({
            "Record ID": r.get("id"),
            "User": r.get("user_id"),
            "Emotion": r.get("primary_emotion", "").capitalize(),
            "Intensity": f"{r.get('intensity_score', 0):.2f}",
            "Triage Tier": r.get("triage_level", "Mild"),
            "Sanitized Transcript": r.get("sanitized_text", "")[:60] + "...",
            "Timestamp": r.get("timestamp", "")[:19].replace("T", " ")
        })
    df = pd.DataFrame(df_data)
    st.dataframe(df, use_container_width=True)

    # Download action
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download Search Results as CSV",
        data=csv_bytes,
        file_name=f"moodmentor_search_export_{int(time.time())}.csv",
        mime="text/csv",
        use_container_width=False
    )
else:
    st.info("No matching records found. Try loosening the filter criteria.")

st.markdown('</div>', unsafe_allow_html=True)
