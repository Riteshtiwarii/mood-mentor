"""
MoodMentor — Longitudinal Analytics & Emotional Trends
Page 1 of Enterprise SaaS Platform
"""

import streamlit as st
import os
import sys
from datetime import datetime, timedelta

# Ensure parent and package paths are discoverable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dashboard.ui_theme import apply_enterprise_theme, render_top_navbar, render_sidebar_context
from dashboard.trend_visualizer import (
    render_intensity_trajectory,
    render_emotion_distribution,
    calculate_burnout_risk_index,
    PALETTE
)
from moodmentor.storage.db import DatabaseManager

apply_enterprise_theme("Longitudinal Analytics & Trends — MoodMentor", "📈")
render_top_navbar("Longitudinal Affect Analytics & Burnout Risk")
active_user = render_sidebar_context()

# Database & telemetry
db = DatabaseManager()
history = db.get_user_interactions(active_user, limit=30)

# If database has no history, provide rich initial telemetry for demo
if not history:
    now = datetime.utcnow()
    sample_emotions = [
        ("fear", 0.78, "Elevated"),
        ("sadness", 0.62, "Moderate"),
        ("anger", 0.45, "Moderate"),
        ("joy", 0.82, "Mild"),
        ("surprise", 0.55, "Moderate"),
        ("fear", 0.88, "Critical"),
        ("joy", 0.74, "Mild")
    ]
    for idx, (emo, score, triage) in enumerate(sample_emotions):
        ts = (now - timedelta(days=6 - idx, hours=idx * 2)).isoformat()
        db.save_interaction(
            user_id=active_user,
            raw_text=f"Sample wellness telemetry entry {idx + 1}",
            primary_emotion=emo,
            confidence=0.92,
            intensity_score=score,
            tier="Moderate",
            triage_level=triage,
            vader_compound=0.1,
            timestamp=ts
        )
    history = db.get_user_interactions(active_user, limit=30)

# Top KPI Metric Strip
burnout_score, burnout_label, burnout_color = calculate_burnout_risk_index(history)
avg_intensity = sum(h["intensity_score"] for h in history) / len(history) if history else 0.0
total_sessions = len(history)
elevated_count = sum(1 for h in history if h.get("triage_level") in ["High", "Critical", "Elevated"])

col_kpi1, col_kpi2, col_kpi3, col_kpi4 = st.columns(4)

with col_kpi1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Burnout Risk Index</div>
        <div class="metric-value" style="color: {burnout_color};">{burnout_score}%</div>
        <div class="metric-subtext">Clinical Status: <b>{burnout_label}</b></div>
    </div>
    """, unsafe_allow_html=True)

with col_kpi2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Mean Intensity (30d)</div>
        <div class="metric-value">{avg_intensity:.2f}</div>
        <div class="metric-subtext">Calibrated on 0.00–1.00 spectrum</div>
    </div>
    """, unsafe_allow_html=True)

with col_kpi3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Logged Interactions</div>
        <div class="metric-value">{total_sessions}</div>
        <div class="metric-subtext">Verified telemetry records</div>
    </div>
    """, unsafe_allow_html=True)

with col_kpi4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Critical / Elevated Triage</div>
        <div class="metric-value" style="color: {'#EF4444' if elevated_count > 0 else '#10B981'};">{elevated_count}</div>
        <div class="metric-subtext">Actionable intervention alerts</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

# Main Visualization Section
chart_col_left, chart_col_right = st.columns([1.7, 1.1])

with chart_col_left:
    st.markdown("""
    <div class="enterprise-card">
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px;">
            <div>
                <h4 style="margin:0; font-size:16px; font-weight:700;">Longitudinal Affective Trajectory</h4>
                <p style="margin:0; font-size:12px; color:#64748B;">Interactive spline curve tracking clinical intensity shifts across time</p>
            </div>
            <span class="pill-tag">Plotly Engine</span>
        </div>
    """, unsafe_allow_html=True)
    trajectory_fig = render_intensity_trajectory(history)
    if trajectory_fig:
        st.plotly_chart(trajectory_fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with chart_col_right:
    st.markdown("""
    <div class="enterprise-card">
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px;">
            <div>
                <h4 style="margin:0; font-size:16px; font-weight:700;">Dominant Affect Breakdown</h4>
                <p style="margin:0; font-size:12px; color:#64748B;">Distribution across 6 primary Ekman dimensions</p>
            </div>
            <span class="pill-tag">Multi-Label</span>
        </div>
    """, unsafe_allow_html=True)
    donut_fig = render_emotion_distribution(history)
    if donut_fig:
        st.plotly_chart(donut_fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# Historical Telemetry Table
st.markdown("""
<div class="enterprise-card">
    <h4 style="margin:0 0 4px 0; font-size:16px; font-weight:700;">Audit Timeline Log</h4>
    <p style="margin:0 0 16px 0; font-size:12px; color:#64748B;">Cryptographically timestamped interaction history for clinical audit</p>
""", unsafe_allow_html=True)

table_rows = []
for h in history[:10]:
    triage = h.get("triage_level", "Mild")
    badge_color = "#10B981" if triage == "Mild" else ("#0284C7" if triage == "Moderate" else "#EF4444")
    table_rows.append({
        "ID": h.get("id"),
        "Timestamp": h.get("timestamp")[:19].replace("T", " "),
        "Primary Emotion": h.get("primary_emotion", "").capitalize(),
        "Intensity Score": f"{h.get('intensity_score', 0):.2f}",
        "Clinical Triage": triage,
        "Input Snippet": h.get("sanitized_text", "")[:45] + "..."
    })

if table_rows:
    st.dataframe(table_rows, use_container_width=True)
else:
    st.info("No audit entries found for this employee profile.")

st.markdown("</div>", unsafe_allow_html=True)
