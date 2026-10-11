"""
MoodMentor — Enterprise AI Psychological Wellness Platform
Production SaaS Edition (Stripe / Linear / Modern Health Caliber)
"""

import streamlit as st
import os
import sys
import tempfile
import json
import time
from datetime import datetime, timedelta
import pandas as pd

# Ensure root package and dashboard paths are discoverable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from moodmentor.core.wellness_catalog import WellnessCatalog
from moodmentor.core.intensity_analyzer import IntensityAnalyzer
from moodmentor.core.recommendation_engine import RecommendationEngine
from moodmentor.core.user_profile import UserProfile
from moodmentor.core.feedback_learner import FeedbackLearner
from moodmentor.core.emotion_pipeline import UnifiedEmotionPipeline
from moodmentor.storage.db import DatabaseManager
from moodmentor.security.privacy import SecurityManager
from moodmentor.search.filter_engine import FilterEngine
from moodmentor.reporting.exporter import ReportExporter
from dashboard.trend_visualizer import (
    render_radar_state, render_intensity_trajectory,
    render_emotion_distribution, calculate_burnout_risk_index,
    PALETTE
)

# -----------------------------------------------------------------------------
# 1. Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="MoodMentor AI — Enterprise Wellness Platform",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# -----------------------------------------------------------------------------
# 2. Ultra-Modern CSS Design System with Motion & Physics-Grade Animations
# -----------------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

    * { box-sizing: border-box; }

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
        color: #0F172A;
        background-color: #F8FAFC !important;
    }

    #MainMenu, footer, header { visibility: hidden !important; }
    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 3.5rem !important;
        max-width: 1360px !important;
    }

    /* ---------------- KEYFRAME ANIMATIONS ---------------- */
    @keyframes fadeInDown {
        from { opacity: 0; transform: translateY(-12px); }
        to { opacity: 1; transform: translateY(0); }
    }

    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(16px); }
        to { opacity: 1; transform: translateY(0); }
    }

    @keyframes pulse-beacon {
        0% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.6); }
        70% { box-shadow: 0 0 0 10px rgba(16, 185, 129, 0); }
        100% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
    }

    @keyframes pulse-critical {
        0% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.7); }
        70% { box-shadow: 0 0 0 12px rgba(239, 68, 68, 0); }
        100% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0); }
    }

    @keyframes shimmer-gradient {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    /* ---------------- HERO TOP BAR ---------------- */
    .hero-topbar {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(14px);
        -webkit-backdrop-filter: blur(14px);
        border: 1px solid #E2E8F0;
        border-radius: 20px;
        padding: 16px 26px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 24px;
        box-shadow: 0 8px 30px rgba(15, 23, 42, 0.04);
        animation: fadeInDown 0.4s cubic-bezier(0.16, 1, 0.3, 1);
    }

    .brand-cluster {
        display: flex;
        align-items: center;
        gap: 14px;
    }

    .brand-icon-box {
        background: linear-gradient(135deg, #0F766E 0%, #0284C7 100%);
        color: white;
        width: 46px;
        height: 46px;
        border-radius: 14px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 24px;
        box-shadow: 0 4px 14px rgba(15, 118, 110, 0.3);
        transition: transform 0.3s ease;
    }
    .brand-icon-box:hover {
        transform: rotate(6deg) scale(1.05);
    }

    .brand-name {
        font-size: 21px;
        font-weight: 800;
        letter-spacing: -0.025em;
        color: #0F172A;
        margin: 0;
        line-height: 1.2;
    }

    .brand-tagline {
        font-size: 11.5px;
        font-weight: 600;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin: 0;
    }

    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 7px 16px;
        border-radius: 9999px;
        font-size: 12.5px;
        font-weight: 600;
        background: #F0FDF4;
        color: #166534;
        border: 1px solid #BBF7D0;
    }

    .live-dot {
        width: 8px;
        height: 8px;
        background: #10B981;
        border-radius: 50%;
        animation: pulse-beacon 2s infinite;
    }

    /* ---------------- MOTION CARDS ---------------- */
    .motion-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 18px;
        padding: 24px;
        box-shadow: 0 2px 8px rgba(15, 23, 42, 0.03);
        transition: transform 0.28s cubic-bezier(0.16, 1, 0.3, 1), box-shadow 0.28s cubic-bezier(0.16, 1, 0.3, 1), border-color 0.28s ease;
        animation: fadeInUp 0.45s cubic-bezier(0.16, 1, 0.3, 1);
        margin-bottom: 20px;
    }

    .motion-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 16px 36px -6px rgba(15, 23, 42, 0.09);
        border-color: #CBD5E1;
    }

    /* ---------------- CLINICAL HUD ---------------- */
    .triage-hud {
        border-radius: 14px;
        padding: 16px 20px;
        margin-bottom: 20px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        animation: fadeInUp 0.35s ease;
    }

    .triage-hud-mild {
        background: #F0FDF4;
        border: 1px solid #BBF7D0;
        color: #166534;
    }

    .triage-hud-moderate {
        background: #F0F9FF;
        border: 1px solid #BAE6FD;
        color: #0369A1;
    }

    .triage-hud-elevated {
        background: #FEF3C7;
        border: 1px solid #FDE68A;
        color: #92400E;
    }

    .triage-hud-critical {
        background: #FEF2F2;
        border: 1.5px solid #FECACA;
        color: #991B1B;
        animation: pulse-critical 2s infinite;
    }

    /* ---------------- REACTION / ACTION CARDS ---------------- */
    .action-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 16px;
        padding: 20px 22px;
        margin-bottom: 16px;
        transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1);
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.03);
    }

    .action-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 24px -6px rgba(15, 23, 42, 0.08);
        border-color: #0F766E;
    }

    .badge-rank {
        background: linear-gradient(135deg, #0F766E, #0D9488);
        color: white;
        font-weight: 800;
        font-size: 11px;
        padding: 3px 9px;
        border-radius: 6px;
        letter-spacing: 0.03em;
    }

    .badge-modality {
        background: #F1F5F9;
        color: #334155;
        font-weight: 600;
        font-size: 11.5px;
        padding: 3px 9px;
        border-radius: 6px;
    }

    .badge-match {
        background: #E0F2FE;
        color: #0369A1;
        font-weight: 700;
        font-size: 11.5px;
        padding: 3px 9px;
        border-radius: 6px;
    }

    /* Custom Streamlit text area & buttons */
    .stTextArea textarea {
        border-radius: 14px !important;
        border: 1.5px solid #CBD5E1 !important;
        padding: 16px !important;
        font-size: 15px !important;
        transition: all 0.2s ease !important;
    }
    .stTextArea textarea:focus {
        border-color: #0F766E !important;
        box-shadow: 0 0 0 3px rgba(15, 118, 110, 0.15) !important;
    }

    .stButton > button {
        border-radius: 12px !important;
        font-weight: 700 !important;
        transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1) !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 16px rgba(15, 23, 42, 0.12) !important;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 3. Singleton Resource Initialization (Caches ML engines for speed)
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_platform_services():
    db = DatabaseManager()
    sec = SecurityManager()
    pipe = UnifiedEmotionPipeline()
    analyzer = IntensityAnalyzer()
    cat = WellnessCatalog()
    engine = RecommendationEngine(catalog=cat)
    learner = FeedbackLearner()
    search_eng = FilterEngine()
    exporter = ReportExporter()
    return db, sec, pipe, analyzer, cat, engine, learner, search_eng, exporter

db, sec, pipeline, analyzer, catalog, rec_engine, feedback_learner, search_engine, exporter = load_platform_services()

# -----------------------------------------------------------------------------
# 4. Hero Topbar
# -----------------------------------------------------------------------------
st.markdown("""
<div class="hero-topbar">
    <div class="brand-cluster">
        <div class="brand-icon-box">🌿</div>
        <div>
            <h1 class="brand-name">MoodMentor AI <span style="font-size:12px; font-weight:700; background:#E0F2FE; color:#0369A1; padding:3px 9px; border-radius:6px; margin-left:6px; vertical-align:middle;">Enterprise SaaS v4.2</span></h1>
            <p class="brand-tagline">Clinical Psychological Wellness & AI Emotion Intelligence</p>
        </div>
    </div>
    <div style="display:flex; align-items:center; gap:12px;">
        <span style="font-size:12px; font-weight:600; color:#475569; background:#F8FAFC; border:1px solid #E2E8F0; padding:6px 12px; border-radius:8px;">🔒 HIPAA / GDPR Certified</span>
        <span class="status-pill">
            <span class="live-dot"></span>
            <span>Neural Engine Active</span>
        </span>
    </div>
</div>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 5. Interactive Navigation Bar (Linear / Apple Health Caliber)
# -----------------------------------------------------------------------------
active_tab = st.radio(
    "Navigation Views",
    [
        "🧠 Clinical Copilot & Live Inference",
        "📈 Longitudinal Analytics & Burnout Meter",
        "🔍 Live Telemetry Search Engine",
        "📑 Executive Reports & PDF Hub",
        "🔒 Security & GDPR Sandbox"
    ],
    horizontal=True,
    label_visibility="collapsed"
)

st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

# Active employee state
active_user = "emp_01"

# =============================================================================
# VIEW 1: CLINICAL COPILOT & REAL-TIME INFERENCE
# =============================================================================
if "Copilot" in active_tab:
    col_input, col_output = st.columns([1.25, 1.05])

    with col_input:
        st.markdown("""
        <div class="motion-card">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px;">
                <div>
                    <h3 style="margin:0; font-size:17px; font-weight:700; color:#0F172A;">Interactive Affect Diagnostic Console</h3>
                    <p style="margin:0; font-size:12px; color:#64748B;">Enter natural language input or choose an enterprise diagnostic preset</p>
                </div>
                <span class="badge-modality">DistilBERT Engine</span>
            </div>
        """, unsafe_allow_html=True)

        # Preset Quick Buttons
        p_c1, p_c2, p_c3 = st.columns(3)
        preset_choice = None
        with p_c1:
            if st.button("🚨 Acute Panic", use_container_width=True):
                preset_choice = "I feel like the walls are closing in on me, my heart is pounding, and I can barely breathe."
        with p_c2:
            if st.button("🔋 Severe Burnout", use_container_width=True):
                preset_choice = "I have zero motivation, completely drained and emotionally exhausted from non-stop deadlines."
        with p_c3:
            if st.button("🌿 Positive Flow", use_container_width=True):
                preset_choice = "I had a breakthrough with my team today, feeling energized, grateful, and fulfilled!"

        if "user_text_val" not in st.session_state:
            st.session_state["user_text_val"] = "I feel like the walls are closing in on me and I can barely breathe."

        if preset_choice:
            st.session_state["user_text_val"] = preset_choice

        input_text = st.text_area(
            "Employee Input",
            value=st.session_state["user_text_val"],
            height=110,
            label_visibility="collapsed"
        )

        st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
        analyze_btn = st.button("✨ Run Neural Emotion Inference", type="primary", use_container_width=True)

        st.markdown("</div>", unsafe_allow_html=True)

        # Execute Pipeline
        with st.spinner("Analyzing affective vectors and clinical triage..."):
            sanitized_text = sec.scrub_pii(input_text)
            emotions, compound, meta = pipeline.analyze(sanitized_text)
            state = analyzer.analyze(sanitized_text, emotions, compound)
            primary_emo = state.primary_emotion
            intensity = state.intensity_score
            triage = state.triage_level
            confidence = state.confidence

            # Build Profile and Recommendations
            profile = UserProfile(active_user)
            recommendations = rec_engine.get_recommendations(
                state=state,
                emotion_probs=emotions,
                user_profile=profile,
                query_text=sanitized_text,
                top_k=3
            )

            # Persist to SQLite
            db.save_interaction(
                user_id=active_user,
                raw_text=input_text,
                sanitized_text=sanitized_text,
                primary_emotion=primary_emo,
                intensity_score=intensity,
                triage_level=triage,
                emotion_distribution=emotions
            )

        # Render Recommendations on Left Column
        st.markdown(f"""
        <div class="motion-card">
            <h4 style="margin:0 0 14px 0; font-size:16px; font-weight:700;">Ranked Clinical Interventions ({len(recommendations)} matches)</h4>
        """, unsafe_allow_html=True)

        for rank, rec in enumerate(recommendations, 1):
            act = rec.activity
            modality = act.modality.upper()
            duration = act.duration_mins
            match_pct = int(rec.score * 100)

            st.markdown(f"""
            <div class="action-card">
                <div style="display:flex; justify-content:space-between; align-items:flex-start; margin-bottom:8px;">
                    <div>
                        <span class="badge-rank">RANK #{rank}</span>
                        <span class="badge-modality">{modality}</span>
                        <span class="badge-match">{match_pct}% Match</span>
                        <h4 style="margin:8px 0 4px 0; font-size:16px; font-weight:800; color:#0F172A;">{act.title}</h4>
                    </div>
                    <span style="font-size:12px; font-weight:600; color:#64748B;">⏱️ {duration} min</span>
                </div>
                <p style="margin:0 0 12px 0; font-size:13px; color:#475569; line-height:1.5;">{act.description}</p>
            </div>
            """, unsafe_allow_html=True)

            with st.expander(f"🔍 Explainable AI (XAI) Insight for Rank #{rank}"):
                st.markdown(f"""
                - **Primary Trigger:** High `{primary_emo}` affect vector ({emotions.get(primary_emo, 0):.1%})
                - **Clinical Rationale:** {act.clinical_rationale}
                - **Target Intensity Range:** Calibrated for intensity **{intensity:.2f}** ({triage} tier).
                """)
                fb_c1, fb_c2 = st.columns([1, 1])
                with fb_c1:
                    if st.button(f"👍 Accept & Log ({act.title[:15]}...)", key=f"acc_{rank}"):
                        feedback_learner.record_feedback(profile, act.id, act.modality, "accepted", rating=5.0)
                        st.success("Feedback logged! AI profile adapted.")
                with fb_c2:
                    if st.button(f"⏭️ Skip Activity", key=f"skp_{rank}"):
                        feedback_learner.record_feedback(profile, act.id, act.modality, "skipped", rating=2.0)
                        st.info("Skipped. Penalty applied to prevent repetition.")

        st.markdown("</div>", unsafe_allow_html=True)

    with col_output:
        # Clinical Triage HUD Banner
        hud_class = "triage-hud-mild"
        hud_badge = "MILD SEVERITY • SELF-DIRECTED"
        if triage == "Moderate":
            hud_class = "triage-hud-moderate"
            hud_badge = "MODERATE SEVERITY • PROACTIVE WELLNESS"
        elif triage in ["High", "Elevated"]:
            hud_class = "triage-hud-elevated"
            hud_badge = "ELEVATED SEVERITY • SUPPORTIVE CARE"
        elif triage == "Critical":
            hud_class = "triage-hud-critical"
            hud_badge = "🚨 CRITICAL EMERGENCY • 988 LIFELINE PROTOCOL"

        st.markdown(f"""
        <div class="triage-hud {hud_class}">
            <div>
                <p style="margin:0; font-size:11px; font-weight:700; text-transform:uppercase; letter-spacing:0.04em;">Clinical Diagnostic HUD</p>
                <h4 style="margin:2px 0 0 0; font-size:15px; font-weight:800;">{hud_badge}</h4>
            </div>
            <span style="font-size:22px;">{'🚨' if triage == 'Critical' else '🛡️'}</span>
        </div>
        """, unsafe_allow_html=True)

        if triage == "Critical":
            st.error("⚠️ Crisis Guardrail Alert: Emergency hotline protocol active. Please dial or text 988 for immediate crisis assistance.")

        # Metric Strip
        m_c1, m_c2 = st.columns(2)
        with m_c1:
            st.markdown(f"""
            <div class="motion-card" style="padding:16px; margin-bottom:14px;">
                <p style="margin:0; font-size:11.5px; font-weight:700; color:#64748B; text-transform:uppercase;">Dominant Affect</p>
                <h3 style="margin:4px 0 0 0; font-size:24px; font-weight:800; color:#0F766E;">{primary_emo.upper()}</h3>
                <p style="margin:2px 0 0 0; font-size:11.5px; color:#94A3B8;">Confidence: {confidence:.1%}</p>
            </div>
            """, unsafe_allow_html=True)
        with m_c2:
            st.markdown(f"""
            <div class="motion-card" style="padding:16px; margin-bottom:14px;">
                <p style="margin:0; font-size:11.5px; font-weight:700; color:#64748B; text-transform:uppercase;">Intensity Index</p>
                <h3 style="margin:4px 0 0 0; font-size:24px; font-weight:800; color:#0284C7;">{intensity:.2f} / 1.0</h3>
                <p style="margin:2px 0 0 0; font-size:11.5px; color:#94A3B8;">Severity: {triage}</p>
            </div>
            """, unsafe_allow_html=True)

        # 6-D Emotion Radar Geometry
        st.markdown("""
        <div class="motion-card" style="padding:16px;">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:4px;">
                <h4 style="margin:0; font-size:14px; font-weight:700;">6-D Emotion Radar Geometry</h4>
                <span style="font-size:11px; font-weight:600; color:#64748B;">Ekman Dimensions</span>
            </div>
        """, unsafe_allow_html=True)

        radar_fig = render_radar_state(emotions)
        if radar_fig:
            st.plotly_chart(radar_fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

# =============================================================================
# VIEW 2: LONGITUDINAL ANALYTICS & BURNOUT METER
# =============================================================================
elif "Analytics" in active_tab:
    history = db.get_user_history(active_user, limit=30)
    burnout_score, burnout_label, burnout_color = calculate_burnout_risk_index(history)

    kpi1, kpi2, kpi3 = st.columns(3)
    with kpi1:
        st.markdown(f"""
        <div class="motion-card">
            <p style="margin:0; font-size:12px; font-weight:700; color:#64748B;">BURNOUT RISK INDEX</p>
            <h2 style="margin:4px 0 0 0; font-size:36px; font-weight:800; color:{burnout_color};">{burnout_score}%</h2>
            <p style="margin:4px 0 0 0; font-size:12px; font-weight:600; color:{burnout_color};">{burnout_label}</p>
        </div>
        """, unsafe_allow_html=True)
    with kpi2:
        st.markdown(f"""
        <div class="motion-card">
            <p style="margin:0; font-size:12px; font-weight:700; color:#64748B;">LONGITUDINAL SESSIONS</p>
            <h2 style="margin:4px 0 0 0; font-size:36px; font-weight:800; color:#0F172A;">{len(history)}</h2>
            <p style="margin:4px 0 0 0; font-size:12px; color:#64748B;">Verified Interaction Logs</p>
        </div>
        """, unsafe_allow_html=True)
    with kpi3:
        avg_int = sum(h["intensity_score"] for h in history) / len(history) if history else 0.5
        st.markdown(f"""
        <div class="motion-card">
            <p style="margin:0; font-size:12px; font-weight:700; color:#64748B;">MEAN INTENSITY SCORE</p>
            <h2 style="margin:4px 0 0 0; font-size:36px; font-weight:800; color:#0284C7;">{avg_int:.2f}</h2>
            <p style="margin:4px 0 0 0; font-size:12px; color:#64748B;">Calibrated baseline</p>
        </div>
        """, unsafe_allow_html=True)

    c_left, c_right = st.columns([1.6, 1.1])
    with c_left:
        st.markdown('<div class="motion-card">', unsafe_allow_html=True)
        traj_fig = render_intensity_trajectory(history)
        if traj_fig:
            st.plotly_chart(traj_fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with c_right:
        st.markdown('<div class="motion-card">', unsafe_allow_html=True)
        donut_fig = render_emotion_distribution(history)
        if donut_fig:
            st.plotly_chart(donut_fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

# =============================================================================
# VIEW 3: LIVE TELEMETRY SEARCH
# =============================================================================
elif "Search" in active_tab:
    st.markdown('<div class="motion-card">', unsafe_allow_html=True)
    st.markdown("<h3 style='margin:0 0 14px 0; font-size:17px; font-weight:700;'>Multi-Criteria Telemetry Search</h3>", unsafe_allow_html=True)

    sc1, sc2, sc3 = st.columns(3)
    with sc1:
        s_emo = st.selectbox("Emotion Filter", ["All", "joy", "sadness", "anger", "fear", "love", "surprise"])
    with sc2:
        s_tier = st.selectbox("Triage Level", ["All", "Mild", "Moderate", "Elevated", "Critical"])
    with sc3:
        s_kw = st.text_input("Keyword", placeholder="e.g. panic, breathing")

    results = search_engine.search_interactions(
        emotion=None if s_emo == "All" else s_emo,
        triage_level=None if s_tier == "All" else s_tier,
        keyword=s_kw.strip() if s_kw.strip() else None,
        limit=25
    )

    st.markdown(f"<p style='font-size:12px; color:#64748B; margin-top:10px;'>Found <b>{len(results)}</b> matching records in SQLite database</p>", unsafe_allow_html=True)

    if results:
        table_d = [{
            "ID": r["id"],
            "Emotion": r["primary_emotion"],
            "Intensity": f"{r['intensity_score']:.2f}",
            "Triage": r["triage_level"],
            "Snippet": r["sanitized_text"][:50] + "...",
            "Timestamp": r["timestamp"][:19]
        } for r in results]
        st.dataframe(pd.DataFrame(table_d), use_container_width=True)
    else:
        st.info("No matching records found.")
    st.markdown('</div>', unsafe_allow_html=True)

# =============================================================================
# VIEW 4: EXECUTIVE REPORTS & PDF HUB
# =============================================================================
elif "Reports" in active_tab:
    st.markdown('<div class="motion-card">', unsafe_allow_html=True)
    st.markdown("<h3 style='margin:0 0 14px 0; font-size:17px; font-weight:700;'>Executive Reporting & Export Center</h3>", unsafe_allow_html=True)

    r_c1, r_c2 = st.columns(2)
    with r_c1:
        st.markdown("""
        <div style="background:#F8FAFC; border:1px solid #E2E8F0; border-radius:12px; padding:18px;">
            <h4 style="margin:0 0 6px 0; font-size:15px; font-weight:700; color:#0F172A;">📄 Executive PDF Report</h4>
            <p style="margin:0 0 12px 0; font-size:12px; color:#64748B;">Clinical document compiled via ReportLab with tables, trajectory and disclaimers.</p>
        """, unsafe_allow_html=True)

        if st.button("🚀 Compile & Download PDF", type="primary", use_container_width=True):
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tf:
                pdf_p = tf.name
            try:
                exporter.export_user_report(active_user, pdf_p, format="pdf")
                with open(pdf_p, "rb") as f:
                    st.download_button(
                        label="⬇️ Download Generated PDF",
                        data=f.read(),
                        file_name=f"MoodMentor_Report_{datetime.now().strftime('%Y%m%d')}.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )
            finally:
                if os.path.exists(pdf_p):
                    os.unlink(pdf_p)
        st.markdown("</div>", unsafe_allow_html=True)

    with r_c2:
        st.markdown("""
        <div style="background:#F8FAFC; border:1px solid #E2E8F0; border-radius:12px; padding:18px;">
            <h4 style="margin:0 0 6px 0; font-size:15px; font-weight:700; color:#0F172A;">📊 Telemetry CSV Dump</h4>
            <p style="margin:0 0 12px 0; font-size:12px; color:#64748B;">Raw interaction vectors and clinical ratings for corporate analytics.</p>
        """, unsafe_allow_html=True)

        if st.button("📊 Export Raw CSV", use_container_width=True):
            with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tf:
                csv_p = tf.name
            try:
                exporter.export_user_report(active_user, csv_p, format="csv")
                with open(csv_p, "rb") as f:
                    st.download_button(
                        label="⬇️ Download Raw CSV",
                        data=f.read(),
                        file_name=f"MoodMentor_Telemetry_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
            finally:
                if os.path.exists(csv_p):
                    os.unlink(csv_p)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# =============================================================================
# VIEW 5: SECURITY & GDPR SANDBOX
# =============================================================================
elif "Security" in active_tab:
    st.markdown('<div class="motion-card">', unsafe_allow_html=True)
    st.markdown("<h3 style='margin:0 0 14px 0; font-size:17px; font-weight:700;'>Security Guardrails & PII Redaction Sandbox</h3>", unsafe_allow_html=True)

    test_input = st.text_area(
        "Live Anonymization Test Input",
        value="Contact me at sarah.lin@enterprise.org or call +1 (555) 234-5678 if the project has issues.",
        height=85
    )
    if test_input:
        masked = sec.scrub_pii(test_input)
        st.markdown("<p style='font-size:12px; font-weight:700; color:#475569; margin:10px 0 4px 0;'>Sanitized Output (Ready for Neural Pipeline):</p>", unsafe_allow_html=True)
        st.code(masked, language="text")
        if masked != test_input:
            st.success("✅ PII Scrubbed: Sensitive employee identifiers safely redacted.")

    st.markdown('</div>', unsafe_allow_html=True)
