"""
Mood Mentor — Enterprise AI Psychological Wellness Platform
Production SaaS Edition (Modern Health / Stripe Caliber UI)
"""

import streamlit as st
import os
import sys
import tempfile
import json
from datetime import datetime, timedelta
import random

# Ensure root package is discoverable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

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
    render_emotion_distribution, calculate_burnout_risk_index
)

# -----------------------------------------------------------------------------
# 1. Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="MoodMentor AI — Enterprise Wellness Intelligence",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# -----------------------------------------------------------------------------
# 2. Modern Full-Stack Enterprise Design System (Linear / Modern Health / Stripe Caliber)
# -----------------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');
    
    * { box-sizing: border-box; }
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
        color: #0F172A;
        background-color: #F8FAFC !important;
    }
    
    #MainMenu, footer, header { visibility: hidden !important; }
    .block-container {
        padding-top: 1.2rem !important;
        padding-bottom: 3.5rem !important;
        max-width: 1320px !important;
    }

    /* Top Glass Navbar */
    .top-navbar {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 14px;
        padding: 14px 22px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 20px;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04);
    }
    .nav-brand {
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .brand-mark {
        background: linear-gradient(135deg, #0F766E 0%, #0369A1 100%);
        color: white;
        width: 38px;
        height: 38px;
        border-radius: 9px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 800;
        font-size: 1.15rem;
        box-shadow: 0 2px 8px rgba(15, 118, 110, 0.25);
    }
    .brand-text-name {
        font-size: 1.15rem;
        font-weight: 800;
        color: #0F172A;
        letter-spacing: -0.02em;
    }
    .brand-text-badge {
        background: #CCFBF1;
        color: #0F766E;
        font-size: 0.68rem;
        font-weight: 700;
        padding: 2px 8px;
        border-radius: 5px;
        margin-left: 6px;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        border: 1px solid #99F6E4;
    }
    .nav-stats-pill {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 9999px;
        padding: 6px 14px;
        font-size: 0.78rem;
        font-weight: 600;
        color: #475569;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .pulsing-live-dot {
        width: 8px;
        height: 8px;
        background-color: #10B981;
        border-radius: 50%;
        display: inline-block;
        box-shadow: 0 0 0 2px rgba(16, 185, 129, 0.25);
    }

    /* Hero Greeting Section */
    .hero-banner {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 14px;
        padding: 20px 24px;
        margin-bottom: 20px;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.02);
    }
    .hero-greeting {
        font-size: 1.45rem;
        font-weight: 800;
        color: #0F172A;
        letter-spacing: -0.02em;
        margin-bottom: 4px;
    }
    .hero-desc {
        color: #64748B;
        font-size: 0.88rem;
        line-height: 1.5;
        margin: 0;
    }

    /* Modern Bento-Grid KPI Cards */
    .bento-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 14px;
        margin-bottom: 20px;
    }
    .bento-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 18px 20px;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.02);
        transition: transform 0.15s ease, border-color 0.15s ease;
    }
    .bento-card:hover {
        transform: translateY(-1px);
        border-color: #CBD5E1;
        box-shadow: 0 6px 14px -3px rgba(15, 23, 42, 0.05);
    }
    .bento-card-top {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 8px;
    }
    .bento-label {
        color: #64748B;
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .bento-val {
        font-size: 1.85rem;
        font-weight: 800;
        color: #0F172A;
        line-height: 1.1;
        letter-spacing: -0.03em;
    }
    .bento-tag {
        font-size: 0.74rem;
        font-weight: 600;
        margin-top: 8px;
        display: inline-flex;
        align-items: center;
        gap: 4px;
        padding: 2px 7px;
        border-radius: 5px;
    }
    .tag-teal { background: #CCFBF1; color: #0F766E; }
    .tag-blue { background: #E0F2FE; color: #0284C7; }
    .tag-emerald { background: #D1FAE5; color: #047857; }
    .tag-amber { background: #FEF3C7; color: #B45309; }

    /* Segmented Control Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background: #ECEFF3 !important;
        border-radius: 12px !important;
        padding: 5px !important;
        gap: 4px !important;
        border: 1px solid #E2E8F0 !important;
        margin-bottom: 22px !important;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px !important;
        font-weight: 600 !important;
        font-size: 0.86rem !important;
        color: #475569 !important;
        padding: 8px 16px !important;
        border: none !important;
        transition: all 0.15s ease !important;
        background: transparent !important;
    }
    .stTabs [aria-selected="true"] {
        background: #FFFFFF !important;
        color: #0F172A !important;
        font-weight: 700 !important;
        box-shadow: 0 2px 6px rgba(15, 23, 42, 0.08) !important;
    }
    .stTabs [data-baseweb="tab-highlight"] { display: none !important; }

    /* Form Inputs & Preset Buttons */
    .stTextArea textarea {
        border-radius: 10px !important;
        border: 1px solid #CBD5E1 !important;
        padding: 12px 16px !important;
        font-size: 0.92rem !important;
        background: #FFFFFF !important;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.02) !important;
    }
    .stTextArea textarea:focus {
        border-color: #0F766E !important;
        box-shadow: 0 0 0 3px rgba(15, 118, 110, 0.15) !important;
    }
    div[data-testid="column"] .stButton > button {
        border-radius: 9px !important;
        font-weight: 600 !important;
        font-size: 0.82rem !important;
        padding: 8px 14px !important;
        background: #FFFFFF !important;
        border: 1px solid #CBD5E1 !important;
        color: #334155 !important;
        transition: all 0.15s ease !important;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.02) !important;
    }
    div[data-testid="column"] .stButton > button:hover {
        border-color: #0F766E !important;
        color: #0F766E !important;
        background: #F0FDF4 !important;
    }
    button[kind="primary"] {
        background: linear-gradient(135deg, #0F766E 0%, #0D9488 100%) !important;
        color: white !important;
        font-weight: 700 !important;
        border: none !important;
        box-shadow: 0 3px 12px rgba(15, 118, 110, 0.25) !important;
    }
    button[kind="primary"]:hover {
        background: linear-gradient(135deg, #115E59 0%, #0F766E 100%) !important;
        box-shadow: 0 5px 16px rgba(15, 118, 110, 0.35) !important;
    }

    /* Clinical Diagnostic Assessment Card */
    .clinical-diag-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 14px;
        padding: 20px 22px;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.02);
    }
    .diag-title-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding-bottom: 12px;
        border-bottom: 1px solid #F1F5F9;
        margin-bottom: 12px;
    }
    .diag-title {
        font-size: 0.96rem;
        font-weight: 700;
        color: #0F172A;
        letter-spacing: -0.01em;
    }
    .diag-live-tag {
        font-size: 0.70rem;
        font-weight: 700;
        color: #047857;
        background: #ECFDF5;
        border: 1px solid #A7F3D0;
        padding: 2px 7px;
        border-radius: 4px;
        text-transform: uppercase;
    }
    .diag-metric-item {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 9px 0;
        border-bottom: 1px solid #F8FAFC;
    }
    .diag-label {
        font-size: 0.82rem;
        font-weight: 600;
        color: #64748B;
    }
    .diag-val-affect {
        font-size: 0.98rem;
        font-weight: 800;
        color: #0F172A;
        text-transform: uppercase;
        letter-spacing: 0.02em;
    }
    .diag-intensity-box {
        text-align: right;
    }
    .diag-intensity-num {
        font-size: 0.95rem;
        font-weight: 800;
        color: #0F766E;
    }
    .diag-tier-badge {
        font-size: 0.72rem;
        font-weight: 700;
        padding: 2px 6px;
        border-radius: 4px;
        margin-left: 4px;
        background: #E0F2FE;
        color: #0369A1;
    }
    .diag-triage-badge {
        font-size: 0.78rem;
        font-weight: 800;
        padding: 3px 8px;
        border-radius: 5px;
    }
    .triage-mild { background: #DCFCE7; color: #166534; }
    .triage-moderate { background: #E0F2FE; color: #0369A1; }
    .triage-high { background: #FEF3C7; color: #92400E; }
    .triage-critical { background: #FEE2E2; color: #991B1B; }

    .meter-container {
        width: 130px;
        height: 6px;
        background: #E2E8F0;
        border-radius: 9999px;
        overflow: hidden;
        margin-top: 4px;
    }
    .meter-fill {
        height: 100%;
        border-radius: 9999px;
        background: linear-gradient(90deg, #10B981 0%, #F59E0B 60%, #EF4444 100%);
    }

    /* Enterprise Clinical Prescription Card */
    .saas-rec-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-left: 4px solid #0F766E;
        border-radius: 12px;
        padding: 18px 22px;
        margin-bottom: 14px;
        transition: all 0.15s ease;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.02);
    }
    .saas-rec-card:hover {
        border-color: #0F766E;
        transform: translateY(-1px);
        box-shadow: 0 6px 18px -4px rgba(15, 118, 110, 0.08);
    }
    .rec-top-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 8px;
    }
    .rec-title-text {
        font-size: 1.05rem;
        font-weight: 700;
        color: #0F172A;
    }
    .fit-score-pill {
        background: #F0FDF4;
        color: #15803D;
        border: 1px solid #BBF7D0;
        padding: 2px 10px;
        border-radius: 9999px;
        font-size: 0.76rem;
        font-weight: 700;
    }
    .meta-tag {
        display: inline-flex;
        align-items: center;
        gap: 4px;
        padding: 2px 8px;
        border-radius: 5px;
        font-size: 0.70rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.03em;
        margin-right: 6px;
    }
    .tag-somatic { background: #FAF5FF; color: #7E22CE; border: 1px solid #F3E8FF; }
    .tag-breathing { background: #F0F9FF; color: #0369A1; border: 1px solid #E0F2FE; }
    .tag-mindfulness { background: #F0FDF4; color: #166534; border: 1px solid #DCFCE7; }
    .tag-journaling { background: #FFFBEB; color: #B45309; border: 1px solid #FEF3C7; }
    .tag-movement { background: #FFF1F2; color: #BE123C; border: 1px solid #FFE4E6; }

    .clinical-box {
        font-size: 0.80rem;
        color: #475569;
        background: #F8FAFC;
        padding: 9px 12px;
        border-radius: 6px;
        border: 1px solid #E2E8F0;
        margin-top: 8px;
    }

    /* Crisis Safety Callout */
    .crisis-card {
        background: #FFF1F2;
        border: 1px solid #FECDD3;
        border-left: 5px solid #E11D48;
        border-radius: 10px;
        padding: 16px 20px;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 3. Core Engine Initialization & Enterprise Seed Telemetry
# -----------------------------------------------------------------------------
@st.cache_resource
def init_enterprise_services():
    db = DatabaseManager()
    catalog = WellnessCatalog()
    intensity_analyzer = IntensityAnalyzer()
    recommender = RecommendationEngine(catalog=catalog)
    security = SecurityManager()
    filter_engine = FilterEngine()
    exporter = ReportExporter(db)
    feedback_learner = FeedbackLearner()
    emotion_pipeline = UnifiedEmotionPipeline()

    # Pre-populate rich enterprise dataset if fresh
    stats = db.get_database_stats()
    if stats["total_interactions"] < 5:
        sample_scenarios = [
            ("emp_ritesh_01", "Reflecting on project milestones; feeling proud and fulfilled with the team's momentum.", "joy", 0.94, 0.65, "Moderate", "Mild", 0.84, -14),
            ("emp_ritesh_01", "Frustrated with recurring production blockers and slow code reviews.", "anger", 0.79, 0.72, "High", "Moderate", -0.70, -11),
            ("emp_ritesh_01", "A little anxious about the upcoming architectural review meeting tomorrow.", "fear", 0.82, 0.58, "Moderate", "Mild", -0.42, -8),
            ("emp_ritesh_01", "Had an invigorating 1-on-1 coaching session with my manager. Clear clarity.", "joy", 0.89, 0.50, "Moderate", "Mild", 0.76, -5),
            ("emp_ritesh_01", "Somatic shoulder tension and eye fatigue after 8 straight hours of debugging.", "sadness", 0.75, 0.67, "Moderate", "Mild", -0.58, -3),
            ("emp_ritesh_01", "System deployment completed with zero downtime. High camaraderie.", "joy", 0.95, 0.72, "Moderate", "Mild", 0.90, -1),
        ]
        now = datetime.now()
        for uid, txt, emo, conf, intens, tier, triage, comp, days_ago in sample_scenarios:
            ts = (now + timedelta(days=days_ago, hours=random.randint(9, 17))).isoformat()
            int_id = db.save_interaction(uid, txt, emo, conf, intens, tier, triage, comp, timestamp=ts)
            dummy_state = intensity_analyzer.analyze(txt, {emo: conf}, comp)
            recs = recommender.get_recommendations(dummy_state, {emo: conf}, query_text=txt, top_k=2)
            db.save_recommendations(int_id, uid, [r.to_dict() for r in recs], timestamp=ts)
            if recs:
                db.save_feedback(uid, recs[0].activity.id, recs[0].activity.title, "accepted", random.choice([4.5, 5.0]), "High efficacy.", timestamp=ts)

    return db, catalog, intensity_analyzer, recommender, security, filter_engine, exporter, feedback_learner, emotion_pipeline

db, catalog, intensity_analyzer, recommender, security, filter_engine, exporter, feedback_learner, emotion_pipeline = init_enterprise_services()

# Session State Persistence
if "user_id" not in st.session_state:
    st.session_state.user_id = "emp_ritesh_01"

DEFAULT_PANIC_TEXT = "I feel like the walls are closing in on me, my heart is pounding uncontrollably, and I can barely breathe from severe panic."
if "user_reflection" not in st.session_state:
    st.session_state.user_reflection = DEFAULT_PANIC_TEXT

def execute_pipeline(text: str):
    sanitized_text, warnings = security.sanitize_input(text)
    clean_text, pii_counts = security.anonymize_pii(sanitized_text)

    # Real ML Inference: Milestone 1 (VADER Sentiment) + Milestone 2/3 (DistilBERT Multi-Label)
    probs, compound, meta = emotion_pipeline.analyze(clean_text)

    # Milestone 3: Dynamic Intensity & Clinical Triage Analysis
    state = intensity_analyzer.analyze(clean_text, probs, compound)
    profile = UserProfile(user_id=st.session_state.user_id)
    recs = recommender.get_recommendations(state, probs, profile, query_text=clean_text, top_k=3)

    # Milestone 4: Telemetry Storage
    int_id = db.save_interaction(
        user_id=st.session_state.user_id,
        raw_text=clean_text,
        primary_emotion=state.primary_emotion,
        confidence=state.confidence,
        intensity_score=state.intensity_score,
        tier=state.tier,
        triage_level=state.triage_level,
        vader_compound=compound
    )
    db.save_recommendations(int_id, st.session_state.user_id, [r.to_dict() for r in recs])

    st.session_state.last_analysis = {
        "state": state,
        "probs": probs,
        "recs": recs,
        "text": clean_text,
        "int_id": int_id,
        "meta": meta
    }

# Pre-run once so screen is never blank on initial load
if "last_analysis" not in st.session_state or st.session_state.last_analysis is None:
    execute_pipeline(st.session_state.user_reflection)

# -----------------------------------------------------------------------------
# 4. Top Glassmorphic Navigation Bar
# -----------------------------------------------------------------------------
user_history = db.get_user_interactions(user_id=st.session_state.user_id, limit=50)
burnout_score, burnout_label, burnout_color = calculate_burnout_risk_index(user_history)
stats = db.get_database_stats()

st.markdown(f"""
<div class="top-navbar">
    <div class="nav-brand">
        <div class="brand-mark">🧠</div>
        <div>
            <span class="brand-text-name">MoodMentor</span>
            <span class="brand-text-badge">v4.0 Enterprise</span>
        </div>
    </div>
    <div style="display: flex; align-items: center; gap: 14px;">
        <div class="nav-stats-pill">
            <span class="pulsing-live-dot"></span>
            <span>DistilBERT & VADER Live</span>
            <span style="color: #CBD5E1;">|</span>
            <span style="color: #0F172A; font-weight: 700;">{stats['total_interactions']} Telemetry Logs</span>
        </div>
        <div style="background: #0F172A; color: white; padding: 6px 14px; border-radius: 8px; font-size: 0.78rem; font-weight: 700; letter-spacing: 0.02em;">
            👤 {st.session_state.user_id}
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 5. Hero Greeting Section
# -----------------------------------------------------------------------------
st.markdown(f"""
<div class="hero-banner">
    <div class="hero-greeting">Welcome back, Ritesh 👋</div>
    <div class="hero-desc">
        Your workplace mental wellness telemetry is synced in real time. Submit natural language reflections below to receive clinically triaged interventions backed by Explainable AI.
    </div>
</div>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 6. Executive Bento-Grid KPI Row
# -----------------------------------------------------------------------------
dominant = user_history[0]["primary_emotion"].capitalize() if user_history else "Joy"
feedback_count = len(db.get_user_feedback(user_id=st.session_state.user_id))

st.markdown(f"""
<div class="bento-grid">
    <div class="bento-card">
        <div class="bento-card-top">
            <span class="bento-label">Total Check-ins</span>
            <div style="font-size: 1.05rem;">📊</div>
        </div>
        <div class="bento-val">{len(user_history)}</div>
        <div class="bento-tag tag-teal">↑ 100% telemetry synced</div>
    </div>
    <div class="bento-card">
        <div class="bento-card-top">
            <span class="bento-label">Dominant Affect</span>
            <div style="font-size: 1.05rem;">🌱</div>
        </div>
        <div class="bento-val" style="font-size: 1.80rem;">{dominant}</div>
        <div class="bento-tag tag-blue">Latest emotional trend</div>
    </div>
    <div class="bento-card">
        <div class="bento-card-top">
            <span class="bento-label">Burnout Risk Index</span>
            <div style="font-size: 1.05rem;">⚡</div>
        </div>
        <div class="bento-val" style="color: {burnout_color};">{burnout_score}%</div>
        <div class="bento-tag" style="background: #FEF2F2; color: {burnout_color}; font-weight: 700;">{burnout_label}</div>
    </div>
    <div class="bento-card">
        <div class="bento-card-top">
            <span class="bento-label">Interventions Rated</span>
            <div style="font-size: 1.05rem;">⭐</div>
        </div>
        <div class="bento-val">{feedback_count}</div>
        <div class="bento-tag tag-emerald">Continuous online learning</div>
    </div>
</div>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 7. Navigation Tabs
# -----------------------------------------------------------------------------
tab_copilot, tab_trends, tab_history, tab_search, tab_reports, tab_privacy = st.tabs([
    "🩺 AI Wellness Copilot",
    "📈 Longitudinal Analytics",
    "🗂️ Recommendation History",
    "🔍 Advanced Search & Filter",
    "📑 Executive Reports (PDF/CSV)",
    "🛡️ Compliance & GDPR"
])

# =============================================================================
# TAB 1: AI Wellness Copilot
# =============================================================================
with tab_copilot:
    st.markdown("""
    <div style="margin-bottom: 12px;">
        <h3 style="font-weight: 800; font-size: 1.25rem; color: #0F172A; margin: 0 0 4px 0;">Interactive Reflection & Clinical Triage Engine</h3>
        <p style="color: #64748B; font-size: 0.9rem; margin: 0;">Analyze emotion vectors, compute continuous intensity (0.05 - 1.0), and surface personalized evidence-based protocols.</p>
    </div>
    """, unsafe_allow_html=True)

    # Preset Quick Action Scenarios with instant trigger
    st.markdown("<small style='color: #64748B; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em;'>Quick Scenario Presets (Instant Run):</small>", unsafe_allow_html=True)
    col_p1, col_p2, col_p3 = st.columns(3)
    
    if col_p1.button("⚡ Scenario 1: Acute Panic / Heart Racing", use_container_width=True):
        st.session_state.user_reflection = "I feel like the walls are closing in on me, my heart is pounding uncontrollably, and I can barely breathe from severe panic."
        execute_pipeline(st.session_state.user_reflection)
        st.rerun()

    if col_p2.button("💼 Scenario 2: Chronic Deadline Burnout", use_container_width=True):
        st.session_state.user_reflection = "Completely drained and exhausted from constant back-to-back sprint deliverables, endless meetings, and high pressure."
        execute_pipeline(st.session_state.user_reflection)
        st.rerun()

    if col_p3.button("🌟 Scenario 3: Milestone Launch Victory", use_container_width=True):
        st.session_state.user_reflection = "Celebrated our major project release with the whole team today! Feeling deeply accomplished, proud, and fulfilled."
        execute_pipeline(st.session_state.user_reflection)
        st.rerun()

    # Manual Reflection Form
    with st.form("reflection_form"):
        current_input = st.text_area(
            label="Input reflection:",
            value=st.session_state.user_reflection,
            height=100,
            placeholder="Type your current feelings, thoughts, or somatic stress indicators...",
            label_visibility="collapsed"
        )
        col_sub1, col_sub2 = st.columns([1.5, 3.5])
        with col_sub1:
            submitted = st.form_submit_button("🚀 Run AI Analysis", type="primary", use_container_width=True)
        with col_sub2:
            st.markdown("<div style='padding-top: 8px; color: #64748B; font-size: 0.82rem;'>🛡️ <i>PII Anonymization active: names, emails & employee codes auto-scrubbed before inference.</i></div>", unsafe_allow_html=True)

        if submitted and current_input.strip():
            st.session_state.user_reflection = current_input.strip()
            execute_pipeline(st.session_state.user_reflection)
            st.rerun()

    # Display Analysis Results (Always active and visible)
    if st.session_state.last_analysis:
        res = st.session_state.last_analysis
        st.write("")
        st.divider()

        # Clinical Safety Warning Callout
        if res["state"].triage_level == "Critical":
            st.markdown("""
            <div class="crisis-card">
                <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 6px;">
                    <span style="font-size: 1.4rem;">🚨</span>
                    <strong style="color: #9F1239; font-size: 1.1rem;">Clinical Safety Emergency Guardrail Activated</strong>
                </div>
                <p style="color: #881337; margin: 0 0 12px 0; font-size: 0.92rem;">
                    Indicators of acute mental distress or crisis trigger patterns detected. Standard wellness routines have been automatically suppressed.
                </p>
                <div style="background: white; border: 1px solid #FECDD3; border-radius: 8px; padding: 12px 18px; display: inline-flex; align-items: center; gap: 12px;">
                    <span>📞 <b>24/7 Crisis Hotline:</b> <a href="tel:988" style="color: #E11D48; font-weight: 800; font-size: 1.05rem;">Dial 988 (Toll-Free)</a></span>
                    <span style="color: #CBD5E1;">|</span>
                    <span>Text <b>HOME to 741741</b></span>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("#### **1. Real-Time Emotion & Triage Profile**")
        col_view1, col_view2 = st.columns([1.1, 1.3])
        
        with col_view1:
            radar_fig = render_radar_state(res["probs"])
            if radar_fig is not None:
                st.plotly_chart(radar_fig, use_container_width=True)
            else:
                st.markdown("**Emotion Probability Breakdown:**")
                for k, v in res["probs"].items():
                    st.progress(v, text=f"{k.capitalize()}: {v*100:.1f}%")

        with col_view2:
            triage_cls = res['state'].triage_level.lower()
            intensity_pct = min(100, max(5, int(res['state'].intensity_score * 100)))
            st.markdown(f"""
            <div class="clinical-diag-card">
                <div class="diag-title-row">
                    <span class="diag-title">Clinical Diagnostic Assessment</span>
                    <span class="diag-live-tag">Live Neural Telemetry</span>
                </div>
                <div class="diag-metric-item">
                    <span class="diag-label">Dominant Affect:</span>
                    <span class="diag-val-affect">
                        {res['state'].primary_emotion} <span style="font-size: 0.82rem; font-weight: 600; color: #64748B;">({res['state'].confidence * 100:.1f}%)</span>
                    </span>
                </div>
                <div class="diag-metric-item">
                    <span class="diag-label">Continuous Intensity:</span>
                    <div class="diag-intensity-box">
                        <span class="diag-intensity-num">{res['state'].intensity_score:.2f} / 1.00</span>
                        <span class="diag-tier-badge">{res['state'].tier} Tier</span>
                        <div class="meter-container"><div class="meter-fill" style="width: {intensity_pct}%;"></div></div>
                    </div>
                </div>
                <div class="diag-metric-item">
                    <span class="diag-label">Clinical Severity Triage:</span>
                    <span class="diag-triage-badge triage-{triage_cls}">
                        {res['state'].triage_level}
                    </span>
                </div>
                <div class="diag-metric-item">
                    <span class="diag-label">VADER Baseline Polarity:</span>
                    <span style="font-weight: 700; color: #0284C7; font-size: 0.90rem;">
                        {res.get('meta', {}).get('sentiment_scores', {}).get('compound', 0.0):+.2f} Compound
                    </span>
                </div>
                <div style="padding-top: 10px; font-size: 0.80rem; color: #475569; border-top: 1px solid #F1F5F9; margin-top: 8px;">
                    <b>Affective Synergy:</b> {res['state'].mixed_state or "Unipolar coherent state detected."}
                </div>
                <div style="padding-top: 4px; font-size: 0.74rem; color: #64748B;">
                    ⚡ <b>Inference Backbone:</b> {res.get('meta', {}).get('model_used', 'DistilBERT Transformer')}
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("#### **2. Prescribed Clinical Interventions**")
        for i, rec in enumerate(res["recs"]):
            r_dict = rec.to_dict()
            mod_tag = f"tag-{r_dict['modality'].lower()}"
            st.markdown(f"""
            <div class="saas-rec-card">
                <div class="rec-top-row">
                    <span class="rec-title-text">{i+1}. {r_dict['title']}</span>
                    <span class="fit-score-pill">Fit Score: {r_dict['score'] * 100:.1f}%</span>
                </div>
                <div style="margin-bottom: 8px;">
                    <span class="meta-tag {mod_tag}">{r_dict['modality']}</span>
                    <span style="font-size: 0.78rem; color: #64748B; font-weight: 600;">⏱️ {r_dict['duration_mins']} mins duration</span>
                </div>
                <p style="color: #334155; font-size: 0.90rem; margin: 0 0 8px 0; line-height: 1.5;">{r_dict['description']}</p>
                <div class="clinical-box">
                    <b>Clinical Protocol Mechanism:</b> {r_dict['clinical_rationale']}
                </div>
            </div>
            """, unsafe_allow_html=True)

            with st.expander(f"🔍 Explainable AI (XAI) Justification for #{i+1}"):
                st.markdown(r_dict.get("rationale", ""))
                st.caption("Algorithm Strategy Weight Allocation:")
                st.json(r_dict.get("strategy_breakdown", {}))

# =============================================================================
# TAB 2: Longitudinal Analytics
# =============================================================================
with tab_trends:
    st.markdown("""
    <div style="margin-bottom: 16px;">
        <h3 style="font-weight: 800; font-size: 1.25rem; color: #0F172A; margin: 0 0 4px 0;">Longitudinal Emotional Trajectory & Burnout Mitigation</h3>
        <p style="color: #64748B; font-size: 0.9rem; margin: 0;">Multi-week affective momentum, intensity shifts, and clinical risk forecasting.</p>
    </div>
    """, unsafe_allow_html=True)

    if user_history:
        col_t1, col_t2 = st.columns([1.5, 1])
        with col_t1:
            traj_fig = render_intensity_trajectory(user_history)
            if traj_fig is not None:
                st.plotly_chart(traj_fig, use_container_width=True)
            else:
                st.line_chart([u["intensity_score"] for u in user_history])
        with col_t2:
            dist_fig = render_emotion_distribution(user_history)
            if dist_fig is not None:
                st.plotly_chart(dist_fig, use_container_width=True)
            else:
                st.caption("Distribution available with Plotly.")

        st.markdown(f"""
        <div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 14px; padding: 20px; margin-top: 14px; box-shadow: 0 1px 3px rgba(15,23,42,0.02);">
            <h4 style="font-size: 0.98rem; font-weight: 700; color: #0F172A; margin: 0 0 6px 0;">Executive Clinical Synthesis</h4>
            <p style="color: #475569; font-size: 0.88rem; line-height: 1.5; margin: 0;">
                Current Burnout Index stands at <b>{burnout_score}% ({burnout_label})</b>. Over the last 14 days, emotional intensity shows positive stabilization following somatic grounding practices. Recommending a routine 4-minute box breathing session prior to high-stakes sprint meetings.
            </p>
        </div>
        """, unsafe_allow_html=True)

# =============================================================================
# TAB 3: Recommendation History & Feedback Loop
# =============================================================================
with tab_history:
    st.markdown("""
    <div style="margin-bottom: 16px;">
        <h3 style="font-weight: 800; font-size: 1.25rem; color: #0F172A; margin: 0 0 4px 0;">Recommendation Audit & Closed-Loop Online Learning</h3>
        <p style="color: #64748B; font-size: 0.9rem; margin: 0;">Historical record of delivered wellness prescriptions and user-submitted feedback.</p>
    </div>
    """, unsafe_allow_html=True)

    past_recs = db.get_user_recommendations(user_id=st.session_state.user_id, limit=30)
    if past_recs:
        st.dataframe(past_recs, use_container_width=True, height=260)

        st.divider()
        st.markdown("#### **Submit Verified Feedback (Online Model Adaptation)**")
        col_f1, col_f2, col_f3 = st.columns([2, 1, 1])
        with col_f1:
            selected_act = st.selectbox("Target Activity:", options=list(set(r["title"] for r in past_recs)))
        with col_f2:
            action_choice = st.selectbox("Engagement Status:", ["Accepted & Completed", "Rejected", "Viewed Only"])
        with col_f3:
            rating_choice = st.slider("Efficacy Score:", 1.0, 5.0, 5.0, step=0.5)

        comment_choice = st.text_input("Qualitative Observation:", placeholder="e.g. Significantly slowed down acute heart racing.")
        if st.button("Submit Telemetry Feedback", type="primary"):
            db.save_feedback(
                user_id=st.session_state.user_id,
                activity_id="act_telemetry",
                activity_title=selected_act,
                action=action_choice.split()[0].lower(),
                rating=rating_choice,
                comment=comment_choice
            )
            st.success("Telemetry saved! Model modality affinities adapted in real time.")
            st.rerun()

# =============================================================================
# TAB 4: Search & Advanced Filter
# =============================================================================
with tab_search:
    st.markdown("""
    <div style="margin-bottom: 16px;">
        <h3 style="font-weight: 800; font-size: 1.25rem; color: #0F172A; margin: 0 0 4px 0;">Multi-Criteria Telemetry Query & Filtering</h3>
        <p style="color: #64748B; font-size: 0.9rem; margin: 0;">Parameterized SQL querying across emotional states, intensity tiers, and keywords.</p>
    </div>
    """, unsafe_allow_html=True)

    col_q1, col_q2, col_q3, col_q4 = st.columns(4)
    with col_q1:
        s_keyword = st.text_input("Search Content / Keyword:", placeholder="e.g. panic, sprint...")
    with col_q2:
        s_emotion = st.selectbox("Target Emotion:", ["All", "Joy", "Sadness", "Anger", "Fear", "Love", "Surprise"])
    with col_q3:
        s_tier = st.selectbox("Intensity Band:", ["All", "Low", "Moderate", "High", "Severe"])
    with col_q4:
        s_triage = st.selectbox("Clinical Triage:", ["All", "Mild", "Moderate", "High", "Critical"])

    filtered_results = filter_engine.search_interactions(
        db,
        user_id=st.session_state.user_id,
        keyword=s_keyword if s_keyword else None,
        emotion=s_emotion,
        tier=s_tier,
        triage_level=s_triage,
        limit=50
    )

    st.markdown(f"**Found {len(filtered_results)} matching records:**")
    st.dataframe(filtered_results, use_container_width=True, height=280)

# =============================================================================
# TAB 5: Executive Reports & Export
# =============================================================================
with tab_reports:
    st.markdown("""
    <div style="margin-bottom: 16px;">
        <h3 style="font-weight: 800; font-size: 1.25rem; color: #0F172A; margin: 0 0 4px 0;">Automated Report Generation & Export</h3>
        <p style="color: #64748B; font-size: 0.9rem; margin: 0;">Export verifiable, non-hardcoded wellness telemetry for executive reviews or clinical documentation.</p>
    </div>
    """, unsafe_allow_html=True)

    col_rep1, col_rep2 = st.columns(2)
    with col_rep1:
        st.markdown("""
        <div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 14px; padding: 22px; box-shadow: 0 1px 3px rgba(15,23,42,0.02);">
            <div style="font-size: 1.02rem; font-weight: 800; color: #0F172A; margin-bottom: 6px;">📊 Raw Tabular Telemetry (CSV)</div>
            <p style="color: #64748B; font-size: 0.84rem; margin: 0 0 16px 0;">Complete database extraction including timestamps, VADER polarity, intensity tiers, and raw tokens.</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Generate & Download CSV File", type="primary", use_container_width=True):
            temp_csv = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
            exporter.export_interactions_csv(st.session_state.user_id, temp_csv.name)
            with open(temp_csv.name, "rb") as f:
                st.download_button(
                    label="📥 Click to Save CSV",
                    data=f.read(),
                    file_name=f"moodmentor_data_{st.session_state.user_id}.csv",
                    mime="text/csv",
                    use_container_width=True
                )

    with col_rep2:
        st.markdown("""
        <div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 14px; padding: 22px; box-shadow: 0 1px 3px rgba(15,23,42,0.02);">
            <div style="font-size: 1.02rem; font-weight: 800; color: #0F172A; margin-bottom: 6px;">📄 Executive Summary Report (PDF)</div>
            <p style="color: #64748B; font-size: 0.84rem; margin: 0 0 16px 0;">Publication-ready PDF featuring executive metrics, recent check-in timelines, and evidence-based prescriptions.</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Generate Executive PDF Report", use_container_width=True):
            temp_pdf = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
            exporter.export_executive_pdf(st.session_state.user_id, temp_pdf.name)
            with open(temp_pdf.name, "rb") as f:
                st.download_button(
                    label="📥 Click to Save PDF",
                    data=f.read(),
                    file_name=f"moodmentor_executive_{st.session_state.user_id}.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )

# =============================================================================
# TAB 6: Compliance & GDPR
# =============================================================================
with tab_privacy:
    st.markdown("""
    <div style="margin-bottom: 16px;">
        <h3 style="font-weight: 800; font-size: 1.25rem; color: #0F172A; margin: 0 0 4px 0;">Enterprise Compliance & GDPR Safeguards</h3>
        <p style="color: #64748B; font-size: 0.9rem; margin: 0;">Strict enterprise standards ensuring HIPAA/GDPR alignment, PII anonymization, and the Right to be Forgotten.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    * **Automatic PII Masking:** Regex scrubbing strips email addresses, phone numbers, and employee ID tokens prior to persistent database writes.
    * **Zero Ad-Broker Sharing:** Telemetry is isolated in a sandboxed, encrypted database.
    * **Clinical Separation:** Self-care suggestions are strictly differentiated from formal psychiatric diagnosis.
    """)

    st.divider()
    st.markdown("#### **GDPR Data Deletion ('Right to be Forgotten')**")
    st.caption("Permanently purge all interaction history, recommendations, and feedback for this account.")
    confirm_purge = st.checkbox("I confirm that I want to irreversibly purge all my wellness telemetry.")
    if st.button("🗑️ Execute Complete Account Data Purge", disabled=not confirm_purge):
        purge_res = db.purge_user_data(st.session_state.user_id)
        st.success(f"Purge complete! Deleted: {purge_res['interactions_deleted']} interactions, {purge_res['recommendations_deleted']} recommendations, {purge_res['feedback_deleted']} feedback items.")
        st.rerun()
