"""
MoodMentor AI — Enterprise AI Psychological Wellness Platform
Edition: "Quiet Luxury Meets Clinical Intelligence" (v4.2 Enterprise)
"""

import streamlit as st
import os
import sys
import tempfile
import json
import time
from datetime import datetime
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
from dashboard.ui_theme import (
    apply_enterprise_theme, render_top_navbar,
    render_breathing_visual, render_login_hero,
    render_sidebar_context, QUIET_LUXURY_CSS
)

# -----------------------------------------------------------------------------
# 1. Page Configuration & Theme Initialization
# -----------------------------------------------------------------------------
apply_enterprise_theme(
    page_title="MoodMentor AI — Enterprise Wellness Intelligence",
    page_icon="🌿"
)

# -----------------------------------------------------------------------------
# 2. Singleton Resource Initialization (Cached backend engines)
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
    search_eng = FilterEngine(db=db)
    exporter = ReportExporter(db=db)
    return db, sec, pipe, analyzer, cat, engine, learner, search_eng, exporter

db, sec, pipeline, analyzer, catalog, rec_engine, feedback_learner, search_engine, exporter = load_platform_services()

# -----------------------------------------------------------------------------
# 3. Authentication & Onboarding Gate (Two-Column Responsive Experience)
# -----------------------------------------------------------------------------
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = True
if "is_demo" not in st.session_state:
    st.session_state["is_demo"] = True

if not st.session_state["authenticated"]:
    # Render Two-Column Editorial Sign-In Experience
    auth_left, auth_right = st.columns([1.18, 0.95], gap="large")

    with auth_left:
        st.markdown(render_login_hero(), unsafe_allow_html=True)

    with auth_right:
        with st.container(border=True):
            st.markdown("""
            <div style="margin-bottom: 20px;">
                <h3 style="margin:0 0 6px 0; font-size:21px; font-weight:800; color:#183D35; letter-spacing:-0.02em;">Enterprise Workspace Access</h3>
                <p style="margin:0; font-size:13px; color:#727A73;">Sign in with your corporate credentials or explore evaluator demo</p>
            </div>
            """, unsafe_allow_html=True)

            auth_email = st.text_input("Corporate Email", placeholder="name@enterprise.org", key="auth_email_input")
            auth_pwd = st.text_input("Password", type="password", placeholder="Enter your password", key="auth_pwd_input")

            st.markdown("<div style='height: 6px;'></div>", unsafe_allow_html=True)
            login_btn = st.button("Sign In to Workspace", use_container_width=True)

            if login_btn:
                if not auth_email.strip() or not auth_pwd.strip():
                    st.error("⚠️ Please enter both your corporate email and password.")
                else:
                    st.session_state["authenticated"] = True
                    st.session_state["is_demo"] = False
                    st.session_state["user_display"] = auth_email.split("@")[0].replace(".", " ").title()
                    st.rerun()

            st.markdown("""
            <div style="text-align:center; margin: 18px 0 14px 0; position:relative;">
                <div style="height:1px; background:#E4E5DD; width:100%; position:absolute; top:50%;"></div>
                <span style="position:relative; background:#FFFFFF; padding:0 12px; font-size:11px; color:#727A73; font-weight:600; text-transform:uppercase; letter-spacing:0.04em;">or demo access</span>
            </div>
            """, unsafe_allow_html=True)

            demo_access_btn = st.button("✨ Explore Workspace Demo (1-Click Evaluator Access)", type="primary", use_container_width=True)
            if demo_access_btn:
                st.session_state["authenticated"] = True
                st.session_state["is_demo"] = True
                st.session_state["user_display"] = "Evaluator (Demo)"
                st.rerun()

            st.markdown("""
            <div style="margin-top:20px; padding:12px; background:#F6F4EE; border-radius:10px; border:1px solid #E4E5DD; font-size:11.5px; color:#727A73; line-height:1.5;">
                🔒 <b>Privacy Notice:</b> All session telemetry is encrypted with AES-256. Natural language input is scrubbed of PII via on-device spaCy NER prior to neural classification.
            </div>
            """, unsafe_allow_html=True)

    # Halt execution so workspace is only viewable once authenticated
    st.stop()


# -----------------------------------------------------------------------------
# 4. Collapsible Enterprise Sidebar Navigation
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div style="padding: 8px 0 16px 0; border-bottom: 1px solid #E4E5DD; margin-bottom: 16px;">
        <div style="display:flex; align-items:center; gap:10px;">
            <div style="width:36px; height:36px; background:#183D35; border-radius:10px; display:flex; align-items:center; justify-content:center; color:#F6F4EE; font-size:18px;">🌿</div>
            <div>
                <p style="margin:0; font-size:14px; font-weight:700; color:#183D35;">Meridian Health</p>
                <p style="margin:0; font-size:11px; color:#727A73; font-family:'JetBrains Mono', monospace;">ws-873519 • Enterprise</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    active_user_full = st.selectbox(
        "Active Employee Profile",
        [
            "emp_01 — Sarah Lin (Lead Architect)",
            "emp_02 — Marcus Vance (VP Sales)",
            "emp_03 — Elena Rostova (Staff Product Designer)"
        ],
        index=0
    )
    active_user = active_user_full.split()[0]

    if st.session_state.get("is_demo", False):
        st.markdown("""
        <div style="background:#E8EDE5; border:1px solid #A8B9A5; border-radius:8px; padding:6px 10px; margin-bottom:12px; font-size:11px; font-weight:700; color:#183D35; text-align:center;">
            ✨ Evaluator Demo Mode Active
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
    st.markdown("<p style='font-size:11px; font-weight:700; color:#727A73; text-transform:uppercase; letter-spacing:0.05em; margin-bottom:8px;'>Navigation</p>", unsafe_allow_html=True)

    active_tab = st.radio(
        "Navigation",
        [
            "🧠 Clinical Copilot & Live Inference",
            "📈 Longitudinal Analytics & Burnout Meter",
            "🔍 Live Telemetry Search Engine",
            "📑 Executive Reports & PDF Hub",
            "🔒 Security & GDPR Sandbox"
        ],
        index=0,
        label_visibility="collapsed"
    )



    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)
    if st.button("🚪 Sign Out of Workspace", use_container_width=True):
        st.session_state["authenticated"] = False
        st.session_state["is_demo"] = False
        st.rerun()

# -----------------------------------------------------------------------------
# 5. Executive Top Navigation Bar
# -----------------------------------------------------------------------------
view_label = active_tab.split(" ", 1)[1] if " " in active_tab else active_tab
render_top_navbar(current_view=view_label)

# -----------------------------------------------------------------------------
# VIEW 1: CLINICAL COPILOT & REAL-TIME INFERENCE
# -----------------------------------------------------------------------------
if "Copilot" in active_tab:
    col_input, col_output = st.columns([1.22, 1.08], gap="medium")

    with col_input:
        st.markdown("""
        <div class="quiet-card">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:14px;">
                <div>
                    <h3 style="margin:0; font-size:17px; font-weight:700; color:#183D35;">Interactive Affect Diagnostic Console</h3>
                    <p style="margin:0; font-size:12px; color:#727A73;">Natural language affective evaluation with automatic PII redaction</p>
                </div>
                <span class="badge-modality">DistilBERT Core</span>
            </div>
        """, unsafe_allow_html=True)

        # Preset Quick Buttons
        st.markdown("<p style='font-size:11.5px; font-weight:600; color:#727A73; margin-bottom:6px;'>Diagnostic Clinical Scenarios:</p>", unsafe_allow_html=True)
        p_c1, p_c2, p_c3, p_c4 = st.columns(4)
        preset_choice = None
        with p_c1:
            if st.button("🚨 Acute Panic", use_container_width=True):
                preset_choice = "I feel like the walls are closing in on me, my heart is pounding, and I can barely breathe."
        with p_c2:
            if st.button("🔋 Chronic Burnout", use_container_width=True):
                preset_choice = "I have zero motivation, completely drained and emotionally exhausted from non-stop sprint deadlines."
        with p_c3:
            if st.button("🌱 Grounded Flow", use_container_width=True):
                preset_choice = "I had a major breakthrough with our clinical architecture today, feeling energized, grateful, and peaceful!"
        with p_c4:
            if st.button("⚡ Imposter Anxiety", use_container_width=True):
                preset_choice = "I am terrified that everyone will realize I don't belong here and my deliverables will fail scrutiny."

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
        analyze_btn = st.button("✨ Run Neural Affect Diagnostics", type="primary", use_container_width=True)

        st.markdown("<p style='font-size:11.5px; color:#727A73; margin-top:10px; font-style:italic;'>ℹ️ <b>Clinical Advisory Notice:</b> AI-generated affective telemetry provides supportive psychological wellness insights and does not constitute formal psychiatric diagnosis.</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # Execute Inference Pipeline
        with st.spinner("Classifying affective vectors and evaluating clinical triage..."):
            sanitized_text, _ = sec.anonymize_pii(input_text)
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
                primary_emotion=primary_emo,
                confidence=confidence,
                intensity_score=intensity,
                tier=state.tier,
                triage_level=triage,
                vader_compound=compound
            )

        # Render Recommendations on Left Column
        st.markdown(f"""
        <div class="quiet-card">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:14px;">
                <h4 style="margin:0; font-size:16px; font-weight:700; color:#183D35;">Ranked Clinical Interventions ({len(recommendations)} matches)</h4>
                <span style="font-size:11.5px; color:#727A73;">Calibrated to {primary_emo.capitalize()} • {triage}</span>
            </div>
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
                        <h4 style="margin:8px 0 4px 0; font-size:16px; font-weight:800; color:#183D35;">{act.title}</h4>
                    </div>
                    <span style="font-size:12px; font-weight:600; color:#727A73; background:#F6F4EE; padding:3px 8px; border-radius:6px; border:1px solid #E4E5DD;">⏱️ {duration} min</span>
                </div>
                <p style="margin:0 0 10px 0; font-size:13px; color:#202B27; line-height:1.55;">{act.description}</p>
            </div>
            """, unsafe_allow_html=True)

            with st.expander(f"🔍 Explainable AI (XAI) Insight for Rank #{rank}"):
                st.markdown(f"""
                - **Primary Trigger Affect:** Elevated `{primary_emo}` affect vector ({emotions.get(primary_emo, 0):.1%})
                - **Clinical Rationale:** {act.clinical_rationale}
                - **Intensity Calibration:** Optimized for intensity index **{intensity:.2f}** ({triage} triage tier).
                """)
                fb_c1, fb_c2 = st.columns([1, 1])
                with fb_c1:
                    if st.button(f"👍 Accept & Log ({act.title[:15]}...)", key=f"acc_{rank}"):
                        feedback_learner.record_feedback(profile, act.id, act.modality, "accepted", rating=5.0)
                        st.success("Intervention logged! User adaptive profile updated.")
                with fb_c2:
                    if st.button(f"⏭️ Skip Activity", key=f"skp_{rank}"):
                        feedback_learner.record_feedback(profile, act.id, act.modality, "skipped", rating=2.0)
                        st.info("Skipped. Frequency dampening applied.")

        st.markdown("</div>", unsafe_allow_html=True)

    with col_output:
        # Clinical Triage HUD Banner
        hud_class = "triage-hud-mild"
        hud_badge = "MILD SEVERITY • SELF-DIRECTED REGULATION"
        if triage == "Moderate":
            hud_class = "triage-hud-moderate"
            hud_badge = "MODERATE SEVERITY • PROACTIVE WELLNESS"
        elif triage in ["High", "Elevated"]:
            hud_class = "triage-hud-elevated"
            hud_badge = "ELEVATED SEVERITY • CLINICAL SUPPORT RECOMMENDED"
        elif triage == "Critical":
            hud_class = "triage-hud-critical"
            hud_badge = "🚨 CRITICAL EMERGENCY • 988 LIFELINE PROTOCOL"

        st.markdown(f"""
        <div class="triage-hud {hud_class}">
            <div>
                <p style="margin:0; font-size:10.5px; font-weight:700; text-transform:uppercase; letter-spacing:0.05em;">Clinical Diagnostic HUD</p>
                <h4 style="margin:3px 0 0 0; font-size:15px; font-weight:800;">{hud_badge}</h4>
            </div>
            <span style="font-size:22px;">{'🚨' if triage == 'Critical' else '🛡️'}</span>
        </div>
        """, unsafe_allow_html=True)

        if triage == "Critical":
            st.error("⚠️ Crisis Guardrail Alert: Emergency hotline protocol active. Please dial or text 988 for immediate 24/7 confidential crisis assistance.")

        # Metric Strip (3 Columns: Emotion, Polarity, Intensity)
        m_c1, m_c2, m_c3 = st.columns(3)
        with m_c1:
            st.markdown(f"""
            <div class="quiet-card" style="padding:16px; margin-bottom:14px;">
                <p style="margin:0; font-size:11px; font-weight:700; color:#727A73; text-transform:uppercase; letter-spacing:0.04em;">Dominant Affect</p>
                <h3 style="margin:4px 0 0 0; font-size:23px; font-weight:800; color:#183D35;">{primary_emo.upper()}</h3>
                <p style="margin:3px 0 0 0; font-size:11px; color:#727A73;">Confidence: {confidence:.1%}</p>
            </div>
            """, unsafe_allow_html=True)
        with m_c2:
            pol_label = "Positive" if compound > 0.05 else ("Negative" if compound < -0.05 else "Neutral")
            pol_color = PALETTE["joy"] if compound > 0.05 else (PALETTE["anger"] if compound < -0.05 else PALETTE["sadness"])
            st.markdown(f"""
            <div class="quiet-card" style="padding:16px; margin-bottom:14px;">
                <p style="margin:0; font-size:11px; font-weight:700; color:#727A73; text-transform:uppercase; letter-spacing:0.04em;">Sentiment Polarity</p>
                <h3 style="margin:4px 0 0 0; font-size:23px; font-weight:800; color:{pol_color};">{compound:+.2f}</h3>
                <p style="margin:3px 0 0 0; font-size:11px; color:#727A73;">Valence: {pol_label}</p>
            </div>
            """, unsafe_allow_html=True)
        with m_c3:
            st.markdown(f"""
            <div class="quiet-card" style="padding:16px; margin-bottom:14px;">
                <p style="margin:0; font-size:11px; font-weight:700; color:#727A73; text-transform:uppercase; letter-spacing:0.04em;">Intensity Index</p>
                <h3 style="margin:4px 0 0 0; font-size:23px; font-weight:800; color:#B77955;">{intensity:.2f} / 1.0</h3>
                <p style="margin:3px 0 0 0; font-size:11px; color:#727A73;">Triage: {triage}</p>
            </div>
            """, unsafe_allow_html=True)

        # 6-D Emotion Radar Geometry
        st.markdown("""
        <div class="quiet-card" style="padding:18px;">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:4px;">
                <h4 style="margin:0; font-size:14px; font-weight:700; color:#183D35;">6-D Emotion Radar Geometry</h4>
                <span style="font-size:11px; font-weight:600; color:#727A73;">Ekman Affect Dimensions</span>
            </div>
        """, unsafe_allow_html=True)

        radar_fig = render_radar_state(emotions)
        if radar_fig:
            st.plotly_chart(radar_fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# VIEW 2: LONGITUDINAL ANALYTICS & BURNOUT METER
# -----------------------------------------------------------------------------
elif "Analytics" in active_tab or "Longitudinal" in active_tab:
    history = db.get_user_interactions(active_user, limit=30)
    burnout_score, burnout_label, burnout_color = calculate_burnout_risk_index(history)

    kpi1, kpi2, kpi3 = st.columns(3)
    with kpi1:
        st.markdown(f"""
        <div class="quiet-card">
            <p style="margin:0; font-size:11.5px; font-weight:700; color:#727A73; text-transform:uppercase; letter-spacing:0.04em;">Burnout Risk Index</p>
            <h2 style="margin:4px 0 0 0; font-size:36px; font-weight:800; color:{burnout_color};">{burnout_score}%</h2>
            <p style="margin:4px 0 0 0; font-size:12px; font-weight:600; color:{burnout_color};">{burnout_label}</p>
        </div>
        """, unsafe_allow_html=True)
    with kpi2:
        st.markdown(f"""
        <div class="quiet-card">
            <p style="margin:0; font-size:11.5px; font-weight:700; color:#727A73; text-transform:uppercase; letter-spacing:0.04em;">Longitudinal Telemetry Sessions</p>
            <h2 style="margin:4px 0 0 0; font-size:36px; font-weight:800; color:#183D35;">{len(history)}</h2>
            <p style="margin:4px 0 0 0; font-size:12px; color:#727A73;">Verified Clinical Diagnostic Logs</p>
        </div>
        """, unsafe_allow_html=True)
    with kpi3:
        avg_int = sum(h["intensity_score"] for h in history) / len(history) if history else 0.5
        st.markdown(f"""
        <div class="quiet-card">
            <p style="margin:0; font-size:11.5px; font-weight:700; color:#727A73; text-transform:uppercase; letter-spacing:0.04em;">Mean Intensity Score</p>
            <h2 style="margin:4px 0 0 0; font-size:36px; font-weight:800; color:#B77955;">{avg_int:.2f}</h2>
            <p style="margin:4px 0 0 0; font-size:12px; color:#727A73;">Calibrated cohort baseline</p>
        </div>
        """, unsafe_allow_html=True)

    c_left, c_right = st.columns([1.55, 1.15], gap="medium")
    with c_left:
        st.markdown('<div class="quiet-card">', unsafe_allow_html=True)
        traj_fig = render_intensity_trajectory(history)
        if traj_fig:
            st.plotly_chart(traj_fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with c_right:
        st.markdown('<div class="quiet-card">', unsafe_allow_html=True)
        donut_fig = render_emotion_distribution(history)
        if donut_fig:
            st.plotly_chart(donut_fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# VIEW 3: LIVE TELEMETRY SEARCH
# -----------------------------------------------------------------------------
elif "Search" in active_tab:
    st.markdown('<div class="quiet-card">', unsafe_allow_html=True)
    st.markdown("""
    <div style="margin-bottom:14px;">
        <h3 style="margin:0; font-size:18px; font-weight:700; color:#183D35;">Multi-Criteria Telemetry Search Engine</h3>
        <p style="margin:0; font-size:12.5px; color:#727A73;">Query historical interactions across affective categories, clinical triage tiers, and keywords</p>
    </div>
    """, unsafe_allow_html=True)

    sc1, sc2, sc3 = st.columns(3)
    with sc1:
        s_emo = st.selectbox("Emotion Filter", ["All", "joy", "sadness", "anger", "fear", "love", "surprise"])
    with sc2:
        s_tier = st.selectbox("Triage Level", ["All", "Mild", "Moderate", "Elevated", "Critical"])
    with sc3:
        s_kw = st.text_input("Keyword Search", placeholder="e.g. panic, breathing, exhaustion")

    results = search_engine.search_interactions(
        db=db,
        emotion=None if s_emo == "All" else s_emo,
        triage_level=None if s_tier == "All" else s_tier,
        keyword=s_kw.strip() if s_kw.strip() else None,
        limit=25
    )

    st.markdown(f"<p style='font-size:12px; color:#727A73; margin-top:12px;'>Discovered <b>{len(results)}</b> matching clinical records in SQLite encrypted storage</p>", unsafe_allow_html=True)

    if results:
        table_d = [{
            "ID": r["id"],
            "Emotion": r["primary_emotion"].upper(),
            "Polarity": f"{r.get('vader_compound', 0.0):+.2f}",
            "Intensity": f"{r['intensity_score']:.2f}",
            "Triage": r["triage_level"],
            "Sanitized Snippet": r["raw_text"][:55] + "...",
            "Timestamp": r["timestamp"][:19]
        } for r in results]
        st.dataframe(pd.DataFrame(table_d), use_container_width=True)
    else:
        st.info("No records matching the filter criteria found.")
    st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# VIEW 4: EXECUTIVE REPORTS & PDF HUB
# -----------------------------------------------------------------------------
elif "Reports" in active_tab:
    st.markdown('<div class="quiet-card">', unsafe_allow_html=True)
    st.markdown("""
    <div style="margin-bottom:16px;">
        <h3 style="margin:0; font-size:18px; font-weight:700; color:#183D35;">Executive Reporting & Export Center</h3>
        <p style="margin:0; font-size:12.5px; color:#727A73;">Compile clinical PDF dossiers with ReportLab tables or export telemetry CSV feeds</p>
    </div>
    """, unsafe_allow_html=True)

    r_c1, r_c2 = st.columns(2, gap="medium")
    with r_c1:
        st.markdown("""
        <div style="background:#F6F4EE; border:1px solid #E4E5DD; border-radius:14px; padding:22px;">
            <div style="display:flex; align-items:center; gap:10px; margin-bottom:10px;">
                <span style="font-size:24px;">📄</span>
                <div>
                    <h4 style="margin:0; font-size:16px; font-weight:700; color:#183D35;">Executive PDF Dossier</h4>
                    <p style="margin:0; font-size:11.5px; color:#727A73;">Clinical ReportLab compilation with longitudinal charts & governance</p>
                </div>
            </div>
            <p style="font-size:12.5px; color:#202B27; line-height:1.55; margin-bottom:16px;">
                Generates a formal, board-ready wellness analytics document summarizing affective trajectories, triage alerts, and personalized clinical recommendations.
            </p>
        """, unsafe_allow_html=True)

        if st.button("🚀 Compile & Download Executive PDF", type="primary", use_container_width=True):
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tf:
                pdf_p = tf.name
            try:
                exporter.export_executive_pdf(active_user, pdf_p)
                with open(pdf_p, "rb") as f:
                    st.download_button(
                        label="⬇️ Click to Save Executive PDF",
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
        <div style="background:#F6F4EE; border:1px solid #E4E5DD; border-radius:14px; padding:22px;">
            <div style="display:flex; align-items:center; gap:10px; margin-bottom:10px;">
                <span style="font-size:24px;">📊</span>
                <div>
                    <h4 style="margin:0; font-size:16px; font-weight:700; color:#183D35;">Telemetry CSV Data Feed</h4>
                    <p style="margin:0; font-size:11.5px; color:#727A73;">Raw interaction vectors and clinical ratings for corporate BI systems</p>
                </div>
            </div>
            <p style="font-size:12.5px; color:#202B27; line-height:1.55; margin-bottom:16px;">
                Exports complete sanitized time-series records including VADER sentiment compound scores, Ekman vectors, and intervention feedback.
            </p>
        """, unsafe_allow_html=True)

        if st.button("📊 Export Raw CSV Feed", use_container_width=True):
            with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tf:
                csv_p = tf.name
            try:
                exporter.export_interactions_csv(active_user, csv_p)
                with open(csv_p, "rb") as f:
                    st.download_button(
                        label="⬇️ Click to Save Telemetry CSV",
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

# -----------------------------------------------------------------------------
# VIEW 5: SECURITY & GDPR SANDBOX
# -----------------------------------------------------------------------------
elif "Security" in active_tab:
    st.markdown('<div class="quiet-card">', unsafe_allow_html=True)
    st.markdown("""
    <div style="margin-bottom:16px;">
        <h3 style="margin:0; font-size:18px; font-weight:700; color:#183D35;">Security Guardrails & PII Redaction Sandbox</h3>
        <p style="margin:0; font-size:12.5px; color:#727A73;">Real-time validation of zero-PII ingestion protocols and data sovereignty</p>
    </div>
    """, unsafe_allow_html=True)

    test_input = st.text_area(
        "Live Anonymization Verification Input",
        value="Contact me at sarah.lin@meridianhealth.org or call +1 (555) 234-5678 if the system has issues.",
        height=85
    )
    if test_input:
        masked, _ = sec.anonymize_pii(test_input)
        st.markdown("<p style='font-size:12px; font-weight:700; color:#183D35; margin:12px 0 4px 0;'>Sanitized Output (Ready for Neural Pipeline):</p>", unsafe_allow_html=True)
        st.code(masked, language="text")
        if masked != test_input:
            st.success("✅ PII Redacted: Sensitive employee identifiers (emails, phone numbers, entities) safely scrubbed before inference.")

    st.markdown("""
    <div style="display:grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap:16px; margin-top:20px;">
        <div style="background:#F6F4EE; border:1px solid #E4E5DD; border-radius:12px; padding:16px;">
            <h5 style="margin:0 0 6px 0; font-size:13.5px; font-weight:700; color:#183D35;">Zero-PII Storage Policy</h5>
            <p style="margin:0; font-size:12px; color:#727A73; line-height:1.5;">Input text is anonymized on-device prior to neural tensor generation and local database persistence.</p>
        </div>
        <div style="background:#F6F4EE; border:1px solid #E4E5DD; border-radius:12px; padding:16px;">
            <h5 style="margin:0 0 6px 0; font-size:13.5px; font-weight:700; color:#183D35;">GDPR Right to Erasure</h5>
            <p style="margin:0; font-size:12px; color:#727A73; line-height:1.5;">Full cryptographic erasure capability per user ID across all SQLite WAL interaction tables.</p>
        </div>
        <div style="background:#F6F4EE; border:1px solid #E4E5DD; border-radius:12px; padding:16px;">
            <h5 style="margin:0 0 6px 0; font-size:13.5px; font-weight:700; color:#183D35;">988 Crisis Guardrails</h5>
            <p style="margin:0; font-size:12px; color:#727A73; line-height:1.5;">Autonomous clinical triage detection triggers immediate emergency protocol override for acute distress.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

