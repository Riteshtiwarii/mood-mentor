"""
MoodMentor AI — Enterprise Design System & Shared Motion Theme
Concept: "Quiet Luxury Meets Clinical Intelligence"
"""

import streamlit as st

QUIET_LUXURY_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

    *, *::before, *::after {
        box-sizing: border-box;
    }

    html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
        color: #202B27 !important;
        background-color: #F6F4EE !important;
    }

    /* Streamlit Chrome adjustments */
    #MainMenu, footer { visibility: hidden !important; }
    header[data-testid="stHeader"] {
        background-color: transparent !important;
        z-index: 100 !important;
    }
    header[data-testid="stHeader"] [data-testid="stToolbar"] {
        display: none !important;
    }
    [data-testid="collapsedControl"] {
        display: flex !important;
        visibility: visible !important;
        opacity: 1 !important;
        z-index: 999999 !important;
        background-color: #183D35 !important;
        color: #F6F4EE !important;
        border-radius: 8px !important;
        padding: 5px 8px !important;
        margin-left: 12px !important;
        margin-top: 8px !important;
        box-shadow: 0 2px 10px rgba(24, 61, 53, 0.25) !important;
        cursor: pointer !important;
    }
    [data-testid="collapsedControl"] svg {
        fill: #F6F4EE !important;
        stroke: #F6F4EE !important;
    }
    /* Multipage sidebar nav visible */
    [data-testid="stSidebarNav"] {
        display: block !important;
    }
    /* Eliminate dulling/dimming fade during execution reruns */
    .stApp[data-test-script-state="running"] > .main,
    .stApp[data-test-script-state="running"] [data-testid="stAppViewContainer"],
    .stApp[data-test-script-state="running"] .block-container,
    [data-testid="stAppViewContainer"] {
        opacity: 1 !important;
        filter: none !important;
        transition: none !important;
    }
    .block-container {
        padding-top: 1.2rem !important;
        padding-bottom: 3.5rem !important;
        max-width: 1380px !important;
    }

    /* ---------------- KEYFRAME ANIMATIONS ---------------- */
    @keyframes breathe-calm {
        0% {
            transform: scale(0.92) rotate(0deg);
            opacity: 0.75;
        }
        50% {
            transform: scale(1.08) rotate(180deg);
            opacity: 0.95;
        }
        100% {
            transform: scale(0.92) rotate(360deg);
            opacity: 0.75;
        }
    }

    @keyframes breathe-core {
        0% {
            transform: scale(0.88);
            box-shadow: 0 0 35px rgba(183, 121, 85, 0.25);
        }
        50% {
            transform: scale(1.06);
            box-shadow: 0 0 65px rgba(24, 61, 53, 0.35);
        }
        100% {
            transform: scale(0.88);
            box-shadow: 0 0 35px rgba(183, 121, 85, 0.25);
        }
    }

    @keyframes aura-float {
        0%, 100% {
            transform: translateY(0px) rotate(0deg);
        }
        50% {
            transform: translateY(-8px) rotate(3deg);
        }
    }

    @keyframes pulse-dot-calm {
        0% { box-shadow: 0 0 0 0 rgba(45, 106, 79, 0.6); }
        70% { box-shadow: 0 0 0 8px rgba(45, 106, 79, 0); }
        100% { box-shadow: 0 0 0 0 rgba(45, 106, 79, 0); }
    }

    @keyframes pulse-critical-calm {
        0% { box-shadow: 0 0 0 0 rgba(185, 28, 28, 0.6); }
        70% { box-shadow: 0 0 0 10px rgba(185, 28, 28, 0); }
        100% { box-shadow: 0 0 0 0 rgba(185, 28, 28, 0); }
    }

    @keyframes cardFadeIn {
        from {
            opacity: 0;
            transform: translateY(10px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    /* ---------------- TOP EXECUTIVE GLASS NAVBAR ---------------- */
    .top-navbar {
        background: rgba(255, 255, 255, 0.88);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid #E4E5DD;
        border-radius: 18px;
        padding: 14px 26px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 22px;
        box-shadow: 0 4px 20px -2px rgba(32, 43, 39, 0.03);
        animation: cardFadeIn 0.35s ease-out;
    }

    .nav-brand {
        display: flex;
        align-items: center;
        gap: 14px;
    }

    .brand-mark {
        background: linear-gradient(135deg, #183D35 0%, #102C27 100%);
        color: #F6F4EE;
        width: 42px;
        height: 42px;
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 20px;
        border: 1px solid rgba(168, 185, 165, 0.3);
        box-shadow: 0 4px 14px rgba(24, 61, 53, 0.18);
    }

    .brand-title {
        font-size: 19px;
        font-weight: 700;
        letter-spacing: -0.02em;
        color: #183D35;
        margin: 0;
        line-height: 1.2;
    }

    .brand-subtitle {
        font-size: 11px;
        color: #727A73;
        margin: 0;
        font-weight: 500;
        letter-spacing: 0.04em;
        text-transform: uppercase;
    }

    .nav-badges {
        display: flex;
        align-items: center;
        gap: 12px;
    }

    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 6px 14px;
        border-radius: 9999px;
        font-size: 12px;
        font-weight: 600;
        background: #E8EDE5;
        color: #183D35;
        border: 1px solid #A8B9A5;
    }

    .pulse-dot {
        width: 8px;
        height: 8px;
        background: #2D6A4F;
        border-radius: 50%;
        animation: pulse-dot-calm 2.2s infinite;
    }

    .compliance-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 6px 12px;
        border-radius: 8px;
        font-size: 11.5px;
        font-weight: 500;
        background: #FFFFFF;
        color: #727A73;
        border: 1px solid #E4E5DD;
        font-family: 'JetBrains Mono', monospace;
    }

    /* ---------------- CARDS & ELEVATIONS ---------------- */
    .quiet-card, .enterprise-card {
        background: #FFFFFF;
        border: 1px solid #E4E5DD;
        border-radius: 18px;
        padding: 24px;
        box-shadow: 0 1px 3px rgba(32, 43, 39, 0.03);
        margin-bottom: 20px;
        transition: transform 0.25s cubic-bezier(0.16, 1, 0.3, 1), box-shadow 0.25s cubic-bezier(0.16, 1, 0.3, 1);
        animation: cardFadeIn 0.4s ease-out;
    }

    .quiet-card:hover, .enterprise-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 24px -4px rgba(24, 61, 53, 0.06);
        border-color: #A8B9A5;
    }

    .pill-tag {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 11px;
        font-weight: 600;
        margin-right: 6px;
        background: #E8EDE5;
        color: #183D35;
        border: 1px solid #A8B9A5;
    }

    .metric-card {
        background: #FFFFFF;
        border: 1px solid #E4E5DD;
        border-radius: 16px;
        padding: 18px 20px;
        box-shadow: 0 1px 3px rgba(32, 43, 39, 0.02);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }

    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 18px -4px rgba(24, 61, 53, 0.05);
        border-color: #A8B9A5;
    }

    .metric-label {
        font-size: 11.5px;
        font-weight: 600;
        color: #727A73;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 6px;
    }

    .metric-value {
        font-size: 26px;
        font-weight: 800;
        color: #183D35;
        line-height: 1.15;
        letter-spacing: -0.02em;
    }

    .metric-subtext {
        font-size: 11.5px;
        color: #727A73;
        margin-top: 5px;
    }

    /* ---------------- CLINICAL HUD & SEVERITY BADGES ---------------- */
    .triage-hud {
        border-radius: 14px;
        padding: 16px 20px;
        margin-bottom: 20px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        animation: cardFadeIn 0.3s ease-out;
    }

    .triage-hud-mild {
        background: #E8EDE5;
        border: 1px solid #A8B9A5;
        color: #183D35;
    }

    .triage-hud-moderate {
        background: #F0F4F2;
        border: 1px solid #CBD7C9;
        color: #2D6A4F;
    }

    .triage-hud-elevated {
        background: #FBF2EC;
        border: 1px solid #E8CDBF;
        color: #B77955;
    }

    .triage-hud-critical {
        background: #FFF1F2;
        border: 1.5px solid #FECACA;
        color: #B91C1C;
        animation: pulse-critical-calm 2s infinite;
    }

    /* ---------------- ACTION / INTERVENTION CARDS ---------------- */
    .action-card {
        background: #FFFFFF;
        border: 1px solid #E4E5DD;
        border-radius: 16px;
        padding: 20px 22px;
        margin-bottom: 16px;
        transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1);
        box-shadow: 0 1px 3px rgba(32, 43, 39, 0.02);
    }

    .action-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 24px -4px rgba(24, 61, 53, 0.07);
        border-color: #183D35;
    }

    .badge-rank {
        background: #183D35;
        color: #F6F4EE;
        font-weight: 700;
        font-size: 10.5px;
        padding: 3px 9px;
        border-radius: 6px;
        letter-spacing: 0.04em;
    }

    .badge-modality {
        background: #E8EDE5;
        color: #183D35;
        font-weight: 600;
        font-size: 11px;
        padding: 3px 9px;
        border-radius: 6px;
    }

    .badge-match {
        background: #FBF2EC;
        color: #B77955;
        font-weight: 700;
        font-size: 11px;
        padding: 3px 9px;
        border-radius: 6px;
        border: 1px solid #E8CDBF;
    }

    /* ---------------- 3D ORGANIC BREATHING SIGNATURE ---------------- */
    .breathing-stage {
        position: relative;
        width: 250px;
        height: 250px;
        margin: 16px auto;
        display: flex;
        align-items: center;
        justify-content: center;
        cursor: default;
        transition: transform 0.35s ease;
    }

    .breathing-stage:hover {
        transform: scale(1.02);
    }

    .breathing-ring-outer {
        position: absolute;
        width: 230px;
        height: 230px;
        border-radius: 42% 58% 70% 30% / 45% 45% 55% 55%;
        background: radial-gradient(circle, rgba(168, 185, 165, 0.30) 0%, rgba(24, 61, 53, 0.04) 70%);
        border: 1px dashed rgba(24, 61, 53, 0.22);
        animation: breathe-calm 16s infinite linear;
    }

    .breathing-ring-mid {
        position: absolute;
        width: 170px;
        height: 170px;
        border-radius: 60% 40% 30% 70% / 60% 30% 70% 40%;
        background: radial-gradient(circle, rgba(183, 121, 85, 0.18) 0%, rgba(45, 106, 79, 0.08) 80%);
        border: 1px solid rgba(183, 121, 85, 0.3);
        animation: breathe-calm 10s infinite reverse ease-in-out;
    }

    .breathing-ring-core {
        position: absolute;
        width: 105px;
        height: 105px;
        border-radius: 50%;
        background: linear-gradient(135deg, #183D35 0%, #102C27 100%);
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        color: #F6F4EE;
        box-shadow: 0 8px 30px rgba(16, 44, 39, 0.35);
        animation: breathe-core 6s infinite ease-in-out;
        transition: transform 0.3s ease;
    }

    .breathing-stage:hover .breathing-ring-core {
        transform: scale(1.05);
        box-shadow: 0 10px 38px rgba(16, 44, 39, 0.45);
    }

    @media (prefers-reduced-motion: reduce) {
        .breathing-ring-outer,
        .breathing-ring-mid,
        .breathing-ring-core,
        .pulse-dot {
            animation: none !important;
        }
    }

    /* ---------------- STREAMLIT COMPONENT OVERRIDES ---------------- */
    /* Radio navigation styling */
    div[data-testid="stRadio"] div[role="radiogroup"] {
        background: #FFFFFF !important;
        padding: 8px 14px !important;
        border-radius: 14px !important;
        border: 1px solid #E4E5DD !important;
        box-shadow: 0 1px 3px rgba(32, 43, 39, 0.03) !important;
        gap: 14px !important;
    }

    div[data-testid="stRadio"] label, div[data-testid="stRadio"] p, div[data-testid="stRadio"] span {
        color: #202B27 !important;
        font-weight: 600 !important;
        font-size: 13.5px !important;
    }

    .stTextArea textarea {
        border-radius: 14px !important;
        border: 1.5px solid #E4E5DD !important;
        background-color: #FFFFFF !important;
        color: #202B27 !important;
        padding: 16px !important;
        font-size: 14.5px !important;
        transition: all 0.2s ease !important;
    }
    .stTextArea textarea:focus {
        border-color: #183D35 !important;
        box-shadow: 0 0 0 3px rgba(24, 61, 53, 0.12) !important;
    }

    .stTextInput input {
        border-radius: 12px !important;
        border: 1.5px solid #E4E5DD !important;
        background-color: #FFFFFF !important;
        color: #202B27 !important;
        font-size: 14px !important;
    }
    .stTextInput input:focus {
        border-color: #183D35 !important;
        box-shadow: 0 0 0 3px rgba(24, 61, 53, 0.12) !important;
    }

    .stButton > button {
        border-radius: 12px !important;
        font-weight: 600 !important;
        font-size: 14px !important;
        padding: 10px 22px !important;
        transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1) !important;
    }
    .stButton > button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 14px rgba(24, 61, 53, 0.14) !important;
    }

    /* Primary button style */
    .stButton > button[kind="primary"] {
        background-color: #183D35 !important;
        border-color: #183D35 !important;
        color: #F6F4EE !important;
    }
    .stButton > button[kind="primary"]:hover {
        background-color: #102C27 !important;
        border-color: #102C27 !important;
    }

    /* Permanent Left Sidebar Locked Open */
    section[data-testid="stSidebar"] {
        display: block !important;
        visibility: visible !important;
        transform: none !important;
        margin-left: 0 !important;
        min-width: 300px !important;
        max-width: 320px !important;
        background-color: #FFFFFF !important;
        border-right: 1px solid #E4E5DD !important;
    }
    section[data-testid="stSidebar"] > div {
        transform: none !important;
        visibility: visible !important;
    }
    section[data-testid="stSidebar"][aria-expanded="false"] {
        display: block !important;
        visibility: visible !important;
        transform: none !important;
        margin-left: 0 !important;
    }

    /* Expander styling */
    details[data-testid="stExpander"] {
        background-color: #FFFFFF !important;
        border: 1px solid #E4E5DD !important;
        border-radius: 12px !important;
    }
</style>
"""

def apply_enterprise_theme(page_title: str = "MoodMentor AI — Enterprise", page_icon: str = "🌿"):
    """Applies page configuration and Quiet Luxury CSS tokens."""
    st.set_page_config(
        page_title=page_title,
        page_icon=page_icon,
        layout="wide",
        initial_sidebar_state="expanded"
    )
    st.markdown(QUIET_LUXURY_CSS, unsafe_allow_html=True)
    # Auto-expand sidebar if browser persisted it in collapsed state
    st.markdown("""
    <img src="data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7"
         onload="(function(){
             try {
                 var btn = window.parent.document.querySelector('[data-testid=\\'collapsedControl\\'] button') || window.parent.document.querySelector('[data-testid=\\'collapsedControl\\']');
                 if(btn) { btn.click(); }
             } catch(e) {}
         })()"
         style="display:none;" />
    """, unsafe_allow_html=True)

def render_top_navbar(current_view: str = "Clinical Copilot"):
    """Renders the executive top glassmorphic navbar with Quiet Luxury styling."""
    st.markdown(f"""
    <div class="top-navbar">
        <div class="nav-brand">
            <div class="brand-mark">🌿</div>
            <div>
                <p class="brand-title">MoodMentor AI <span style="font-size:11px; font-weight:700; color:#183D35; background:#E8EDE5; padding:2px 8px; border-radius:6px; margin-left:6px; border:1px solid #A8B9A5;">Enterprise v4.2</span></p>
                <p class="brand-subtitle">Quiet Luxury Clinical Intelligence • <b>{current_view}</b></p>
            </div>
        </div>
        <div class="nav-badges">
            <span class="compliance-badge">🔒 HIPAA & GDPR Tier-4</span>
            <span class="status-badge">
                <span class="pulse-dot"></span>
                <span>Neural Engine Online</span>
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)

def render_sidebar_context(default_index: int = 0) -> str:
    """Renders enterprise team, user profile context and quick platform telemetry in the sidebar."""
    with st.sidebar:
        st.markdown("""
        <div style="padding: 6px 0 14px 0; border-bottom: 1px solid #E4E5DD; margin-bottom: 14px;">
            <div style="display:flex; align-items:center; gap:10px;">
                <div style="width:34px; height:34px; background:#183D35; border-radius:8px; display:flex; align-items:center; justify-content:center; color:#F6F4EE; font-size:16px;">🌿</div>
                <div>
                    <p style="margin:0; font-size:13.5px; font-weight:700; color:#183D35;">Meridian Health</p>
                    <p style="margin:0; font-size:11px; color:#727A73; font-family:'JetBrains Mono', monospace;">ws-873519 • Enterprise</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        user_id = st.selectbox(
            "👤 Active Employee Profile",
            [
                "emp_01 (Sarah Lin - Eng)",
                "emp_02 (Marcus Vance - Sales)",
                "emp_03 (Elena Rostova - Design)"
            ],
            index=default_index
        )

        st.markdown("<p style='font-size:11px; font-weight:700; color:#727A73; text-transform:uppercase; letter-spacing:0.05em; margin:16px 0 8px 0;'>Navigation Menu</p>", unsafe_allow_html=True)
        st.page_link("app.py", label="Clinical Copilot (Text Emotion)", icon="🧠")
        st.page_link("pages/1_📈_Analytics_&_Trends.py", label="Longitudinal Analytics & Burnout", icon="📈")
        st.page_link("pages/2_🔍_Telemetry_Search.py", label="Live Telemetry Search Engine", icon="🔍")
        st.page_link("pages/3_📑_Executive_Reports.py", label="Executive Reports & PDF Hub", icon="📑")
        st.page_link("pages/4_🔒_Security_&_Compliance.py", label="Security & GDPR Sandbox", icon="🔒")

        return user_id.split()[0]

def render_breathing_visual() -> str:
    """Renders the organic 3D breathing visual signature in HTML/CSS."""
    return """<div class="breathing-stage">
<div class="breathing-ring-outer"></div>
<div class="breathing-ring-mid"></div>
<div class="breathing-ring-core">
<span style="font-size:24px; line-height:1;">🌿</span>
<span style="font-size:9px; font-weight:700; letter-spacing:0.08em; text-transform:uppercase; margin-top:4px;">Equilibrium</span>
</div>
</div>"""

def render_login_hero() -> str:
    """Renders the refined editorial hero section on the left side of the login experience."""
    breathing = render_breathing_visual()
    return f"""<div style="background: linear-gradient(145deg, #183D35 0%, #102C27 100%); border-radius: 22px; padding: 34px 30px; color: #F6F4EE; box-shadow: 0 12px 36px rgba(16, 44, 39, 0.2); border: 1px solid rgba(168, 185, 165, 0.25);">
<div style="display:flex; align-items:center; gap:12px; margin-bottom: 20px;">
<div style="width:40px; height:40px; border-radius:10px; background:rgba(255,255,255,0.12); display:flex; align-items:center; justify-content:center; font-size:20px; border:1px solid rgba(255,255,255,0.2);">🌿</div>
<div>
<h2 style="margin:0; font-size:18px; font-weight:700; color:#FFFFFF; letter-spacing:-0.02em;">MoodMentor AI</h2>
<p style="margin:0; font-size:11px; font-weight:600; color:#A8B9A5; text-transform:uppercase; letter-spacing:0.05em;">Enterprise Clinical Intelligence</p>
</div>
</div>
<h1 style="margin:0 0 12px 0; font-size:27px; font-weight:800; line-height:1.25; color:#FFFFFF; letter-spacing:-0.02em;">
Understand the emotion. <br><span style="color:#A8B9A5;">Support the person.</span>
</h1>
<p style="margin:0 0 16px 0; font-size:13.5px; color:#E4E5DD; line-height:1.6;">
Enterprise AI emotional intelligence platform providing real-time affective computing, longitudinal burnout telemetry, and evidence-based clinical interventions.
</p>
{breathing}
<div style="display:flex; flex-direction:column; gap:10px; margin-top:18px;">
<div style="display:flex; align-items:flex-start; gap:10px; background:rgba(255,255,255,0.06); padding:10px 14px; border-radius:10px; border:1px solid rgba(255,255,255,0.08);">
<span style="font-size:16px;">🧠</span>
<div>
<p style="margin:0; font-size:12.5px; font-weight:700; color:#FFFFFF;">Clinical Precision</p>
<p style="margin:2px 0 0 0; font-size:11.5px; color:#E4E5DD; line-height:1.4;">Ekman 6-D affective mapping calibrated with DistilBERT and VADER sentiment vectors.</p>
</div>
</div>
<div style="display:flex; align-items:flex-start; gap:10px; background:rgba(255,255,255,0.06); padding:10px 14px; border-radius:10px; border:1px solid rgba(255,255,255,0.08);">
<span style="font-size:16px;">🛡️</span>
<div>
<p style="margin:0; font-size:12.5px; font-weight:700; color:#FFFFFF;">Zero-PII Privacy</p>
<p style="margin:2px 0 0 0; font-size:11.5px; color:#E4E5DD; line-height:1.4;">On-device entity scrubbing ensuring complete employee anonymity.</p>
</div>
</div>
<div style="display:flex; align-items:flex-start; gap:10px; background:rgba(255,255,255,0.06); padding:10px 14px; border-radius:10px; border:1px solid rgba(255,255,255,0.08);">
<span style="font-size:16px;">📉</span>
<div>
<p style="margin:0; font-size:12.5px; font-weight:700; color:#FFFFFF;">Proactive Resilience</p>
<p style="margin:2px 0 0 0; font-size:11.5px; color:#E4E5DD; line-height:1.4;">Early warning burnout indicators reducing unmanaged stress escalations.</p>
</div>
</div>
</div>
</div>"""


