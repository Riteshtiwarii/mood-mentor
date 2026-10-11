"""
MoodMentor — Enterprise Design System & Shared Motion Theme
Caliber: Linear / Stripe / Modern Health
"""

import streamlit as st

ENTERPRISE_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

    * { box-sizing: border-box; }

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
        color: #0F172A;
        background-color: #F8FAFC !important;
    }

    /* Streamlit Chrome adjustments */
    #MainMenu, footer, header { visibility: hidden !important; }
    .block-container {
        padding-top: 1.2rem !important;
        padding-bottom: 3.5rem !important;
        max-width: 1360px !important;
    }

    /* ---------------- MOTION & KEYFRAME ANIMATIONS ---------------- */
    @keyframes pulse-glow {
        0% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.6); }
        70% { box-shadow: 0 0 0 10px rgba(16, 185, 129, 0); }
        100% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
    }

    @keyframes pulse-warning {
        0% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.6); }
        70% { box-shadow: 0 0 0 10px rgba(239, 68, 68, 0); }
        100% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0); }
    }

    @keyframes cardFadeIn {
        from {
            opacity: 0;
            transform: translateY(8px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    @keyframes shimmer {
        0% { background-position: -468px 0; }
        100% { background-position: 468px 0; }
    }

    /* ---------------- TOP GLASS NAVBAR ---------------- */
    .top-navbar {
        background: rgba(255, 255, 255, 0.92);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid #E2E8F0;
        border-radius: 16px;
        padding: 14px 24px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 22px;
        box-shadow: 0 4px 20px -2px rgba(15, 23, 42, 0.04);
        animation: cardFadeIn 0.35s ease-out;
    }

    .nav-brand {
        display: flex;
        align-items: center;
        gap: 14px;
    }

    .brand-mark {
        background: linear-gradient(135deg, #0F766E 0%, #0284C7 100%);
        color: white;
        width: 42px;
        height: 42px;
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 22px;
        font-weight: 800;
        box-shadow: 0 4px 12px rgba(15, 118, 110, 0.25);
    }

    .brand-title {
        font-size: 19px;
        font-weight: 800;
        letter-spacing: -0.02em;
        color: #0F172A;
        margin: 0;
        line-height: 1.2;
    }

    .brand-subtitle {
        font-size: 11.5px;
        color: #64748B;
        margin: 0;
        font-weight: 500;
        letter-spacing: 0.02em;
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
        background: #F0FDF4;
        color: #166534;
        border: 1px solid #BBF7D0;
    }

    .pulse-dot {
        width: 8px;
        height: 8px;
        background: #10B981;
        border-radius: 50%;
        animation: pulse-glow 2s infinite;
    }

    .compliance-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 6px 12px;
        border-radius: 8px;
        font-size: 11.5px;
        font-weight: 600;
        background: #F8FAFC;
        color: #475569;
        border: 1px solid #E2E8F0;
        font-family: 'JetBrains Mono', monospace;
    }

    /* ---------------- CARDS & ELEVATIONS WITH HOVER MOTION ---------------- */
    .enterprise-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 16px;
        padding: 22px 24px;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04);
        margin-bottom: 20px;
        transition: transform 0.25s cubic-bezier(0.16, 1, 0.3, 1), box-shadow 0.25s cubic-bezier(0.16, 1, 0.3, 1);
        animation: cardFadeIn 0.4s ease-out;
    }

    .enterprise-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 28px -6px rgba(15, 23, 42, 0.08);
        border-color: #CBD5E1;
    }

    .metric-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 14px;
        padding: 18px 20px;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.03);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }

    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 16px -4px rgba(15, 23, 42, 0.06);
    }

    .metric-label {
        font-size: 12px;
        font-weight: 600;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.03em;
        margin-bottom: 6px;
    }

    .metric-value {
        font-size: 26px;
        font-weight: 800;
        color: #0F172A;
        line-height: 1.1;
        letter-spacing: -0.02em;
    }

    .metric-subtext {
        font-size: 11.5px;
        color: #94A3B8;
        margin-top: 5px;
    }

    /* ---------------- CLINICAL HUD & SEVERITY BADGES ---------------- */
    .hud-banner {
        border-radius: 14px;
        padding: 16px 20px;
        margin-bottom: 20px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        animation: cardFadeIn 0.3s ease-out;
    }

    .hud-mild {
        background: #F0FDF4;
        border: 1px solid #BBF7D0;
        color: #166534;
    }

    .hud-moderate {
        background: #F0F9FF;
        border: 1px solid #BAE6FD;
        color: #0369A1;
    }

    .hud-elevated {
        background: #FEF3C7;
        border: 1px solid #FDE68A;
        color: #92400E;
    }

    .hud-critical {
        background: #FEF2F2;
        border: 1px solid #FECACA;
        color: #991B1B;
        animation: pulse-warning 2s infinite;
    }

    /* ---------------- STREAMLIT BUTTONS & INPUT REFINEMENTS ---------------- */
    .stButton > button {
        border-radius: 10px !important;
        font-weight: 600 !important;
        font-size: 14px !important;
        padding: 10px 22px !important;
        transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1) !important;
    }

    .stButton > button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 12px rgba(15, 23, 42, 0.12) !important;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #FFFFFF !important;
        border-right: 1px solid #E2E8F0 !important;
    }

    /* Pill Tag */
    .pill-tag {
        display: inline-block;
        padding: 3px 9px;
        border-radius: 6px;
        font-size: 11px;
        font-weight: 600;
        margin-right: 6px;
        background: #F1F5F9;
        color: #475569;
    }
</style>
"""


def apply_enterprise_theme(page_title: str = "MoodMentor Enterprise", page_icon: str = "🧠"):
    """Applies page configuration and enterprise CSS rules with motion keyframes."""
    st.set_page_config(
        page_title=page_title,
        page_icon=page_icon,
        layout="wide",
        initial_sidebar_state="expanded"
    )
    st.markdown(ENTERPRISE_CSS, unsafe_allow_html=True)


def render_top_navbar(current_page: str = "Overview"):
    """Renders the top enterprise glassmorphic navbar."""
    st.markdown(f"""
    <div class="top-navbar">
        <div class="nav-brand">
            <div class="brand-mark">🌿</div>
            <div>
                <p class="brand-title">MoodMentor AI <span style="font-size:12px; font-weight:600; color:#0284C7; background:#E0F2FE; padding:2px 8px; border-radius:6px; margin-left:6px;">Enterprise v4.0</span></p>
                <p class="brand-subtitle">Clinical Intelligence & Emotional Wellness SaaS • <b>{current_page}</b></p>
            </div>
        </div>
        <div class="nav-badges">
            <span class="compliance-badge">🔒 HIPAA / GDPR Tier-4</span>
            <span class="status-badge">
                <span class="pulse-dot"></span>
                <span>System Operational</span>
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_sidebar_context():
    """Renders enterprise team, user profile context and quick platform telemetry in the sidebar."""
    with st.sidebar:
        st.markdown("""
        <div style="padding: 10px 0 16px 0; border-bottom: 1px solid #E2E8F0; margin-bottom: 14px;">
            <div style="display:flex; align-items:center; gap:10px;">
                <div style="width:34px; height:34px; background:#0F766E; border-radius:8px; display:flex; align-items:center; justify-content:center; color:white; font-size:16px;">🏢</div>
                <div>
                    <p style="margin:0; font-size:13px; font-weight:700; color:#0F172A;">WellSync Enterprise</p>
                    <p style="margin:0; font-size:11px; color:#64748B;">Workspace ID: <code>ws-873519</code></p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        user_id = st.selectbox(
            "👤 Active Employee Profile",
            ["emp_01 (Sarah Lin - Eng)", "emp_02 (Marcus Vance - Sales)", "emp_03 (Elena Rostova - Design)"],
            index=0
        )
        
        st.markdown("""
        <div style="background:#F8FAFC; border:1px solid #E2E8F0; border-radius:10px; padding:12px; margin-top:16px;">
            <p style="margin:0 0 6px 0; font-size:11px; font-weight:700; color:#64748B; text-transform:uppercase;">Platform Security</p>
            <div style="font-size:11.5px; color:#334155; line-height:1.6;">
                • PII Masking: <b>Active (spaCy NER)</b><br>
                • Model: <b>DistilBERT Multi-Label</b><br>
                • Storage: <b>Encrypted SQLite WAL</b><br>
                • 988 Crisis Guardrail: <b>Enforced</b>
            </div>
        </div>
        """, unsafe_allow_html=True)

        return user_id.split()[0]
