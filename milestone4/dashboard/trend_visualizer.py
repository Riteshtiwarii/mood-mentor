from __future__ import annotations
from typing import List, Dict, Any, Tuple
from datetime import datetime

try:
    import plotly.graph_objects as go
    import plotly.express as px
    HAS_PLOTLY = True
except Exception:
    go = None
    px = None
    HAS_PLOTLY = False

# Enterprise Healthcare SaaS Color Palette (Modern Health / Stripe Caliber)
PALETTE = {
    "joy": "#10B981",       # Emerald Green
    "sadness": "#64748B",   # Slate Blue
    "anger": "#EF4444",     # Soft Crimson
    "fear": "#F59E0B",      # Warm Amber
    "love": "#EC4899",      # Rose Pink
    "surprise": "#8B5CF6",  # Indigo Violet
    "primary": "#0F766E",   # Deep Clinical Teal
    "primary_light": "#14B8A6",
    "accent": "#0284C7",    # Oceanic Blue
    "bg_card": "#FFFFFF",
    "text_dark": "#0F172A",
    "text_muted": "#64748B",
    "grid": "#F1F5F9",
    "border": "#E2E8F0"
}

FONT_FAMILY = "Plus Jakarta Sans, -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif"

def render_radar_state(emotion_probs: Dict[str, float]) -> Any:
    """Generates an executive 6-D emotion radar geometry with clinical precision"""
    if not HAS_PLOTLY or go is None:
        return None
    categories = [k.capitalize() for k in emotion_probs.keys()]
    values = list(emotion_probs.values())
    categories.append(categories[0])
    values.append(values[0])

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        fillcolor='rgba(15, 118, 110, 0.16)',
        line=dict(color=PALETTE["primary"], width=2.5),
        marker=dict(size=6, color=PALETTE["primary"], symbol="circle"),
        hovertemplate="<b>%{theta}</b>: %{r:.1%}<extra></extra>",
        name="Affect Profile"
    ))

    fig.update_layout(
        font=dict(family=FONT_FAMILY),
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1.0],
                gridcolor=PALETTE["border"],
                linecolor=PALETTE["border"],
                tickfont=dict(size=9, color=PALETTE["text_muted"])
            ),
            angularaxis=dict(
                gridcolor=PALETTE["border"],
                linecolor=PALETTE["border"],
                tickfont=dict(size=11, color=PALETTE["text_dark"], weight=600)
            ),
            bgcolor="rgba(248, 250, 252, 0.5)"
        ),
        showlegend=False,
        margin=dict(l=35, r=35, t=25, b=25),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=270
    )
    return fig

def render_intensity_trajectory(interactions: List[Dict[str, Any]]) -> Any:
    """Renders longitudinal emotional intensity timeline with calibrated severity zones"""
    if not HAS_PLOTLY or go is None:
        return None
    if not interactions:
        fig = go.Figure()
        fig.add_annotation(
            text="No telemetry recorded yet",
            showarrow=False,
            font=dict(family=FONT_FAMILY, size=13, color=PALETTE["text_muted"])
        )
        return fig

    # Sort chronologically
    sorted_data = sorted(interactions, key=lambda x: x["timestamp"])
    dates = [d["timestamp"][:16].replace("T", " ") for d in sorted_data]
    intensities = [d["intensity_score"] for d in sorted_data]
    emotions = [d["primary_emotion"].capitalize() for d in sorted_data]

    fig = go.Figure()

    # Severity reference bands
    fig.add_hrect(
        y0=0.0, y1=0.35,
        fillcolor="#F0FDF4", opacity=0.5, line_width=0,
        annotation_text="Mild Zone (Optimal)",
        annotation_position="top left",
        annotation_font=dict(family=FONT_FAMILY, size=10, color="#166534")
    )
    fig.add_hrect(
        y0=0.35, y1=0.65,
        fillcolor="#F0F9FF", opacity=0.5, line_width=0,
        annotation_text="Moderate Zone (Managed)",
        annotation_position="top left",
        annotation_font=dict(family=FONT_FAMILY, size=10, color="#0369A1")
    )
    fig.add_hrect(
        y0=0.65, y1=1.0,
        fillcolor="#FFF1F2", opacity=0.5, line_width=0,
        annotation_text="High / Critical Zone",
        annotation_position="top left",
        annotation_font=dict(family=FONT_FAMILY, size=10, color="#991B1B")
    )

    # Intensity trajectory spline curve
    fig.add_trace(go.Scatter(
        x=dates,
        y=intensities,
        mode="lines+markers",
        line=dict(color=PALETTE["primary"], width=3, shape="spline"),
        marker=dict(
            size=8,
            color=PALETTE["accent"],
            line=dict(color="#FFFFFF", width=2)
        ),
        text=emotions,
        hovertemplate="<b>%{text}</b><br>Intensity: %{y:.2f}<br>Time: %{x}<extra></extra>",
        name="Intensity Score"
    ))

    fig.update_layout(
        font=dict(family=FONT_FAMILY),
        title=dict(
            text="<b>Longitudinal Affective Intensity Trajectory</b>",
            font=dict(size=14, color=PALETTE["text_dark"])
        ),
        xaxis=dict(
            title=dict(text="Timeline", font=dict(size=11, color=PALETTE["text_muted"])),
            gridcolor=PALETTE["grid"],
            linecolor=PALETTE["border"],
            tickfont=dict(size=10, color=PALETTE["text_muted"])
        ),
        yaxis=dict(
            title=dict(text="Intensity (0.00 – 1.00)", font=dict(size=11, color=PALETTE["text_muted"])),
            range=[0, 1.05],
            gridcolor=PALETTE["grid"],
            linecolor=PALETTE["border"],
            tickfont=dict(size=10, color=PALETTE["text_muted"])
        ),
        margin=dict(l=45, r=30, t=45, b=45),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=320
    )
    return fig

def render_emotion_distribution(interactions: List[Dict[str, Any]]) -> Any:
    """Generates an executive donut chart of emotional distribution"""
    if not HAS_PLOTLY or go is None:
        return None
    if not interactions:
        fig = go.Figure()
        fig.add_annotation(
            text="No telemetry recorded yet",
            showarrow=False,
            font=dict(family=FONT_FAMILY, size=13, color=PALETTE["text_muted"])
        )
        return fig

    emotion_counts: Dict[str, int] = {}
    for item in interactions:
        emo = item["primary_emotion"].lower()
        emotion_counts[emo] = emotion_counts.get(emo, 0) + 1

    labels = [k.capitalize() for k in emotion_counts.keys()]
    values = list(emotion_counts.values())
    colors = [PALETTE.get(k.lower(), "#94A3B8") for k in emotion_counts.keys()]

    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.64,
        marker=dict(colors=colors, line=dict(color="#FFFFFF", width=2)),
        textinfo="percent+label",
        hoverinfo="label+value+percent",
        textfont=dict(family=FONT_FAMILY, size=11)
    )])

    fig.update_layout(
        font=dict(family=FONT_FAMILY),
        title=dict(
            text="<b>Dominant Affect Distribution</b>",
            font=dict(size=14, color=PALETTE["text_dark"])
        ),
        showlegend=False,
        margin=dict(l=25, r=25, t=45, b=25),
        paper_bgcolor='rgba(0,0,0,0)',
        height=270
    )
    return fig

def calculate_burnout_risk_index(interactions: List[Dict[str, Any]]) -> Tuple[int, str, str]:
    """
    Calculates dynamic Burnout Risk Index (0-100) based on:
    - Frequency of high negative emotions (Fear, Anger, Sadness)
    - Intensity momentum over the past 5 interactions
    """
    if not interactions:
        return 15, "Low", "#10B981"

    recent = interactions[:10]
    negative_count = sum(1 for i in recent if i["primary_emotion"] in ["fear", "anger", "sadness"])
    avg_intensity = sum(i["intensity_score"] for i in recent) / len(recent)
    critical_count = sum(1 for i in recent if i["triage_level"] in ["High", "Critical"])

    score = int((negative_count / len(recent) * 45) + (avg_intensity * 40) + (critical_count * 15))
    score = max(5, min(95, score))

    if score < 35:
        return score, "Low Risk (Healthy)", "#10B981"
    elif score < 65:
        return score, "Moderate Risk (Managed)", "#0284C7"
    elif score < 85:
        return score, "Elevated Burnout Risk", "#F59E0B"
    else:
        return score, "Critical Exhaustion Alert", "#EF4444"
