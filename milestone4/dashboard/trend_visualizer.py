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

# Enterprise Healthcare SaaS Color Palette
PALETTE = {
    "joy": "#10B981",       # Emerald Green
    "sadness": "#64748B",   # Slate Blue
    "anger": "#EF4444",     # Soft Crimson
    "fear": "#F59E0B",      # Warm Amber
    "love": "#EC4899",      # Rose Pink
    "surprise": "#8B5CF6",  # Indigo Violet
    "primary": "#0F766E",   # Deep Teal
    "accent": "#0284C7",    # Ocean Blue
    "bg_card": "#FFFFFF",
    "text_dark": "#0F172A",
    "grid": "#F1F5F9"
}

def render_radar_state(emotion_probs: Dict[str, float]) -> Any:
    """Generates a clean 6-D emotion radar chart for live user text"""
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
        fillcolor='rgba(15, 118, 110, 0.18)',
        line=dict(color=PALETTE["primary"], width=2.5),
        name="Emotion Profile"
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1.0], gridcolor=PALETTE["grid"]),
            angularaxis=dict(gridcolor=PALETTE["grid"], tickfont=dict(size=11, color=PALETTE["text_dark"]))
        ),
        showlegend=False,
        margin=dict(l=30, r=30, t=20, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=260
    )
    return fig

def render_intensity_trajectory(interactions: List[Dict[str, Any]]) -> Any:
    """Renders longitudinal emotional intensity timeline with severity bands"""
    if not HAS_PLOTLY or go is None:
        return None
    if not interactions:
        fig = go.Figure()
        fig.add_annotation(text="No interaction telemetry recorded yet", showarrow=False)
        return fig

    # Sort chronologically
    sorted_data = sorted(interactions, key=lambda x: x["timestamp"])
    dates = [d["timestamp"][:16].replace("T", " ") for d in sorted_data]
    intensities = [d["intensity_score"] for d in sorted_data]
    emotions = [d["primary_emotion"].capitalize() for d in sorted_data]

    fig = go.Figure()

    # Severity reference bands
    fig.add_hrect(y0=0.0, y1=0.35, fillcolor="#DEF7EC", opacity=0.35, line_width=0, annotation_text="Mild Zone", annotation_position="top left")
    fig.add_hrect(y0=0.35, y1=0.65, fillcolor="#E1EFFE", opacity=0.35, line_width=0, annotation_text="Moderate Zone", annotation_position="top left")
    fig.add_hrect(y0=0.65, y1=1.0, fillcolor="#FEE2E2", opacity=0.35, line_width=0, annotation_text="High / Critical", annotation_position="top left")

    # Intensity trajectory line
    fig.add_trace(go.Scatter(
        x=dates,
        y=intensities,
        mode="lines+markers",
        line=dict(color=PALETTE["accent"], width=3, shape="spline"),
        marker=dict(size=8, color=PALETTE["primary"]),
        text=emotions,
        hovertemplate="<b>Date:</b> %{x}<br><b>Intensity:</b> %{y:.2f}<br><b>Emotion:</b> %{text}<extra></extra>",
        name="Intensity Score"
    ))

    fig.update_layout(
        title=dict(text="<b>Longitudinal Emotional Intensity Trajectory</b>", font=dict(size=14, color=PALETTE["text_dark"])),
        xaxis=dict(title="Timestamp", gridcolor=PALETTE["grid"]),
        yaxis=dict(title="Intensity (0.0 - 1.0)", range=[0, 1.05], gridcolor=PALETTE["grid"]),
        margin=dict(l=40, r=30, t=40, b=40),
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
        fig.add_annotation(text="No interaction telemetry available", showarrow=False)
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
        hole=0.6,
        marker=dict(colors=colors),
        textinfo="percent+label",
        hoverinfo="label+value+percent"
    )])

    fig.update_layout(
        title=dict(text="<b>Dominant Emotion Distribution</b>", font=dict(size=14, color=PALETTE["text_dark"])),
        showlegend=False,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        height=260
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
        return score, "Moderate Risk (Needs Attention)", "#0284C7"
    elif score < 85:
        return score, "Elevated Burnout Risk", "#F59E0B"
    else:
        return score, "Critical Exhaustion Alert", "#EF4444"
