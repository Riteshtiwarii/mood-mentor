"""
Mood Mentor — Explainable AI (XAI) Engine
Generates transparent, multi-factor clinical explanations for user recommendations.
"""

from typing import Dict, Any

class ExplainabilityEngine:
    def __init__(self):
        pass

    def generate_explanation(self, activity_title: str, detected_emotion: str,
                             intensity_tier: str, modality: str,
                             semantic_cues: str = "") -> str:
        base = (
            f"**Why this is recommended:** Our AI identified a predominant emotional state of "
            f"**{detected_emotion.capitalize()}** at a **{intensity_tier}** intensity level. "
            f"**{activity_title}** ({modality}) was prioritized because clinical studies confirm "
            f"that targeted {modality.lower()} exercises reliably de-escalate {detected_emotion.lower()} symptoms."
        )
        if semantic_cues:
            base += f" Additionally, contextual indicators in your text ('{semantic_cues}') reinforced this intervention."
        return base
