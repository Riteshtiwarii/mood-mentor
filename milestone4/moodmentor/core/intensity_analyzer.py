"""
Mood Mentor — Intensity & Clinical Triage Analyzer
Calculates continuous intensity, discrete tiers, severity triage, and compound psychological states.
"""

import re
from typing import Dict, Any, List, Optional

class EmotionalState:
    def __init__(self, primary_emotion: str, confidence: float,
                 intensity_score: float, tier: str, triage_level: str,
                 mixed_state: Optional[str] = None, triggers_detected: Optional[List[str]] = None):
        self.primary_emotion = primary_emotion
        self.confidence = confidence
        self.intensity_score = intensity_score  # 0.05 to 1.0
        self.tier = tier  # Low, Moderate, High, Severe
        self.triage_level = triage_level  # Mild, Moderate, High, Critical
        self.mixed_state = mixed_state
        self.triggers_detected = triggers_detected or []

    def to_dict(self) -> Dict[str, Any]:
        return {
            "primary_emotion": self.primary_emotion,
            "confidence": round(self.confidence, 4),
            "intensity_score": round(self.intensity_score, 4),
            "tier": self.tier,
            "triage_level": self.triage_level,
            "mixed_state": self.mixed_state,
            "triggers_detected": self.triggers_detected
        }

class IntensityAnalyzer:
    CRISIS_KEYWORDS = [
        "suicide", "kill myself", "end my life", "want to die", "self harm",
        "hurt myself", "cut myself", "cannot go on", "no reason to live"
    ]

    AMPLIFIERS = [
        "extremely", "completely", "unbearably", "terribly", "immensely",
        "severely", "overwhelmingly", "totally", "uncontrollably", "desperately"
    ]

    DIMINISHERS = [
        "a bit", "slightly", "somewhat", "mildly", "kind of", "sort of", "a little"
    ]

    def __init__(self):
        pass

    def analyze(self, text: str, emotion_probs: Dict[str, float], vader_compound: float = 0.0) -> EmotionalState:
        text_lower = text.lower()
        
        # 1. Critical Clinical Safety Trigger Check
        detected_triggers = [kw for kw in self.CRISIS_KEYWORDS if kw in text_lower]
        if detected_triggers:
            return EmotionalState(
                primary_emotion="fear",
                confidence=1.0,
                intensity_score=1.0,
                tier="Severe",
                triage_level="Critical",
                mixed_state="Acute Crisis State",
                triggers_detected=detected_triggers
            )

        # 2. Dominant Emotion Selection
        sorted_emotions = sorted(emotion_probs.items(), key=lambda x: x[1], reverse=True)
        primary_emotion, confidence = sorted_emotions[0]
        secondary_emotion, sec_confidence = sorted_emotions[1] if len(sorted_emotions) > 1 else (None, 0.0)

        # 3. Base Intensity Calculation from Probabilities & Valence
        base_intensity = confidence * 0.65 + abs(vader_compound) * 0.35

        # 4. Lexical Modifiers (Amplifiers & Exclamations)
        amp_count = sum(1 for amp in self.AMPLIFIERS if amp in text_lower)
        dim_count = sum(1 for dim in self.DIMINISHERS if dim in text_lower)
        exclamation_count = text.count("!")

        multiplier = 1.0 + (amp_count * 0.12) + (min(exclamation_count, 3) * 0.05) - (dim_count * 0.15)
        continuous_intensity = max(0.05, min(1.0, base_intensity * multiplier))

        # 5. Discrete Tier Mapping
        if continuous_intensity < 0.35:
            tier = "Low"
        elif continuous_intensity < 0.65:
            tier = "Moderate"
        elif continuous_intensity < 0.85:
            tier = "High"
        else:
            tier = "Severe"

        # 6. Clinical Triage Level
        if tier == "Severe" or continuous_intensity >= 0.80:
            triage_level = "High" if primary_emotion not in ["joy", "love"] else "Mild"
        elif tier == "High":
            triage_level = "Moderate" if primary_emotion not in ["joy", "love"] else "Mild"
        elif tier == "Moderate":
            triage_level = "Mild"
        else:
            triage_level = "Mild"

        # 7. Mixed Psychological States
        mixed_state = None
        if sec_confidence >= 0.25:
            combo = {primary_emotion, secondary_emotion}
            if combo == {"joy", "fear"}:
                mixed_state = "Anticipatory Anxiety (Excitement + Apprehension)"
            elif combo == {"anger", "sadness"}:
                mixed_state = "Frustrated Grief / Burnout Resentment"
            elif combo == {"fear", "sadness"}:
                mixed_state = "Hopeless Panic / Vulnerability"
            elif combo == {"joy", "love"}:
                mixed_state = "Prosocial Fulfillment"

        return EmotionalState(
            primary_emotion=primary_emotion,
            confidence=confidence,
            intensity_score=continuous_intensity,
            tier=tier,
            triage_level=triage_level,
            mixed_state=mixed_state,
            triggers_detected=detected_triggers
        )
