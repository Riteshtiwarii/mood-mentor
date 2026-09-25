"""
Mood Mentor — Semantic Wellness Matcher
Matches implicit user expressions and symptoms against evidence-based wellness interventions using dense n-gram and semantic term overlap.
"""

import math
import re
from typing import List, Dict, Any, Tuple
from moodmentor.core.wellness_catalog import WellnessActivity

class SemanticMatcher:
    SYMPTOM_LEXICON = {
        "act_box_breathing": ["breathe", "breath", "pounding", "racing heart", "hyperventilating", "chest tight", "panicking", "suffocating", "shaking"],
        "act_54321_grounding": ["walls closing in", "spinning", "dizzy", "unreal", "dissociating", "detached", "losing control", "overwhelmed", "panic"],
        "act_progressive_relaxation": ["tense", "shoulders", "stiff", "jaw clenching", "knots", "physical ache", "body stress", "headache", "motor tension"],
        "act_gratitude_journaling": ["cynical", "hopeless", "unappreciated", "pointless", "drained", "empty", "gloomy", "worthless"],
        "act_thought_defusion": ["ruminating", "overthinking", "spiraling", "worst case", "catastrophizing", "failure", "cannot stop thinking"],
        "act_crisis_helpline": ["suicide", "end it all", "kill", "die", "hurt myself", "hopeless", "cannot survive", "crisis"],
        "act_mindful_walking": ["screen fatigue", "stuck", "frustrated", "irritated", "restless", "cabin fever", "office stress"],
        "act_compassion_meditation": ["lonely", "isolated", "harsh on myself", "guilt", "blame", "alienated", "inadequate"],
        "act_expressive_writing": ["rage", "furious", "betrayed", "unfair", "screaming inside", "resentful", "injustice"],
        "act_joy_savoring": ["proud", "accomplished", "happy", "grateful", "celebrate", "promotion", "success"]
    }

    def __init__(self):
        pass

    def _tokenize(self, text: str) -> List[str]:
        cleaned = re.sub(r"[^\w\s]", " ", text.lower())
        return [w for w in cleaned.split() if len(w) > 2]

    def match(self, query: str, activities: List[WellnessActivity], top_k: int = 5) -> List[Tuple[WellnessActivity, float]]:
        query_tokens = set(self._tokenize(query))
        query_text = query.lower()
        scored: List[Tuple[WellnessActivity, float]] = []

        for act in activities:
            symptoms = self.SYMPTOM_LEXICON.get(act.id, [])
            score = 0.0

            # 1. Exact phrase matching in query
            for phrase in symptoms:
                if phrase in query_text:
                    score += 0.45

            # 2. Token overlap with activity description and rationale
            act_tokens = set(self._tokenize(act.title + " " + act.description + " " + act.clinical_rationale))
            overlap = query_tokens.intersection(act_tokens)
            if query_tokens:
                jaccard = len(overlap) / (len(query_tokens) + len(act_tokens) - len(overlap) + 1e-5)
                score += jaccard * 0.55

            scored.append((act, min(1.0, score)))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]
