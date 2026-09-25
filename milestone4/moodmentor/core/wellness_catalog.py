"""
Mood Mentor — Wellness Catalog
Contains curated, evidence-based wellness interventions with structured metadata.
"""

from typing import List, Dict, Any, Optional

class WellnessActivity:
    def __init__(self, id: str, title: str, modality: str, duration_mins: int,
                 target_emotions: List[str], target_intensities: List[str],
                 description: str, clinical_rationale: str):
        self.id = id
        self.title = title
        self.modality = modality  # Breathing, Somatic, Mindfulness, Journaling, Movement
        self.duration_mins = duration_mins
        self.target_emotions = target_emotions
        self.target_intensities = target_intensities
        self.description = description
        self.clinical_rationale = clinical_rationale

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "modality": self.modality,
            "duration_mins": self.duration_mins,
            "target_emotions": self.target_emotions,
            "target_intensities": self.target_intensities,
            "description": self.description,
            "clinical_rationale": self.clinical_rationale
        }

class WellnessCatalog:
    def __init__(self):
        self._activities: List[WellnessActivity] = self._init_catalog()

    def _init_catalog(self) -> List[WellnessActivity]:
        return [
            WellnessActivity(
                id="act_box_breathing",
                title="Resonant Box Breathing (4-4-4-4)",
                modality="Breathing",
                duration_mins=4,
                target_emotions=["fear", "anger"],
                target_intensities=["Moderate", "High", "Critical"],
                description="Inhale for 4s, hold for 4s, exhale for 4s, hold for 4s. Proven to stimulate the vagus nerve and reduce acute cortisol spikes.",
                clinical_rationale="Rapidly modulates autonomic nervous system activity and lowers heart rate variability arousal."
            ),
            WellnessActivity(
                id="act_54321_grounding",
                title="5-4-3-2-1 Sensory Grounding",
                modality="Somatic",
                duration_mins=5,
                target_emotions=["fear", "surprise"],
                target_intensities=["High", "Critical"],
                description="Identify 5 things you can see, 4 you can touch, 3 you can hear, 2 you can smell, and 1 you can taste.",
                clinical_rationale="Cognitive defusion exercise that disrupts severe panic spirals by re-anchoring attention into physical reality."
            ),
            WellnessActivity(
                id="act_progressive_relaxation",
                title="Progressive Muscle Relaxation (PMR)",
                modality="Somatic",
                duration_mins=10,
                target_emotions=["anger", "fear", "sadness"],
                target_intensities=["Moderate", "High"],
                description="Systematically tense muscle groups for 5 seconds and release for 15 seconds, starting from feet to facial muscles.",
                clinical_rationale="Reduces somatic motor tension associated with chronic stress and sympathetic nervous system overload."
            ),
            WellnessActivity(
                id="act_gratitude_journaling",
                title="Micro-Gratitude Journaling",
                modality="Journaling",
                duration_mins=5,
                target_emotions=["sadness", "anger"],
                target_intensities=["Low", "Moderate"],
                description="Write down three highly specific, non-obvious occurrences from today that generated a moment of safety or contentment.",
                clinical_rationale="Directs neural plasticity toward dopamine and serotonin pathway reinforcement, countering negative cognitive bias."
            ),
            WellnessActivity(
                id="act_thought_defusion",
                title="CBT Cognitive Defusion & Reframing",
                modality="Mindfulness",
                duration_mins=7,
                target_emotions=["sadness", "anger", "fear"],
                target_intensities=["Moderate", "High"],
                description="Observe troubling automatic thoughts as passing weather rather than immutable absolute truths. Reframe one distortion.",
                clinical_rationale="Interrupts rumination cycles by weakening rigid cognitive fusion through meta-cognitive awareness."
            ),
            WellnessActivity(
                id="act_crisis_helpline",
                title="Clinical Crisis Support & Immediate Helpline",
                modality="Clinical Support",
                duration_mins=0,
                target_emotions=["fear", "sadness", "anger"],
                target_intensities=["Critical"],
                description="Connect immediately with a confidential licensed professional crisis counselor (Toll-Free 24/7).",
                clinical_rationale="Clinical emergency safety guardrail prioritizing direct human intervention during acute mental distress."
            ),
            WellnessActivity(
                id="act_mindful_walking",
                title="Brisk Mindful Walking Reset",
                modality="Movement",
                duration_mins=8,
                target_emotions=["anger", "sadness"],
                target_intensities=["Low", "Moderate"],
                description="Step away from all screens and take a mindful walk outdoors, syncing footsteps with gentle diaphragmatic breathing.",
                clinical_rationale="Promotes neurochemical balance via mild aerobic exertion and bilateral ocular stimulation."
            ),
            WellnessActivity(
                id="act_compassion_meditation",
                title="Loving-Kindness (Metta) Meditation",
                modality="Mindfulness",
                duration_mins=6,
                target_emotions=["sadness", "anger"],
                target_intensities=["Low", "Moderate"],
                description="Silently offer compassionate wishes to yourself, a respected colleague, and all living beings.",
                clinical_rationale="Upregulates oxytocin and parasympathetic activation, buffering against professional alienation and cynicism."
            ),
            WellnessActivity(
                id="act_expressive_writing",
                title="Expressive Emotional Catharsis Journaling",
                modality="Journaling",
                duration_mins=10,
                target_emotions=["anger", "sadness"],
                target_intensities=["Moderate", "High"],
                description="Free-write your raw unfiltered feelings for 10 minutes without judging spelling, logic, or politeness.",
                clinical_rationale="Facilitates emotional processing and limbic de-escalation by moving distressing affect into explicit semantic structures."
            ),
            WellnessActivity(
                id="act_joy_savoring",
                title="Peak Joy Savoring Protocol",
                modality="Mindfulness",
                duration_mins=3,
                target_emotions=["joy", "love"],
                target_intensities=["Low", "Moderate", "High"],
                description="Pause completely to mentally amplify a recent achievement or genuine connection, anchoring physical sensations.",
                clinical_rationale="Broaden-and-build psychological model: strengthens resilience reserves by prolonging positive affective states."
            )
        ]

    def get_all_activities(self) -> List[WellnessActivity]:
        return list(self._activities)

    def get_by_id(self, activity_id: str) -> Optional[WellnessActivity]:
        for act in self._activities:
            if act.id == activity_id:
                return act
        return None

    def filter_activities(self, modality: Optional[str] = None,
                          intensity: Optional[str] = None,
                          emotion: Optional[str] = None) -> List[WellnessActivity]:
        results = []
        for act in self._activities:
            if modality and act.modality.lower() != modality.lower():
                continue
            if intensity and intensity not in act.target_intensities:
                continue
            if emotion and emotion.lower() not in [e.lower() for e in act.target_emotions]:
                continue
            results.append(act)
        return results
