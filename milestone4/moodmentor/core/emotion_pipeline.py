"""
Mood Mentor — Unified Real ML Emotion & Sentiment Pipeline
Integrates:
- Milestone 1: VADER Sentiment Analysis (Compound, Polarity)
- Milestone 2 & 3: DistilBERT Deep Multi-Label Emotion Classification (6 Core Dimensions)
- Robust fallback to lexical/keyword distribution if weights are being loaded or unavailable
"""

import os
import logging
from typing import Dict, Tuple, Any, Optional

logger = logging.getLogger(__name__)

class UnifiedEmotionPipeline:
    EMOTION_LABELS = ["joy", "sadness", "anger", "fear", "surprise", "disgust"]

    def __init__(self, model_dir: Optional[str] = None):
        self.model_dir = model_dir
        self.tokenizer = None
        self.model = None
        self.vader = None
        self._init_models()

    def _init_models(self):
        # 1. Initialize VADER (Milestone 1)
        try:
            from nltk.sentiment.vader import SentimentIntensityAnalyzer
            import nltk
            try:
                self.vader = SentimentIntensityAnalyzer()
            except Exception:
                nltk.download("vader_lexicon", quiet=True)
                self.vader = SentimentIntensityAnalyzer()
        except Exception as e:
            logger.warning(f"VADER init warning: {e}")
            self.vader = None

        # 2. Initialize DistilBERT (Milestone 2 & 3)
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForSequenceClassification

            # Potential candidates for model path
            base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
            candidates = [
                self.model_dir,
                os.path.join(base_dir, "models", "distilbert"),
                "/Users/riteshtiwari/Downloads/mood_mentor_repo/milestone3/models/distilbert",
                "/Users/riteshtiwari/Downloads/mood_mentor_milestone4/models/distilbert",
                "/Users/riteshtiwari/.gemini/antigravity/scratch/mood_mentor_milestone4/models/distilbert",
            ]
            valid_path = None
            for p in candidates:
                if p and os.path.exists(os.path.join(p, "config.json")):
                    valid_path = p
                    break

            if valid_path:
                self.tokenizer = AutoTokenizer.from_pretrained(valid_path)
                self.model = AutoModelForSequenceClassification.from_pretrained(valid_path)
                self.model.eval()
                logger.info(f"Loaded Transformer model from {valid_path}")
        except Exception as e:
            logger.warning(f"Transformer model load warning: {e}")
            self.model = None
            self.tokenizer = None

    def analyze(self, text: str) -> Tuple[Dict[str, float], float, Dict[str, Any]]:
        """
        Returns:
            - emotion_probs: Dict[str, float] for the 6 core emotions
            - compound_sentiment: float (-1.0 to 1.0)
            - meta: Dict with model info and raw sentiment scores
        """
        # Sentiment via VADER
        compound = 0.0
        sentiment_scores = {"neg": 0.0, "neu": 1.0, "pos": 0.0, "compound": 0.0}
        if self.vader:
            try:
                sentiment_scores = self.vader.polarity_scores(text)
                compound = float(sentiment_scores["compound"])
            except Exception:
                pass

        # Transformer Inference
        emotion_probs = None
        model_used = "lexical_prior"

        if self.model and self.tokenizer:
            try:
                import torch
                inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
                with torch.no_grad():
                    logits = self.model(**inputs).logits
                    probs = torch.sigmoid(logits).squeeze().tolist()
                
                if isinstance(probs, list) and len(probs) >= len(self.EMOTION_LABELS):
                    emotion_probs = {
                        label: round(float(prob), 4)
                        for label, prob in zip(self.EMOTION_LABELS, probs[:len(self.EMOTION_LABELS)])
                    }
                    model_used = "DistilBERT Neural Network (Fine-Tuned Safetensors)"
            except Exception as e:
                logger.warning(f"Inference runtime error: {e}")

        # Lexical fallback if model was not loaded
        if not emotion_probs:
            emotion_probs = self._lexical_probs(text, compound)

        return emotion_probs, compound, {
            "model_used": model_used,
            "sentiment_scores": sentiment_scores
        }

    def _lexical_probs(self, text: str, compound: float) -> Dict[str, float]:
        text_lower = text.lower()
        fear_kws = ["panic", "breath", "pounding", "fear", "scared", "terrified", "walls closing", "heart", "nervous", "anxiety", "anxious"]
        anger_kws = ["angry", "furious", "mad", "frustrated", "pressure", "rage", "irritated", "annoyed", "unfair", "hate"]
        joy_kws = ["happy", "celebrated", "proud", "fulfilled", "accomplished", "glad", "joy", "excited", "victory", "success", "delighted"]
        sad_kws = ["sad", "depressed", "exhausted", "drained", "burnout", "lonely", "hopeless", "crying", "miserable", "fatigue", "tired"]
        surprise_kws = ["surprise", "shocked", "amazed", "stunned", "unexpected", "astonished"]
        disgust_kws = ["disgust", "revolting", "nauseating", "sick", "gross", "vile"]

        f_cnt = sum(1 for w in fear_kws if w in text_lower)
        a_cnt = sum(1 for w in anger_kws if w in text_lower)
        j_cnt = sum(1 for w in joy_kws if w in text_lower)
        s_cnt = sum(1 for w in sad_kws if w in text_lower)
        su_cnt = sum(1 for w in surprise_kws if w in text_lower)
        d_cnt = sum(1 for w in disgust_kws if w in text_lower)

        total = f_cnt + a_cnt + j_cnt + s_cnt + su_cnt + d_cnt
        if total > 0:
            return {
                "fear": round(0.08 + 0.85 * (f_cnt / total), 4),
                "sadness": round(0.08 + 0.85 * (s_cnt / total), 4),
                "anger": round(0.08 + 0.85 * (a_cnt / total), 4),
                "joy": round(0.08 + 0.85 * (j_cnt / total), 4),
                "surprise": round(0.04 + 0.85 * (su_cnt / total), 4),
                "disgust": round(0.04 + 0.85 * (d_cnt / total), 4)
            }
        
        # Sentiment-based prior
        if compound >= 0.2:
            return {"joy": 0.84, "surprise": 0.16, "fear": 0.05, "sadness": 0.04, "anger": 0.03, "disgust": 0.02}
        elif compound <= -0.4:
            return {"sadness": 0.65, "fear": 0.58, "anger": 0.42, "disgust": 0.25, "surprise": 0.10, "joy": 0.05}
        else:
            return {"sadness": 0.25, "fear": 0.22, "anger": 0.18, "joy": 0.20, "surprise": 0.12, "disgust": 0.08}
