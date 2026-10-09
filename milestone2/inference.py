"""
Milestone 2: Reusable PyTorch DistilBERT Inference Pipeline.
Provides standalone emotion probability predictions for all 6 target emotions.
Supports CPU and CUDA execution without training on startup.
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional, Union

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

logger = logging.getLogger(__name__)

TARGET_EMOTIONS = ["joy", "sadness", "anger", "fear", "surprise", "disgust"]
ID2LABEL = {0: "joy", 1: "sadness", 2: "anger", 3: "fear", 4: "surprise", 5: "disgust"}
LABEL2ID = {v: k for k, v in ID2LABEL.items()}


class EmotionPredictor:
    """
    Reusable DistilBERT Emotion Classification Predictor.
    """

    def __init__(self, model_dir: Optional[str] = None, device: Optional[str] = None):
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.tokenizer = None
        self.model = None
        self.model_used = "Uninitialized"
        self._load_model(model_dir)

    def _has_weights(self, path: str) -> bool:
        if not path or not os.path.exists(path):
            return False
        weight_files = ["model.safetensors", "pytorch_model.bin", "model.safetensors.index.json", "pytorch_model.bin.index.json"]
        return os.path.exists(os.path.join(path, "config.json")) and any(os.path.exists(os.path.join(path, wf)) for wf in weight_files)

    def _load_model(self, custom_model_dir: Optional[str]):
        # Candidate model directories
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        candidates = [
            custom_model_dir,
            os.path.join(os.path.dirname(__file__), "models", "distilbert"),
            os.path.join(base_dir, "milestone4", "models", "distilbert"),
            os.path.join(base_dir, "models", "distilbert")
        ]

        valid_path = None
        for p in candidates:
            if p and self._has_weights(p):
                valid_path = p
                break

        try:
            if valid_path:
                logger.info(f"Loading fine-tuned DistilBERT model from: {valid_path}")
                self.tokenizer = AutoTokenizer.from_pretrained(valid_path)
                self.model = AutoModelForSequenceClassification.from_pretrained(valid_path)
                self.model_used = f"Fine-Tuned DistilBERT ({os.path.basename(valid_path)})"
            else:
                logger.info("No local fine-tuned weights found. Initializing base 'distilbert-base-uncased'...")
                self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    "distilbert-base-uncased",
                    num_labels=len(TARGET_EMOTIONS),
                    problem_type="multi_label_classification",
                    id2label=ID2LABEL,
                    label2id=LABEL2ID
                )
                self.model_used = "Base DistilBERT (distilbert-base-uncased)"

            self.model.to(self.device)
            self.model.eval()
            logger.info(f"EmotionPredictor initialized successfully on device: {self.device}")
        except Exception as e:
            logger.error(f"Failed to load DistilBERT model: {e}")
            self.model = None
            self.tokenizer = None
            self.model_used = "Lexical Fallback (Initialization Error)"

    def predict_emotions(self, text: str) -> Dict[str, Any]:
        """
        Predict emotion probabilities for all 6 target emotions for a single input text.

        Returns:
            Dict containing:
            - 'emotions': Dict[str, float] mapping emotion label -> probability (0.0 to 1.0)
            - 'dominant_emotion': str (highest probability emotion label)
            - 'confidence': float (highest probability value)
            - 'model_used': str metadata
        """
        if not text or not isinstance(text, str) or not text.strip():
            return {
                "emotions": {e: 0.0 for e in TARGET_EMOTIONS},
                "dominant_emotion": "neutral",
                "confidence": 0.0,
                "model_used": self.model_used
            }

        if self.model is not None and self.tokenizer is not None:
            try:
                inputs = self.tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=128,
                    padding=True
                ).to(self.device)

                with torch.no_grad():
                    logits = self.model(**inputs).logits
                    probs = torch.sigmoid(logits).squeeze(0).tolist()

                if isinstance(probs, float):
                    probs = [probs]
                
                emotion_probs = {}
                for idx, emotion in enumerate(TARGET_EMOTIONS):
                    prob_val = float(probs[idx]) if idx < len(probs) else 0.0
                    emotion_probs[emotion] = round(max(0.0, min(1.0, prob_val)), 4)

                dominant = max(emotion_probs.items(), key=lambda x: x[1])

                return {
                    "emotions": emotion_probs,
                    "dominant_emotion": dominant[0],
                    "confidence": dominant[1],
                    "model_used": self.model_used
                }
            except Exception as e:
                logger.warning(f"DistilBERT inference error: {e}. Falling back to rule-based prior.")

        return self._rule_based_fallback(text)

    def predict_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Predict emotion probabilities for a batch of texts.
        """
        return [self.predict_emotions(txt) for txt in texts]

    def _rule_based_fallback(self, text: str) -> Dict[str, Any]:
        text_lower = text.lower()
        fear_kws = ["panic", "breath", "fear", "scared", "terrified", "nervous", "anxious", "worry"]
        anger_kws = ["angry", "furious", "mad", "frustrated", "rage", "irritated", "annoyed", "hate"]
        joy_kws = ["happy", "celebrate", "proud", "fulfilled", "glad", "joy", "excited", "delighted"]
        sad_kws = ["sad", "depressed", "exhausted", "drained", "burnout", "lonely", "hopeless", "crying"]
        su_kws = ["surprise", "shocked", "amazed", "stunned", "unexpected"]
        dis_kws = ["disgust", "revolting", "nauseating", "sick", "gross", "vile"]

        counts = {
            "joy": sum(1 for w in joy_kws if w in text_lower),
            "sadness": sum(1 for w in sad_kws if w in text_lower),
            "anger": sum(1 for w in anger_kws if w in text_lower),
            "fear": sum(1 for w in fear_kws if w in text_lower),
            "surprise": sum(1 for w in su_kws if w in text_lower),
            "disgust": sum(1 for w in dis_kws if w in text_lower)
        }
        total = sum(counts.values())

        if total > 0:
            probs = {k: round(0.05 + 0.90 * (v / total), 4) for k, v in counts.items()}
        else:
            probs = {"joy": 0.20, "sadness": 0.20, "anger": 0.15, "fear": 0.15, "surprise": 0.15, "disgust": 0.15}

        dominant = max(probs.items(), key=lambda x: x[1])
        return {
            "emotions": probs,
            "dominant_emotion": dominant[0],
            "confidence": dominant[1],
            "model_used": "Rule-Based Lexical Fallback"
        }


if __name__ == "__main__":
    predictor = EmotionPredictor()
    test_text = "I am feeling extremely stressed and anxious about the upcoming deadline, but hopeful that we will succeed!"
    res = predictor.predict_emotions(test_text)
    print("Inference Test Result:")
    print(json.dumps(res, indent=2))
