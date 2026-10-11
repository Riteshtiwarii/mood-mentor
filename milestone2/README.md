# Milestone 2: Deep Learning Emotion Classification

This module implements the **DistilBERT-based 6-Class Emotion Classification Pipeline** for the MoodMentor platform.

## Overview
- **Base Architecture:** `distilbert-base-uncased` (PyTorch + Hugging Face Transformers)
- **Problem Type:** Multi-label sequence classification (`BCEWithLogitsLoss` with Sigmoid activation)
- **Target Emotion Dimensions:**
  1. `joy` (ID 0)
  2. `sadness` (ID 1)
  3. `anger` (ID 2)
  4. `fear` (ID 3)
  5. `surprise` (ID 4)
  6. `disgust` (ID 5)

## Dataset & Label Mapping
Trained on Google Research's **GoEmotions** real human text benchmark dataset (43,410 train, 5,426 val, 5,427 test samples), mapped to the 6 primary Ekman emotion dimensions:
- **Joy:** `joy`, `amusement`, `approval`, `caring`, `excitement`, `gratitude`, `love`, `optimism`, `pride`, `relief`, `admiration`
- **Sadness:** `sadness`, `disappointment`, `embarrassment`, `grief`, `remorse`
- **Anger:** `anger`, `annoyance`, `disapproval`
- **Fear:** `fear`, `nervousness`
- **Surprise:** `surprise`, `realization`
- **Disgust:** `disgust`

## Directory Structure
```text
milestone2/
├── README.md                  # Milestone 2 documentation
├── requirements.txt           # Python package requirements
├── config.json                # Emotion label configuration & training hyperparameters
├── dataset_loader.py          # GoEmotions dataset loader & Ekman mapping
├── train.py                   # PyTorch DistilBERT fine-tuning script
├── evaluate.py                # Evaluation script computing F1-micro, F1-macro, Precision, Recall
├── inference.py               # Reusable standalone inference pipeline
└── models/
    └── distilbert/
        └── config.json        # Saved model metadata (label2id / id2label)
```

## Usage

### 1. Reusable Inference (Dashboard / API Integration)
```python
from milestone2.inference import EmotionPredictor

predictor = EmotionPredictor()
result = predictor.predict_emotions("I feel anxious about the presentation tomorrow, but excited for the results!")

print(result)
# Output:
# {
#   "emotions": {
#     "joy": 0.4215,
#     "sadness": 0.0512,
#     "anger": 0.0312,
#     "fear": 0.7814,
#     "surprise": 0.1245,
#     "disgust": 0.0123
#   },
#   "dominant_emotion": "fear",
#   "confidence": 0.7814,
#   "model_used": "Fine-Tuned DistilBERT"
# }
```

### 2. Model Training
To fine-tune DistilBERT on the GoEmotions dataset:
```bash
python milestone2/train.py --epochs 2 --batch_size 16 --lr 3e-5
```

### 3. Model Evaluation
To evaluate the fine-tuned model on the held-out test split:
```bash
python milestone2/evaluate.py --threshold 0.3
```
Evaluation metrics will be saved to `milestone2/evaluation_report.json`.
