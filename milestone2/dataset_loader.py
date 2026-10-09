"""
Milestone 2: Dataset Loader for GoEmotions mapped to 6 Ekman Emotions.
Handles multi-label binary vectors for DistilBERT fine-tuning and evaluation.
"""

import os
import ssl
import logging
from typing import Dict, List, Tuple, Any

# Ensure SSL certificate verification issues on Windows do not block dataset loading
try:
    ssl._create_default_https_context = ssl._create_unverified_context
    import urllib3.util.ssl_ as u_ssl
    _ctx = ssl.create_default_context()
    _ctx.check_hostname = False
    _ctx.verify_mode = ssl.CERT_NONE
    u_ssl.create_urllib3_context = lambda *a, **kw: _ctx

    import requests
    requests.packages.urllib3.disable_warnings()

    import httpx
    _old_init = httpx.Client.__init__
    httpx.Client.__init__ = lambda self, *a, **kw: _old_init(self, *a, **{**kw, "verify": False})
except Exception:
    pass

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from datasets import load_dataset

logger = logging.getLogger(__name__)

# Target 6 Ekman emotions
TARGET_EMOTIONS = ["joy", "sadness", "anger", "fear", "surprise", "disgust"]
LABEL2ID = {label: i for i, label in enumerate(TARGET_EMOTIONS)}
ID2LABEL = {i: label for i, label in enumerate(TARGET_EMOTIONS)}

# Mapping from go_emotions 28 fine-grained labels to 6 target Ekman emotions
GO_EMOTIONS_MAPPING = {
    # Joy
    0: "joy",   # admiration -> joy
    1: "joy",   # amusement -> joy
    4: "joy",   # approval -> joy
    5: "joy",   # caring -> joy
    13: "joy",  # excitement -> joy
    15: "joy",  # gratitude -> joy
    17: "joy",  # joy -> joy
    18: "joy",  # love -> joy
    20: "joy",  # optimism -> joy
    21: "joy",  # pride -> joy
    23: "joy",  # relief -> joy
    # Sadness
    9: "sadness",   # disappointment -> sadness
    12: "sadness",  # embarrassment -> sadness
    16: "sadness",  # grief -> sadness
    24: "sadness",  # remorse -> sadness
    25: "sadness",  # sadness -> sadness
    # Anger
    2: "anger",   # anger -> anger
    3: "anger",   # annoyance -> anger
    10: "anger",  # disapproval -> anger
    # Fear
    14: "fear",   # fear -> fear
    19: "fear",   # nervousness -> fear
    # Surprise
    22: "surprise",  # realization -> surprise
    26: "surprise",  # surprise -> surprise
    # Disgust
    11: "disgust",  # disgust -> disgust
}

class EmotionMultiLabelDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[List[float]], tokenizer: Any, max_len: int = 128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.float)
        }

def map_go_emotions_to_multilabel(go_label_indices: List[int]) -> List[float]:
    """
    Converts a list of go_emotions label indices into a 6-element binary multi-label vector.
    """
    vector = [0.0] * 6
    for idx in go_label_indices:
        target_emotion = GO_EMOTIONS_MAPPING.get(idx)
        if target_emotion and target_emotion in LABEL2ID:
            vector[LABEL2ID[target_emotion]] = 1.0
    return vector

def load_emotion_data(model_name: str = "distilbert-base-uncased", max_len: int = 128) -> Tuple[Dict[str, List], Any]:
    """
    Loads go_emotions dataset and maps to 6-label multi-label format.
    Returns datasets dict and tokenizer.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    raw_dataset = load_dataset("google-research-datasets/go_emotions", "simplified")

    processed_data = {}
    for split in ["train", "validation", "test"]:
        texts = raw_dataset[split]["text"]
        raw_labels = raw_dataset[split]["labels"]
        multilabels = [map_go_emotions_to_multilabel(lbls) for lbls in raw_labels]
        
        filtered_texts = []
        filtered_labels = []
        for txt, ml in zip(texts, multilabels):
            if sum(ml) > 0:
                filtered_texts.append(txt)
                filtered_labels.append(ml)
        
        processed_data[split] = EmotionMultiLabelDataset(
            texts=filtered_texts,
            labels=filtered_labels,
            tokenizer=tokenizer,
            max_len=max_len
        )
        logger.info(f"Split {split}: {len(filtered_texts)} mapped samples.")

    return processed_data, tokenizer

def get_dataloaders(model_name: str = "distilbert-base-uncased", batch_size: int = 16, max_len: int = 128) -> Tuple[DataLoader, DataLoader, DataLoader, Any]:
    datasets, tokenizer = load_emotion_data(model_name=model_name, max_len=max_len)
    train_loader = DataLoader(datasets["train"], batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(datasets["validation"], batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(datasets["test"], batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader, tokenizer
