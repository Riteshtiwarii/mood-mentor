import re
import spacy


# Load English spaCy model
nlp = spacy.load("en_core_web_sm")


def normalize_text(text: str) -> str:
    """
    Normalize text by:
    - converting to lowercase
    - removing extra whitespace
    - removing unnecessary special characters
    """
    if not isinstance(text, str):
        raise TypeError("Input text must be a string.")

    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def tokenize_and_lemmatize(text: str) -> list[str]:
    """
    Tokenize text and perform lemmatization using spaCy.
    Stopwords and punctuation are removed.
    """

    doc = nlp(text)

    tokens = []

    for token in doc:
        if token.is_space or token.is_punct:
            continue

        if token.is_stop:
            continue

        tokens.append(token.lemma_)

    return tokens


def preprocess_text(text: str) -> dict:
    """
    Complete preprocessing pipeline.
    """

    normalized_text = normalize_text(text)

    tokens = tokenize_and_lemmatize(normalized_text)

    return {
        "normalized_text": normalized_text,
        "tokens": tokens,
        "processed_text": " ".join(tokens)
    }