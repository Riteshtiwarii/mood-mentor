from .pii_cleaner import clean_pii
from .preprocessing import preprocess_text
from .negation import handle_negation
from .sentiment import analyze_sentiment

def process_employee_text(text: str) -> dict:
    """
    Complete WellSync NLP ingestion pipeline.

    Pipeline:
    Raw text
        ↓
    PII cleaning
        ↓
    Text preprocessing
        ↓
    Negation handling
        ↓
    VADER sentiment analysis
    """

    if not isinstance(text, str):
        raise TypeError("Input text must be a string.")

    # 1. Remove/mask PII
    pii_cleaned_text = clean_pii(text)

    # 2. Normalize, tokenize and lemmatize
    preprocessing_result = preprocess_text(pii_cleaned_text)

    # 3. Preserve negation information
    negation_text = handle_negation(pii_cleaned_text)

    # 4. VADER sentiment analysis
    sentiment_result = analyze_sentiment(pii_cleaned_text)

    # 5. Structured output for the next pipeline stage
    return {
        "raw_text": text,
        "pii_cleaned_text": pii_cleaned_text,
        "normalized_text": preprocessing_result["normalized_text"],
        "tokens": preprocessing_result["tokens"],
        "processed_text": preprocessing_result["processed_text"],
        "negation_text": negation_text,
        "sentiment": sentiment_result
    }


if __name__ == "__main__":

    sample_text = (
        "I am not happy with my workload. "
        "Please contact me at teja@example.com."
    )

    result = process_employee_text(sample_text)

    print("\n===== WellSync NLP Pipeline =====\n")

    for key, value in result.items():
        print(f"{key}: {value}") 