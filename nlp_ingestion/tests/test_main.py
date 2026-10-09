from nlp_ingestion.main import process_employee_text


def test_complete_employee_text_pipeline():
    text = (
        "I am not happy with my workload. "
        "Contact me at employee@example.com."
    )

    result = process_employee_text(text)

    # Check main output sections
    assert "raw_text" in result
    assert "pii_cleaned_text" in result
    assert "normalized_text" in result
    assert "tokens" in result
    assert "processed_text" in result
    assert "negation_text" in result
    assert "sentiment" in result

    # Check PII masking
    assert "[EMAIL]" in result["pii_cleaned_text"]
    assert "employee@example.com" not in result["pii_cleaned_text"]

    # Check negation handling
    assert "not" in result["negation_text"]
    assert "happy_NEG" in result["negation_text"]

    # Check sentiment structure
    sentiment = result["sentiment"]

    assert "positive" in sentiment
    assert "negative" in sentiment
    assert "neutral" in sentiment
    assert "compound" in sentiment
    assert "sentiment" in sentiment

    # The example should be negative
    assert sentiment["sentiment"] == "negative"
    assert sentiment["compound"] < -0.05