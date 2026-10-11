from nlp_ingestion.preprocessing import (
    normalize_text,
    tokenize_and_lemmatize,
    preprocess_text,
)


def test_normalize_text():
    text = "  I AM   VERY HAPPY!!!  "

    result = normalize_text(text)

    assert result == "i am very happy"


def test_tokenization():
    text = "I enjoy working with my team"

    tokens = tokenize_and_lemmatize(text)

    assert isinstance(tokens, list)
    assert len(tokens) > 0


def test_stopwords_are_removed():
    text = "I am working on the project"

    tokens = tokenize_and_lemmatize(text)

    assert "I" not in tokens
    assert "am" not in tokens


def test_lemmatization():
    text = "Employees are working on different projects"

    tokens = tokenize_and_lemmatize(text)

    assert "employee" in tokens
    assert "work" in tokens


def test_preprocess_output_structure():
    text = "I am feeling stressed about my workload"

    result = preprocess_text(text)

    assert "normalized_text" in result
    assert "tokens" in result
    assert "processed_text" in result

    assert isinstance(result["tokens"], list)
    assert isinstance(result["processed_text"], str)