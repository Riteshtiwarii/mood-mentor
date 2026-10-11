from nlp_ingestion.negation import (
    preserve_negations,
    mark_negations,
    handle_negation,
)


def test_not_is_preserved():
    text = "I am not happy"

    result = preserve_negations(text)

    assert "not" in result


def test_contraction_is_normalized():
    text = "I dont like this"

    result = preserve_negations(text)

    assert "don't" in result


def test_negated_word_is_marked():
    text = "not happy"

    result = mark_negations(text)

    assert "not" in result
    assert "happy_NEG" in result


def test_multiple_negation_words():
    text = "I never feel comfortable"

    result = handle_negation(text)

    assert "never" in result
    assert "feel_NEG" in result


def test_non_negated_text():
    text = "I enjoy my work"

    result = handle_negation(text)

    assert "enjoy" in result
    assert "enjoy_NEG" not in result