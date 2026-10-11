from nlp_ingestion.pii_cleaner import clean_pii


def test_email_is_removed():
    text = "Contact me at employee@example.com"
    result = clean_pii(text)

    assert "[EMAIL]" in result
    assert "employee@example.com" not in result


def test_phone_number_is_removed():
    text = "My contact number is +91 9876543210"
    result = clean_pii(text)

    assert "[PHONE]" in result
    assert "9876543210" not in result


def test_url_is_removed():
    text = "Visit https://example.com for more information"
    result = clean_pii(text)

    assert "[URL]" in result
    assert "https://example.com" not in result


def test_normal_text_is_preserved():
    text = "I enjoy working with my team"
    result = clean_pii(text)

    assert result == text
def test_person_name_is_removed():
    text = "John Smith is working on the employee wellness project."

    result = clean_pii(text)

    assert "[PERSON]" in result
    assert "John Smith" not in result