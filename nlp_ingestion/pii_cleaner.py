import re
import spacy


# Load spaCy English model
nlp = spacy.load("en_core_web_sm")


def remove_emails(text: str) -> str:
    """Replace email addresses with [EMAIL]."""
    pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
    return re.sub(pattern, "[EMAIL]", text)


def remove_phone_numbers(text: str) -> str:
    """Replace common phone-number formats with [PHONE]."""
    pattern = r"(?<!\d)(?:\+?\d[\d\s().-]{8,}\d)(?!\d)"
    return re.sub(pattern, "[PHONE]", text)


def remove_urls(text: str) -> str:
    """Replace URLs with [URL]."""
    pattern = r"https?://\S+|www\.\S+"
    return re.sub(pattern, "[URL]", text, flags=re.IGNORECASE)


def remove_person_names(text: str) -> str:
    """
    Detect person names using spaCy Named Entity Recognition
    and replace them with [PERSON].
    """

    doc = nlp(text)

    replacements = []

    for entity in doc.ents:
        if entity.label_ == "PERSON":
            replacements.append(
                (entity.start_char, entity.end_char, "[PERSON]")
            )

    # Replace from right to left so character positions remain valid
    for start, end, replacement in reversed(replacements):
        text = text[:start] + replacement + text[end:]

    return text


def clean_pii(text: str) -> str:
    """
    Remove or mask common personally identifiable information.

    PII handled:
    - Email addresses
    - Phone numbers
    - URLs
    - Person names
    """

    if not isinstance(text, str):
        raise TypeError("Input text must be a string.")

    text = remove_emails(text)
    text = remove_phone_numbers(text)
    text = remove_urls(text)
    text = remove_person_names(text)

    return text