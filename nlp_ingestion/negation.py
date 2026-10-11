import re


NEGATION_WORDS = {
    "not",
    "no",
    "never",
    "none",
    "neither",
    "nor",
    "don't",
    "doesn't",
    "didn't",
    "isn't",
    "aren't",
    "wasn't",
    "weren't",
    "can't",
    "cannot",
    "couldn't",
    "won't",
    "wouldn't",
    "shouldn't",
    "mustn't",
    "haven't",
    "hasn't",
    "hadn't",
}


def preserve_negations(text: str) -> str:
    """
    Preserve important negation words while cleaning text.
    """

    if not isinstance(text, str):
        raise TypeError("Input text must be a string.")

    text = text.lower()

    # Normalize common contractions
    replacements = {
        "dont": "don't",
        "doesnt": "doesn't",
        "didnt": "didn't",
        "isnt": "isn't",
        "arent": "aren't",
        "wasnt": "wasn't",
        "werent": "weren't",
        "cant": "can't",
        "couldnt": "couldn't",
        "wont": "won't",
        "wouldnt": "wouldn't",
        "shouldnt": "shouldn't",
        "havent": "haven't",
        "hasnt": "hasn't",
        "hadnt": "hadn't",
        "mustnt": "mustn't",
    }

    for original, corrected in replacements.items():
        text = re.sub(
            rf"\b{re.escape(original)}\b",
            corrected,
            text
        )

    return text


def mark_negations(text: str) -> str:
    """
    Mark words affected by negation.

    Example:
        'not happy' -> 'not happy_NEG'
    """

    if not isinstance(text, str):
        raise TypeError("Input text must be a string.")

    words = text.split()
    result = []

    negation_active = False

    for word in words:
        clean_word = word.strip(".,!?;:")

        if clean_word in NEGATION_WORDS:
            result.append(clean_word)
            negation_active = True
            continue

        if negation_active:
            result.append(f"{clean_word}_NEG")
            negation_active = False
        else:
            result.append(clean_word)

    return " ".join(result)


def handle_negation(text: str) -> str:
    """
    Complete negation-handling pipeline.
    """

    text = preserve_negations(text)
    text = mark_negations(text)

    return text