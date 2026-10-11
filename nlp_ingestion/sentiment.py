from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


# Create VADER analyzer once
analyzer = SentimentIntensityAnalyzer()


def analyze_sentiment(text: str) -> dict:
    """
    Analyze sentiment using VADER.

    Returns:
        positive score
        negative score
        neutral score
        compound score
        sentiment label
    """

    if not isinstance(text, str):
        raise TypeError("Input text must be a string.")

    scores = analyzer.polarity_scores(text)

    compound = scores["compound"]

    if compound >= 0.05:
        label = "positive"
    elif compound <= -0.05:
        label = "negative"
    else:
        label = "neutral"

    return {
        "positive": round(scores["pos"], 4),
        "negative": round(scores["neg"], 4),
        "neutral": round(scores["neu"], 4),
        "compound": round(compound, 4),
        "sentiment": label
    }