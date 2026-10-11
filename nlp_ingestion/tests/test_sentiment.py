from nlp_ingestion.sentiment import analyze_sentiment


def test_positive_sentiment():
    result = analyze_sentiment("I love my work!")

    assert result["sentiment"] == "positive"
    assert result["compound"] > 0.05


def test_negative_sentiment():
    result = analyze_sentiment("I hate my workload.")

    assert result["sentiment"] == "negative"
    assert result["compound"] < -0.05


def test_neutral_sentiment():
    result = analyze_sentiment("I have a meeting at 10 AM.")

    assert result["sentiment"] == "neutral"
    assert result["compound"] == 0.0


def test_sentiment_scores_exist():
    result = analyze_sentiment("I enjoy working with my team.")

    assert "positive" in result
    assert "negative" in result
    assert "neutral" in result
    assert "compound" in result
    assert "sentiment" in result


def test_scores_are_valid():
    result = analyze_sentiment("I am happy with my work.")

    assert 0.0 <= result["positive"] <= 1.0
    assert 0.0 <= result["negative"] <= 1.0
    assert 0.0 <= result["neutral"] <= 1.0
    assert -1.0 <= result["compound"] <= 1.0