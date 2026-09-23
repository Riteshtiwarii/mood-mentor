"""Comprehensive Test Suite for Mood Mentor - Milestone 1.

Validates Tasks 1 through 5:
- Task 1: Text Ingestion Workflow (Direct, TXT, CSV, Validation, Error Handling)
- Task 2: Text Preprocessing (Noise filtering, Tokenization, Negation preservation, Lemmatization)
- Task 3: VADER Baseline Sentiment (Compound, Pos/Neg/Neu scores, Classification)
- Task 4: Emotion & Sentiment Reporting (Aggregates, itemized breakdown, DataFrame export)
- Task 5: Complete Pipeline Integration (End-to-End Orchestration)
"""

from __future__ import annotations

from pathlib import Path
import tempfile
import pytest

from config import NEGATIVE_THRESHOLD, POSITIVE_THRESHOLD
from exceptions import (
    IngestionError,
    MoodMentorError,
    PreprocessingError,
    ReportGenerationError,
    SentimentAnalysisError,
)
from ingestion import (
    ingest_csv_file,
    ingest_direct_text,
    ingest_txt_file,
    validate_and_ingest,
)
from main import run_pipeline
from preprocessing import (
    TextPreprocessor,
    preprocess_corpus,
    preprocess_text,
)
from report import (
    format_terminal_report,
    generate_sentiment_report,
    to_dataframe,
)
from sentiment import (
    VaderSentimentAnalyzer,
    analyze_batch,
    analyze_sentiment,
)


# ============================================================================
# TASK 1: INGESTION WORKFLOW TESTS
# ============================================================================


class TestIngestionWorkflow:
    """Task 1 Test Suite: Text Ingestion Validation."""

    def test_ingest_direct_text_single_and_multiline(self) -> None:
        single = "  I love the company culture!  "
        result_single = ingest_direct_text(single)
        assert len(result_single) == 1
        assert result_single[0] == "I love the company culture!"

        multiline = "Line 1\n\n  Line 2  \nLine 3\n"
        result_multi = ingest_direct_text(multiline)
        assert result_multi == ["Line 1", "Line 2", "Line 3"]

    def test_ingest_direct_text_empty_and_invalid(self) -> None:
        with pytest.raises(IngestionError, match="cannot be empty"):
            ingest_direct_text("   \n\t  ")

        with pytest.raises(IngestionError, match="Expected string input"):
            ingest_direct_text(12345)  # type: ignore

    def test_ingest_txt_file_valid(self, tmp_path: Path) -> None:
        txt_file = tmp_path / "valid.txt"
        txt_file.write_text("Sentence one.\n\nSentence two.\nSentence three.\n", encoding="utf-8")

        lines = ingest_txt_file(txt_file)
        assert len(lines) == 3
        assert lines[0] == "Sentence one."
        assert lines[1] == "Sentence two."
        assert lines[2] == "Sentence three."

    def test_ingest_txt_file_invalid_and_missing(self, tmp_path: Path) -> None:
        # Non-existent file
        with pytest.raises(IngestionError, match="File not found"):
            ingest_txt_file(tmp_path / "does_not_exist.txt")

        # Wrong extension
        wrong_ext = tmp_path / "file.json"
        wrong_ext.write_text('{"key": "value"}', encoding="utf-8")
        with pytest.raises(IngestionError, match="Invalid file extension"):
            ingest_txt_file(wrong_ext)

        # Empty txt file
        empty_txt = tmp_path / "empty.txt"
        empty_txt.write_text("   \n\n   ", encoding="utf-8")
        with pytest.raises(IngestionError, match="contains no valid text"):
            ingest_txt_file(empty_txt)

    def test_ingest_csv_file_valid(self, tmp_path: Path) -> None:
        csv_file = tmp_path / "feedback.csv"
        csv_file.write_text(
            "id,department,text\n"
            "1,HR,\"Great benefits.\"\n"
            "2,Sales,\"Too much pressure.\"\n"
            "3,Engineering,\"   \"\n"  # Empty row should be skipped
            "4,Ops,\"Factual status update.\"\n",
            encoding="utf-8",
        )

        rows = ingest_csv_file(csv_file, text_column="text")
        assert len(rows) == 3
        assert rows == ["Great benefits.", "Too much pressure.", "Factual status update."]

    def test_ingest_csv_custom_column_and_missing_col(self, tmp_path: Path) -> None:
        csv_file = tmp_path / "custom.csv"
        csv_file.write_text(
            "id,comment\n"
            "1,\"Positive team energy.\"\n",
            encoding="utf-8",
        )

        # Custom column
        result = ingest_csv_file(csv_file, text_column="comment")
        assert result == ["Positive team energy."]

        # Missing column
        with pytest.raises(IngestionError, match="Column 'text' not found"):
            ingest_csv_file(csv_file, text_column="text")

    def test_validate_and_ingest_gateway(self, tmp_path: Path) -> None:
        # String list input
        list_input = ["  Valid item 1  ", "", "Valid item 2"]
        assert validate_and_ingest(list_input) == ["Valid item 1", "Valid item 2"]

        # Direct text auto-detection
        assert validate_and_ingest("Direct text input") == ["Direct text input"]

        # None input
        with pytest.raises(IngestionError, match="Source cannot be None"):
            validate_and_ingest(None)  # type: ignore


# ============================================================================
# TASK 2: PREPROCESSING WORKFLOW TESTS
# ============================================================================


class TestPreprocessingWorkflow:
    """Task 2 Test Suite: Text Preprocessing Validation."""

    @pytest.fixture
    def preprocessor(self) -> TextPreprocessor:
        return TextPreprocessor(preserve_negations=True)

    def test_noise_filtering(self, preprocessor: TextPreprocessor) -> None:
        raw = "Check our policy at https://wellness.corp/faq &amp; contact hr@corp.com! <p>Thanks</p>"
        cleaned = preprocessor.clean_noise(raw)
        assert "http" not in cleaned
        assert "@" not in cleaned
        assert "<p>" not in cleaned
        assert "</p>" not in cleaned
        assert "&" not in cleaned
        assert "policy" in cleaned
        assert "thanks" in cleaned

    def test_tokenization_and_special_chars(self, preprocessor: TextPreprocessor) -> None:
        text = "Well-being & work-life balance are high-priority!!"
        tokens = preprocessor.tokenize(preprocessor.clean_noise(text))
        assert "well-being" in tokens or "balance" in tokens
        assert "!" not in tokens

    def test_negation_preservation_in_stopwords(self, preprocessor: TextPreprocessor) -> None:
        """Verify sentiment-critical inversion words ('not', 'never', 'no') are retained."""
        text = "This project is not good and I will never recommend it."
        result = preprocessor.process(text)
        tokens_lower = [t.lower() for t in result.tokens]
        lemmas_lower = [l.lower() for l in result.lemmas]

        assert "not" in tokens_lower, "Negation 'not' must be preserved!"
        assert "never" in tokens_lower, "Negation 'never' must be preserved!"
        assert "not" in lemmas_lower
        assert "never" in lemmas_lower

    def test_lemmatization(self, preprocessor: TextPreprocessor) -> None:
        text = "employees are collaborating and resolving critical blockers"
        result = preprocessor.process(text)
        # Verify base forms
        reconstructed = result.reconstructed_text()
        assert "employee" in reconstructed or "collaborate" in reconstructed or "resolve" in reconstructed

    def test_empty_and_whitespace_preprocessing(self, preprocessor: TextPreprocessor) -> None:
        result = preprocessor.process("   \n\t  ")
        assert result.cleaned_text == ""
        assert result.tokens == []
        assert result.lemmas == []
        assert result.token_count == 0

    def test_preprocess_corpus_batch(self) -> None:
        corpus = [
            "Outstanding collaborative effort today!",
            "Completely exhausted and stressed out.",
        ]
        results = preprocess_corpus(corpus)
        assert len(results) == 2
        assert results[0].token_count > 0
        assert results[1].token_count > 0

    def test_invalid_type_raises_preprocessing_error(self) -> None:
        with pytest.raises(PreprocessingError):
            preprocess_text(1234)  # type: ignore


# ============================================================================
# TASK 3: VADER SENTIMENT TESTS
# ============================================================================


class TestVaderSentimentWorkflow:
    """Task 3 Test Suite: VADER Sentiment Validation."""

    @pytest.fixture
    def analyzer(self) -> VaderSentimentAnalyzer:
        return VaderSentimentAnalyzer(
            positive_threshold=POSITIVE_THRESHOLD,
            negative_threshold=NEGATIVE_THRESHOLD,
        )

    def test_positive_sentiment_detection(self, analyzer: VaderSentimentAnalyzer) -> None:
        text = "I absolutely love working with this amazingly supportive and talented team!"
        score = analyzer.analyze(text)

        assert score.compound >= POSITIVE_THRESHOLD
        assert score.label == "Positive"
        assert score.pos > 0.0
        assert 0.0 <= score.compound <= 1.0
        assert 0.0 <= score.pos <= 1.0
        assert 0.0 <= score.neu <= 1.0
        assert 0.0 <= score.neg <= 1.0

    def test_negative_sentiment_detection(self, analyzer: VaderSentimentAnalyzer) -> None:
        text = "I am feeling completely exhausted, stressed, and overwhelmed by this terrible workload."
        score = analyzer.analyze(text)

        assert score.compound <= NEGATIVE_THRESHOLD
        assert score.label == "Negative"
        assert score.neg > 0.0
        assert -1.0 <= score.compound <= 0.0

    def test_neutral_sentiment_detection(self, analyzer: VaderSentimentAnalyzer) -> None:
        text = "The quarterly financial meeting took place on Tuesday at 10:00 AM."
        score = analyzer.analyze(text)

        assert NEGATIVE_THRESHOLD < score.compound < POSITIVE_THRESHOLD
        assert score.label == "Neutral"
        assert score.neu > 0.5

    def test_negation_sentiment_polarity(self, analyzer: VaderSentimentAnalyzer) -> None:
        positive_text = "The system is great."
        negated_text = "The system is not great."

        pos_score = analyzer.analyze(positive_text)
        negated_score = analyzer.analyze(negated_text)

        assert pos_score.compound > negated_score.compound
        assert negated_score.compound <= 0.0

    def test_empty_string_sentiment(self, analyzer: VaderSentimentAnalyzer) -> None:
        score = analyzer.analyze("   ")
        assert score.compound == 0.0
        assert score.label == "Neutral"

    def test_dynamic_scoring_no_hardcoding(self, analyzer: VaderSentimentAnalyzer) -> None:
        score1 = analyzer.analyze("Excellent performance!")
        score2 = analyzer.analyze("Good performance.")
        assert score1.compound != score2.compound
        assert score1.compound > score2.compound

    def test_analyze_batch(self) -> None:
        texts = [
            "Terrific achievement by the engineering squad!",
            "Catastrophic failure and complete disaster.",
            "The quarterly financial meeting took place on Tuesday at 10:00 AM.",
        ]
        scores = analyze_batch(texts)
        assert len(scores) == 3
        assert scores[0].label == "Positive"
        assert scores[1].label == "Negative"
        assert scores[2].label == "Neutral"


# ============================================================================
# TASK 4: REPORTING & METRICS TESTS
# ============================================================================


class TestReportingWorkflow:
    """Task 4 Test Suite: Initial Sentiment/Emotion Report Validation."""

    def test_generate_sentiment_report_metrics(self) -> None:
        raw_texts = [
            "We had a fantastic breakthrough today!",
            "The update was deployed at noon.",
            "This delay is unacceptable and frustrating.",
        ]
        preprocessed = preprocess_corpus(raw_texts)
        sentiments = analyze_batch(raw_texts)

        report = generate_sentiment_report(raw_texts, preprocessed, sentiments)

        assert report.total_samples == 3
        assert report.positive_count == 1
        assert report.neutral_count == 1
        assert report.negative_count == 1
        assert pytest.approx(report.positive_pct, 0.1) == 33.33
        assert pytest.approx(report.negative_pct, 0.1) == 33.33
        assert pytest.approx(report.neutral_pct, 0.1) == 33.33
        assert report.min_compound_sample is not None
        assert report.max_compound_sample is not None
        assert report.max_compound_sample.compound_score > report.min_compound_sample.compound_score

    def test_report_terminal_formatting(self) -> None:
        raw_texts = ["Outstanding team effort!"]
        preprocessed = preprocess_corpus(raw_texts)
        sentiments = analyze_batch(raw_texts)
        report = generate_sentiment_report(raw_texts, preprocessed, sentiments)

        formatted_text = format_terminal_report(report)
        assert "MOOD MENTOR - MILESTONE 1 REPORT" in formatted_text
        assert "Total Analyzed Samples : 1" in formatted_text
        assert "Positive Samples" in formatted_text
        assert "Outstanding team effort!" in formatted_text

    def test_report_to_dataframe(self) -> None:
        raw_texts = ["Great job!", "Terrible experience."]
        preprocessed = preprocess_corpus(raw_texts)
        sentiments = analyze_batch(raw_texts)
        report = generate_sentiment_report(raw_texts, preprocessed, sentiments)

        df = to_dataframe(report)
        assert len(df) == 2
        assert "Input Text" in df.columns
        assert "Processed Text" in df.columns
        assert "Sentiment" in df.columns
        assert "Compound" in df.columns
        assert df.iloc[0]["Sentiment"] == "Positive"
        assert df.iloc[1]["Sentiment"] == "Negative"

    def test_report_length_mismatch_raises_error(self) -> None:
        with pytest.raises(ReportGenerationError, match="Mismatched input lengths"):
            generate_sentiment_report(["Text 1"], [], [])


# ============================================================================
# TASK 5: COMPLETE PIPELINE INTEGRATION TESTS
# ============================================================================


class TestPipelineIntegration:
    """Task 5 Test Suite: Complete Pipeline Integration."""

    def test_pipeline_on_sample_txt(self) -> None:
        sample_path = Path(__file__).parent / "data" / "sample.txt"
        assert sample_path.exists(), "sample.txt must exist"

        exit_code = run_pipeline(str(sample_path), source_type="txt")
        assert exit_code == 0

    def test_pipeline_on_sample_csv(self, tmp_path: Path) -> None:
        sample_path = Path(__file__).parent / "data" / "sample.csv"
        assert sample_path.exists(), "sample.csv must exist"

        out_csv = tmp_path / "output_test.csv"
        exit_code = run_pipeline(
            str(sample_path),
            source_type="csv",
            csv_column="text",
            output_csv_path=str(out_csv),
        )
        assert exit_code == 0
        assert out_csv.exists()
        assert out_csv.stat().st_size > 0

    def test_pipeline_on_direct_text(self) -> None:
        exit_code = run_pipeline(
            "I feel energized and thrilled to work on this initiative!",
            source_type="direct",
        )
        assert exit_code == 0

    def test_pipeline_graceful_error_handling(self, tmp_path: Path) -> None:
        # Pass non-existent file
        exit_code = run_pipeline(
            str(tmp_path / "missing.txt"),
            source_type="txt",
        )
        assert exit_code == 1
