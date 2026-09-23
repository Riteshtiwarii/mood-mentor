"""Comprehensive Test Suite for Mood Mentor - Milestone 2.

Validates Tasks 1 through 10:
- Task 1: BERT Model Integration
- Task 2: DistilBERT Model Integration
- Task 3: Multi-Label Emotion Classification (Joy, Sadness, Anger, Fear, Surprise, Disgust)
- Task 4: Confidence Score Validation & Probabilities
- Task 5: Model Evaluation Metrics (Accuracy, Precision, Recall, Macro F1) & Comparison
- Task 6: ISEAR Benchmark Validation
- Task 7: Milestone 1 & Milestone 2 Integration Pipeline
- Task 8: Edge Case Validation (Emojis, Short/Long texts, Ambiguous, Empty, Invalid inputs)
- Task 9: Exception Handling & Project Architecture
- Task 10: Final End-to-End Pipeline & CSV Export
"""

from __future__ import annotations

from pathlib import Path
import tempfile
import pytest

from config import (
    DATA_DIR,
    DEFAULT_PREDICTION_THRESHOLD,
    EMOTION_LABELS,
    NUM_EMOTIONS,
)
from emotion_classifier import EmotionPrediction, TransformerEmotionClassifier
from evaluate import (
    compare_models,
    evaluate_model_on_dataset,
    run_isear_benchmark,
)
from exceptions import (
    EmotionModelError,
    IngestionError,
    MoodMentorError,
    PreprocessingError,
    ReportGenerationError,
)
from ingestion import ingest_direct_text, validate_and_ingest
from main import run_pipeline
from preprocessing import TextPreprocessor, preprocess_corpus, preprocess_text
from report import generate_unified_report, to_dataframe
from sentiment import analyze_sentiment


# ============================================================================
# TASK 1 & TASK 2: BERT & DISTILBERT INTEGRATION TESTS
# ============================================================================


class TestTransformerModelIntegration:
    """Validate BERT and DistilBERT model loading, tokenization, and inference."""

    @pytest.fixture(scope="class")
    def distilbert_clf(self) -> TransformerEmotionClassifier:
        return TransformerEmotionClassifier(model_type="distilbert")

    @pytest.fixture(scope="class")
    def bert_clf(self) -> TransformerEmotionClassifier:
        return TransformerEmotionClassifier(model_type="bert")

    def test_distilbert_inference_generates_prediction(
        self, distilbert_clf: TransformerEmotionClassifier
    ) -> None:
        text = "I am delighted and thrilled by our team's outstanding victory!"
        pred = distilbert_clf.predict(text)

        assert isinstance(pred, EmotionPrediction)
        assert pred.text == text
        assert pred.model_name == "distilbert"
        assert len(pred.probabilities) == NUM_EMOTIONS
        for emotion in EMOTION_LABELS:
            assert emotion in pred.probabilities
            assert 0.0 <= pred.probabilities[emotion] <= 1.0

    def test_bert_inference_generates_prediction(
        self, bert_clf: TransformerEmotionClassifier
    ) -> None:
        text = "I am terrified of losing my job in the upcoming corporate layoffs."
        pred = bert_clf.predict(text)

        assert isinstance(pred, EmotionPrediction)
        assert pred.model_name == "bert"
        assert len(pred.probabilities) == NUM_EMOTIONS
        assert pred.primary_emotion in EMOTION_LABELS
        assert 0.0 <= pred.primary_confidence <= 1.0

    def test_batch_prediction(self, distilbert_clf: TransformerEmotionClassifier) -> None:
        texts = [
            "What a magnificent achievement!",
            "I feel so lonely and heartbroken.",
        ]
        preds = distilbert_clf.predict_batch(texts)
        assert len(preds) == 2
        assert preds[0].primary_confidence > 0.0
        assert preds[1].primary_confidence > 0.0


# ============================================================================
# TASK 3: MULTI-LABEL EMOTION CLASSIFICATION TESTS
# ============================================================================


class TestMultiLabelEmotionClassification:
    """Validate multi-label classification across the 6 core emotion dimensions."""

    @pytest.fixture(scope="class")
    def classifier(self) -> TransformerEmotionClassifier:
        return TransformerEmotionClassifier(model_type="distilbert")

    def test_all_six_emotions_represented(self, classifier: TransformerEmotionClassifier) -> None:
        pred = classifier.predict("Testing emotions spectrum")
        assert set(pred.probabilities.keys()) == set(EMOTION_LABELS)

    def test_multi_label_mixed_emotions(self, classifier: TransformerEmotionClassifier) -> None:
        """Verify model handles mixed feelings with multiple active emotions."""
        text = "I am excited about the new opportunity but nervous about the outcome."
        pred = classifier.predict(text)

        assert isinstance(pred.predicted_emotions, list)
        assert pred.primary_emotion in ("joy", "fear")
        # Ensure confidence scores exist for both
        assert pred.probabilities["joy"] > 0.0
        assert pred.probabilities["fear"] > 0.0

    def test_strong_emotional_expression(self, classifier: TransformerEmotionClassifier) -> None:
        text = "I am furious and disgusted with the leadership decisions!"
        pred = classifier.predict(text)

        assert pred.probabilities["anger"] > 0.0
        assert pred.probabilities["disgust"] > 0.0


# ============================================================================
# TASK 4: CONFIDENCE SCORE VALIDATION TESTS
# ============================================================================


class TestConfidenceScoreValidation:
    """Validate dynamic calculation of confidence scores and thresholds."""

    @pytest.fixture(scope="class")
    def classifier(self) -> TransformerEmotionClassifier:
        return TransformerEmotionClassifier(model_type="distilbert")

    def test_confidence_scores_are_dynamic_no_hardcoding(
        self, classifier: TransformerEmotionClassifier
    ) -> None:
        pred1 = classifier.predict("I feel so joyful and thrilled today!")
        pred2 = classifier.predict("This bad experience makes me sad and depressed.")

        assert pred1.probabilities != pred2.probabilities
        assert pred1.primary_confidence != pred2.primary_confidence

    def test_primary_emotion_matches_highest_confidence(
        self, classifier: TransformerEmotionClassifier
    ) -> None:
        text = "We achieved an incredible breakthrough and won the award!"
        pred = classifier.predict(text)

        highest_emotion = max(pred.probabilities.items(), key=lambda x: x[1])[0]
        assert pred.primary_emotion == highest_emotion
        assert pred.primary_confidence == pred.probabilities[highest_emotion]

    def test_custom_threshold_filtering(self) -> None:
        high_threshold_clf = TransformerEmotionClassifier(threshold=0.99)
        pred = high_threshold_clf.predict("Some neutral statement")
        # With threshold 0.99, only primary fallback should appear if non-negligible
        assert len(pred.predicted_emotions) <= 1


# ============================================================================
# TASK 5: MODEL EVALUATION & COMPARISON TESTS
# ============================================================================


class TestModelEvaluation:
    """Validate calculation of Accuracy, Precision, Recall, and Macro F1."""

    def test_evaluation_metrics_generation(self) -> None:
        val_csv = DATA_DIR / "emotion_val.csv"
        assert val_csv.exists(), "emotion_val.csv must exist"

        clf = TransformerEmotionClassifier(model_type="distilbert")
        metrics = evaluate_model_on_dataset(clf, val_csv)

        assert metrics.sample_count > 0
        assert 0.0 <= metrics.subset_accuracy <= 1.0
        assert 0.0 <= metrics.hamming_accuracy <= 1.0
        assert 0.0 <= metrics.macro_precision <= 1.0
        assert 0.0 <= metrics.macro_recall <= 1.0
        assert 0.0 <= metrics.macro_f1 <= 1.0
        assert metrics.avg_inference_time_ms > 0.0

    def test_model_comparison(self) -> None:
        bert_metrics, distilbert_metrics, best_model = compare_models()

        assert best_model in ("bert", "distilbert")
        assert bert_metrics.macro_f1 >= 0.0
        assert distilbert_metrics.macro_f1 >= 0.0


# ============================================================================
# TASK 6: ISEAR BENCHMARK VALIDATION TESTS
# ============================================================================


class TestISEARBenchmarkValidation:
    """Validate model against the held-out ISEAR benchmark subset."""

    def test_isear_benchmark_execution(self) -> None:
        isear_csv = DATA_DIR / "isear_subset.csv"
        assert isear_csv.exists(), "isear_subset.csv must exist"

        clf = TransformerEmotionClassifier(model_type="distilbert")
        results = run_isear_benchmark(clf, isear_csv=isear_csv)

        assert results["total_samples"] > 0
        assert results["overall_accuracy_pct"] >= 0.0
        assert "emotion_breakdown" in results
        for emotion in ("joy", "sadness", "anger", "fear", "disgust"):
            assert emotion in results["emotion_breakdown"]


# ============================================================================
# TASK 7: MILESTONE 1 & MILESTONE 2 INTEGRATION TESTS
# ============================================================================


class TestPipelineIntegration:
    """Validate end-to-end integration of M1 (Sentiment) and M2 (Emotions)."""

    def test_unified_report_generation(self) -> None:
        raw_texts = [
            "I love working with this supportive and talented team!",
            "I am feeling completely exhausted and overwhelmed by the unmanageable workload.",
        ]
        preprocessed = preprocess_corpus(raw_texts)
        sentiments = [analyze_sentiment(t) for t in raw_texts]
        clf = TransformerEmotionClassifier(model_type="distilbert")
        emotions = clf.predict_batch(raw_texts)

        summary = generate_unified_report(raw_texts, preprocessed, sentiments, emotions)

        assert summary.total_samples == 2
        assert len(summary.items) == 2
        assert summary.items[0].sentiment_label in ("Positive", "Negative", "Neutral")
        assert summary.items[0].primary_emotion in EMOTION_LABELS

        # Test DataFrame conversion
        df = to_dataframe(summary)
        assert len(df) == 2
        assert "Sentiment" in df.columns
        assert "Primary Emotion" in df.columns
        assert "Confidence" in df.columns


# ============================================================================
# TASK 8: EDGE CASE VALIDATION TESTS
# ============================================================================


class TestEdgeCaseValidation:
    """Validate system robustness against diverse edge-case inputs."""

    @pytest.fixture(scope="class")
    def classifier(self) -> TransformerEmotionClassifier:
        return TransformerEmotionClassifier(model_type="distilbert")

    def test_emoji_handling(self, classifier: TransformerEmotionClassifier) -> None:
        text_with_emoji = "Great news everyone! 🎉😊"
        preprocessed = preprocess_text(text_with_emoji)
        assert "happy" in preprocessed.cleaned_text or "celebration" in preprocessed.cleaned_text

        pred = classifier.predict(text_with_emoji)
        assert pred.primary_confidence > 0.0

    def test_short_and_informal_text(self, classifier: TransformerEmotionClassifier) -> None:
        pred_short = classifier.predict("yesss!!")
        assert pred_short.primary_confidence >= 0.0

    def test_long_paragraph_text(self, classifier: TransformerEmotionClassifier) -> None:
        long_text = "Our organization has faced immense restructuring. " * 10
        pred_long = classifier.predict(long_text)
        assert pred_long.primary_emotion in EMOTION_LABELS

    def test_empty_and_whitespace_input(self, classifier: TransformerEmotionClassifier) -> None:
        pred_empty = classifier.predict("   \n\t  ")
        assert pred_empty.primary_confidence == 0.0
        assert pred_empty.predicted_emotions == []

    def test_invalid_type_raises_error(self, classifier: TransformerEmotionClassifier) -> None:
        with pytest.raises(EmotionModelError):
            classifier.predict(9999)  # type: ignore


# ============================================================================
# TASK 9 & 10: PROJECT CLEANUP & END-TO-END CLI VALIDATION
# ============================================================================


class TestFinalCLIValidation:
    """Validate end-to-end execution of main.py and CSV exports."""

    def test_run_pipeline_on_sample_txt(self) -> None:
        sample_txt = DATA_DIR / "sample.txt"
        exit_code = run_pipeline(str(sample_txt), source_type="txt", model_type="distilbert")
        assert exit_code == 0

    def test_run_pipeline_on_csv_with_export(self, tmp_path: Path) -> None:
        sample_csv = DATA_DIR / "sample.csv"
        out_csv = tmp_path / "m2_report.csv"

        exit_code = run_pipeline(
            str(sample_csv),
            source_type="csv",
            csv_column="text",
            model_type="distilbert",
            output_csv_path=str(out_csv),
        )
        assert exit_code == 0
        assert out_csv.exists()
        assert out_csv.stat().st_size > 0
