"""Mood Mentor - Milestone 3 CLI & Pipeline Orchestrator.

Integrates Ingestion -> Preprocessing -> VADER Sentiment -> Transformer Emotions ->
Emotion Intensity & Severity Analysis (Task 1) -> Personalized Profile & Trends (Tasks 2, 6) ->
Hybrid Recommendation & Dynamic Ranking (Tasks 3, 4, 5) -> Feedback Learning (Task 7) ->
XAI Explainability (Task 8) -> Benchmark Evaluation (Task 9) -> Unified Reporting (Task 10).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

from config import (
    DATA_DIR,
    DEFAULT_CSV_TEXT_COLUMN,
    DEFAULT_CSV_USER_COLUMN,
    DEFAULT_PREDICTION_THRESHOLD,
    DEFAULT_TOP_K,
    configure_logging,
)
from emotion_classifier import TransformerEmotionClassifier
from evaluation_recommender import evaluate_recommenders, format_evaluation_report
from exceptions import MoodMentorError
from feedback_learner import FeedbackLearner
from ingestion import IngestedFeedback, validate_and_ingest
from intensity_analyzer import EmotionIntensityAnalyzer
from preprocessing import preprocess_corpus
from recommendation_engine import HybridRecommendationEngine
from report import format_terminal_report, generate_milestone3_report, to_dataframe
from sentiment import analyze_batch
from user_profile import UserProfileRegistry

logger = logging.getLogger("mood_mentor_m3")


def run_milestone3_pipeline(
    source: str | list[str],
    source_type: str = "auto",
    csv_text_col: str = DEFAULT_CSV_TEXT_COLUMN,
    csv_user_col: str = DEFAULT_CSV_USER_COLUMN,
    user_id: str | None = None,
    model_type: str = "distilbert",
    top_k: int = DEFAULT_TOP_K,
    explain: bool = True,
    output_csv_path: str | None = None,
) -> int:
    """Execute the end-to-end Milestone 3 processing pipeline."""
    try:
        logger.info("Starting Mood Mentor Milestone 3 pipeline...")

        # Stage 1: Ingestion
        ingested_records: list[IngestedFeedback] = validate_and_ingest(
            source=source,
            source_type=source_type,
            text_column=csv_text_col,
            user_id=user_id,
        )
        raw_texts = [rec.text for rec in ingested_records]
        user_ids = [rec.user_id or user_id for rec in ingested_records]
        logger.info("Stage 1 (Ingestion): %d record(s) loaded.", len(raw_texts))

        # Stage 2: Preprocessing
        preprocessed = preprocess_corpus(raw_texts)
        logger.info("Stage 2 (Preprocessing) completed.")

        # Stage 3: Baseline Sentiment (VADER)
        sentiments = analyze_batch(raw_texts)
        logger.info("Stage 3 (VADER Baseline Sentiment) completed.")

        # Stage 4: Deep Transformer Emotions (BERT / DistilBERT)
        logger.info("Initializing Transformer [%s] for emotion analysis...", model_type.upper())
        classifier = TransformerEmotionClassifier(model_type=model_type)
        emotion_predictions = classifier.predict_batch(raw_texts)
        logger.info("Stage 4 (Transformer Emotion Inference) completed.")

        # Stage 5: Emotion Intensity & Emotional State Analysis (Task 1)
        intensity_analyzer = EmotionIntensityAnalyzer()
        emotional_states = [
            intensity_analyzer.analyze(
                text=raw_texts[i],
                emotion_prediction=emotion_predictions[i],
                sentiment_score=sentiments[i],
            )
            for i in range(len(raw_texts))
        ]
        logger.info("Stage 5 (Task 1: Intensity & Severity Analysis) completed.")

        # Stage 6: Personalized Hybrid Recommendations & Dynamic Ranking (Tasks 2, 3, 4, 5, 6, 7, 8)
        profile_registry = UserProfileRegistry()
        rec_engine = HybridRecommendationEngine()
        recommendation_results = []

        for idx, state in enumerate(emotional_states):
            uid = user_ids[idx] or "anonymous_user"
            profile = profile_registry.get_or_create(uid)
            rec_res = rec_engine.generate_recommendations(
                emotional_state=state,
                user_profile=profile,
                top_k=top_k,
                explain=explain,
            )
            recommendation_results.append(rec_res)

        logger.info("Stage 6 (Tasks 2-8: Hybrid Recommendations & Dynamic Ranking) completed.")

        # Stage 7: Unified Reporting
        summary = generate_milestone3_report(
            raw_texts=raw_texts,
            user_ids=user_ids,
            emotional_states=emotional_states,
            recommendation_results=recommendation_results,
        )

        # Print Terminal Dashboard with Explainability & Trend Display
        print(format_terminal_report(summary, recommendation_results=recommendation_results))

        # Optional CSV Export
        if output_csv_path:
            out_file = Path(output_csv_path)
            out_file.parent.mkdir(parents=True, exist_ok=True)
            df = to_dataframe(summary)
            df.to_csv(out_file, index=False)
            logger.info("Exported Milestone 3 analysis to CSV at '%s'.", out_file.resolve())

        return 0

    except MoodMentorError as app_err:
        logger.error("Pipeline failed with application error: %s", app_err)
        return 1
    except Exception as exc:
        logger.critical("Unexpected failure occurred: %s", exc, exc_info=True)
        return 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="mood_mentor_m3",
        description="Mood Mentor Milestone 3 - AI Emotional State & Personalized Recommendations",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("-t", "--text", type=str, help="Direct text string to analyze.")
    group.add_argument("-f", "--file", type=str, help="Path to .txt or .csv dataset file.")

    parser.add_argument("-u", "--user-id", type=str, help="User ID for personalized profile & trend tracking.")
    parser.add_argument(
        "-m", "--model",
        choices=["distilbert", "bert"],
        default="distilbert",
        help="Transformer model backend (default: distilbert).",
    )
    parser.add_argument(
        "--csv-text-col",
        type=str,
        default=DEFAULT_CSV_TEXT_COLUMN,
        help=f"Text column name for CSV (default: '{DEFAULT_CSV_TEXT_COLUMN}').",
    )
    parser.add_argument(
        "--csv-user-col",
        type=str,
        default=DEFAULT_CSV_USER_COLUMN,
        help=f"User ID column name for CSV (default: '{DEFAULT_CSV_USER_COLUMN}').",
    )
    parser.add_argument(
        "-k", "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help=f"Number of top recommendations to return (default: {DEFAULT_TOP_K}).",
    )
    parser.add_argument(
        "--explain",
        action="store_true",
        default=True,
        help="Enable XAI explainability justifications in recommendations (enabled by default).",
    )
    parser.add_argument(
        "--no-explain",
        action="store_true",
        help="Disable XAI explainability justifications in recommendations.",
    )
    parser.add_argument("-o", "--output", type=str, help="Path to save report output as CSV.")
    parser.add_argument(
        "--evaluate-recs",
        action="store_true",
        help="Run Task 9 benchmark evaluation (Precision@K, Recall@K, NDCG, Baseline vs Advanced ML).",
    )
    parser.add_argument(
        "--feedback",
        type=str,
        help="Item ID to submit feedback for (Task 7). E.g. 'act_box_breathing'.",
    )
    parser.add_argument(
        "--action",
        choices=["viewed", "accepted", "rejected"],
        default="accepted",
        help="Feedback action (default: accepted).",
    )
    parser.add_argument(
        "--rating",
        type=float,
        help="Optional 1-5 star user satisfaction rating for feedback.",
    )
    parser.add_argument(
        "--trend",
        action="store_true",
        help="Show longitudinal trend report for specified --user-id.",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging verbosity level.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    configure_logging(level=log_level)

    base_dir = Path(__file__).parent

    # Mode 1: Benchmark Evaluation (Task 9)
    if args.evaluate_recs:
        logger.info("Executing Task 9 Recommender Benchmark Evaluation...")
        base_metrics, ml_metrics = evaluate_recommenders()
        print(format_evaluation_report(base_metrics, ml_metrics))
        sys.exit(0)

    # Mode 2: Record User Feedback (Task 7)
    if args.feedback:
        uid = args.user_id or "emp_default"
        profile_registry = UserProfileRegistry()
        profile = profile_registry.get_or_create(uid)
        learner = FeedbackLearner()
        rec = learner.record_feedback(
            user_profile=profile,
            item_id=args.feedback,
            action=args.action,
            rating=args.rating,
        )
        print("=" * 70)
        print(f"✅ FEEDBACK RECORDED SUCCESSFULLY (Task 7):")
        print(f"  • User ID  : {rec.user_id}")
        print(f"  • Item ID  : {rec.item_id}")
        print(f"  • Action   : {rec.action.upper()}")
        print(f"  • Rating   : {rec.rating or 'N/A'}")
        summary = learner.get_user_feedback_summary(uid)
        print(f"  • Total Feedback Events for {uid} : {summary['total_feedback_events']} (Acceptance Rate: {summary['acceptance_rate_pct']}%)")
        print("=" * 70)
        sys.exit(0)

    # Mode 3: View Longitudinal Trend Report (Task 6)
    if args.trend:
        uid = args.user_id or "emp_101"
        profile_registry = UserProfileRegistry()
        profile = profile_registry.get_or_create(uid)
        trend = profile.get_trend_report()
        print("=" * 70)
        print(f"📈 LONGITUDINAL EMOTIONAL TREND REPORT (Task 6):")
        print(f"  • User ID             : {trend.user_id}")
        print(f"  • Trend Classification: {trend.trend_label}")
        print(f"  • Trajectory Slope    : {trend.intensity_trajectory_slope:+.4f}")
        print(f"  • Dominant Overall    : {trend.dominant_emotion_overall.capitalize()}")
        print(f"  • Emotion Frequencies : {trend.emotion_frequencies}")
        print(f"  • Repeated Patterns   : {', '.join(trend.repeated_patterns)}")
        print(f"  • Trend Action Bias   : {trend.recommendation_bias}")
        print("=" * 70)
        sys.exit(0)

    # Mode 4: Pipeline Execution (Direct Text / File / Default)
    explain_enabled = not args.no_explain

    if args.text:
        exit_code = run_milestone3_pipeline(
            source=args.text,
            source_type="direct",
            user_id=args.user_id,
            model_type=args.model,
            top_k=args.top_k,
            explain=explain_enabled,
            output_csv_path=args.output,
        )
    elif args.file:
        exit_code = run_milestone3_pipeline(
            source=args.file,
            source_type="auto",
            csv_text_col=args.csv_text_col,
            csv_user_col=args.csv_user_col,
            user_id=args.user_id,
            model_type=args.model,
            top_k=args.top_k,
            explain=explain_enabled,
            output_csv_path=args.output,
        )
    else:
        sample_txt = base_dir / "data" / "sample.txt"
        logger.info("No input provided. Running demonstration over '%s'...", sample_txt.name)
        exit_code = run_milestone3_pipeline(
            source=str(sample_txt),
            source_type="txt",
            user_id=args.user_id,
            model_type=args.model,
            top_k=args.top_k,
            explain=explain_enabled,
        )

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
