"""
Mood Mentor — Command-Line Interface (CLI) Entry Point
Enables terminal commands: analyze, dashboard, report, purge, and stats.
"""

import argparse
import sys
import os
import subprocess

from moodmentor.core.wellness_catalog import WellnessCatalog
from moodmentor.core.intensity_analyzer import IntensityAnalyzer
from moodmentor.core.recommendation_engine import RecommendationEngine
from moodmentor.core.user_profile import UserProfile
from moodmentor.storage.db import DatabaseManager
from moodmentor.reporting.exporter import ReportExporter

def main():
    parser = argparse.ArgumentParser(
        prog="moodmentor",
        description="Mood Mentor — Enterprise AI Psychological Wellness Platform CLI"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available sub-commands")

    # Command 1: Analyze
    p_analyze = subparsers.add_parser("analyze", help="Analyze text for emotions and get recommendations")
    p_analyze.add_argument("text", type=str, help="Text to analyze")
    p_analyze.add_argument("--user", type=str, default="cli_user", help="User ID")

    # Command 2: Dashboard
    p_dash = subparsers.add_parser("dashboard", help="Launch the interactive Streamlit web dashboard")
    p_dash.add_argument("--port", type=int, default=8501, help="Port to run Streamlit on")

    # Command 3: Report
    p_report = subparsers.add_parser("report", help="Export telemetry report as CSV or PDF")
    p_report.add_argument("--user", type=str, required=True, help="User ID to export")
    p_report.add_argument("--format", choices=["csv", "pdf"], default="csv", help="Report format")
    p_report.add_argument("--output", type=str, default=None, help="Output destination filepath")

    # Command 4: Stats
    subparsers.add_parser("stats", help="Display local database telemetry statistics")

    # Command 5: Purge
    p_purge = subparsers.add_parser("purge", help="GDPR data purge: permanently delete user records")
    p_purge.add_argument("--user", type=str, required=True, help="User ID to delete")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    db = DatabaseManager()

    if args.command == "analyze":
        analyzer = IntensityAnalyzer()
        recommender = RecommendationEngine()
        text = args.text
        
        # Simple heuristic mapping for CLI
        t_low = text.lower()
        if any(w in t_low for w in ["panic", "breath", "pounding", "fear", "anxious"]):
            probs = {"fear": 0.85, "sadness": 0.05, "anger": 0.04, "joy": 0.02, "love": 0.02, "surprise": 0.02}
            compound = -0.80
        elif any(w in t_low for w in ["happy", "great", "proud", "fulfilled"]):
            probs = {"joy": 0.88, "love": 0.06, "surprise": 0.02, "fear": 0.01, "anger": 0.01, "sadness": 0.02}
            compound = 0.85
        else:
            probs = {"sadness": 0.45, "fear": 0.25, "anger": 0.15, "joy": 0.08, "love": 0.04, "surprise": 0.03}
            compound = -0.35

        state = analyzer.analyze(text, probs, compound)
        recs = recommender.get_recommendations(state, probs, query_text=text, top_k=3)

        print("\n" + "=" * 65)
        print("  🧠 MOOD MENTOR CLI ANALYSIS REPORT")
        print("=" * 65)
        print(f"Input: \"{text}\"")
        print(f"Dominant Emotion: {state.primary_emotion.upper()} (Confidence: {state.confidence:.2f})")
        print(f"Intensity Score : {state.intensity_score:.2f} [{state.tier} Tier]")
        print(f"Severity Triage : {state.triage_level}")
        if state.mixed_state:
            print(f"Compound State  : {state.mixed_state}")
        print("-" * 65)
        print("TOP RECOMMENDATIONS:")
        for i, r in enumerate(recs):
            print(f"  {i+1}. {r.activity.title} [{r.activity.modality}] (Score: {r.score:.3f})")
            print(f"     Rationale: {r.rationale}")
        print("=" * 65 + "\n")

    elif args.command == "dashboard":
        dash_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "dashboard", "app.py")
        print(f"🚀 Launching Mood Mentor Dashboard on http://localhost:{args.port}...")
        subprocess.run(["streamlit", "run", dash_path, "--server.port", str(args.port)])

    elif args.command == "report":
        exporter = ReportExporter(db)
        out = args.output or f"moodmentor_report_{args.user}.{args.format}"
        if args.format == "csv":
            path = exporter.export_interactions_csv(args.user, out)
        else:
            path = exporter.export_executive_pdf(args.user, out)
        print(f"✅ Generated {args.format.upper()} report at: {path}")

    elif args.command == "stats":
        stats = db.get_database_stats()
        print("\n📊 TELEMETRY DATABASE STATS:")
        print(f"  • Total Interactions: {stats['total_interactions']}")
        print(f"  • Total Recommendations: {stats['total_recommendations']}")
        print(f"  • Total Feedback Records: {stats['total_feedback']}\n")

    elif args.command == "purge":
        res = db.purge_user_data(args.user)
        print(f"🗑️ GDPR Purge completed for user '{args.user}'. Deleted records: {res}")

if __name__ == "__main__":
    main()
