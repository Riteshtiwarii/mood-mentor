"""
Mood Mentor — Report Exporter (CSV & PDF)
Generates structured downloadable CSV data dumps and executive PDF summary reports.
"""

import csv
import os
from datetime import datetime
from typing import List, Dict, Any, Optional
from moodmentor.storage.db import DatabaseManager

class ReportExporter:
    def __init__(self, db: DatabaseManager):
        self.db = db

    def export_interactions_csv(self, user_id: Optional[str], output_path: str,
                                from_date: Optional[str] = None,
                                to_date: Optional[str] = None) -> str:
        """Exports user interactions to CSV format"""
        from moodmentor.search.filter_engine import FilterEngine
        engine = FilterEngine()
        records = engine.search_interactions(self.db, user_id=user_id, from_date=from_date, to_date=to_date, limit=1000)

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        headers = ["ID", "User ID", "Timestamp", "Primary Emotion", "Confidence",
                   "Intensity Score", "Intensity Tier", "Triage Level", "VADER Polarity", "Input Text"]

        with open(output_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            for r in records:
                writer.writerow([
                    r["id"], r["user_id"], r["timestamp"], r["primary_emotion"],
                    round(r["confidence"], 3), round(r["intensity_score"], 3),
                    r["tier"], r["triage_level"], round(r["vader_compound"], 3), r["raw_text"]
                ])

        return output_path

    def export_recommendations_csv(self, user_id: Optional[str], output_path: str) -> str:
        records = self.db.get_user_recommendations(user_id=user_id, limit=1000)
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        headers = ["ID", "Interaction ID", "User ID", "Timestamp", "Activity ID",
                   "Title", "Modality", "Composite Score", "Clinical Rationale"]

        with open(output_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            for r in records:
                writer.writerow([
                    r["id"], r["interaction_id"], r["user_id"], r["timestamp"],
                    r["activity_id"], r["title"], r["modality"], round(r["score"], 3), r["rationale"]
                ])

        return output_path

    def export_executive_pdf(self, user_id: str, output_path: str,
                             from_date: Optional[str] = None,
                             to_date: Optional[str] = None) -> str:
        """
        Generates an executive PDF report summarizing longitudinal wellness telemetry,
        severity triage breakdown, and prioritized recommendations.
        Uses pure-Python PDF byte generation for guaranteed universal deployment.
        """
        from moodmentor.search.filter_engine import FilterEngine
        engine = FilterEngine()
        interactions = engine.search_interactions(self.db, user_id=user_id, from_date=from_date, to_date=to_date, limit=100)
        recs = self.db.get_user_recommendations(user_id=user_id, limit=20)

        total_sessions = len(interactions)
        if total_sessions > 0:
            avg_intensity = sum(i["intensity_score"] for i in interactions) / total_sessions
            emotions = [i["primary_emotion"] for i in interactions]
            dominant_emotion = max(set(emotions), key=emotions.count)
            critical_alerts = sum(1 for i in interactions if i["triage_level"] in ["High", "Critical"])
        else:
            avg_intensity = 0.0
            dominant_emotion = "None"
            critical_alerts = 0

        # Try generating via reportlab if installed, else fallback to standard PDF stream
        try:
            return self._build_reportlab_pdf(output_path, user_id, total_sessions,
                                            dominant_emotion, avg_intensity,
                                            critical_alerts, interactions, recs)
        except ImportError:
            return self._build_pure_pdf(output_path, user_id, total_sessions,
                                       dominant_emotion, avg_intensity,
                                       critical_alerts, interactions, recs)

    def _build_pure_pdf(self, output_path: str, user_id: str, total_sessions: int,
                        dominant_emotion: str, avg_intensity: float,
                        critical_alerts: int, interactions: List[Dict[str, Any]],
                        recs: List[Dict[str, Any]]) -> str:
        """Pure Python fallback PDF generator ensuring zero dependency failures"""
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        lines = [
            f"%PDF-1.4",
            f"1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj",
            f"2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj",
            f"3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >> endobj",
            f"5 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> endobj",
        ]

        # Construct stream content
        stream_text = []
        stream_text.append("BT /F1 18 Tf 50 740 Td (Mood Mentor -- Psychological Wellness Executive Report) Tj ET")
        stream_text.append(f"BT /F1 10 Tf 50 720 Td (Generated: {now_str} | User ID: {user_id}) Tj ET")
        stream_text.append("BT /F1 12 Tf 50 685 Td (EXECUTIVE SUMMARY & TELEMETRY KPI) Tj ET")
        stream_text.append(f"BT /F1 10 Tf 50 665 Td (Total Wellness Check-ins: {total_sessions}) Tj ET")
        stream_text.append(f"BT /F1 10 Tf 50 650 Td (Dominant Emotional State: {dominant_emotion.upper()}) Tj ET")
        stream_text.append(f"BT /F1 10 Tf 50 635 Td (Average Emotion Intensity: {avg_intensity:.2f} / 1.00) Tj ET")
        stream_text.append(f"BT /F1 10 Tf 50 620 Td (Critical/High Severity Alerts: {critical_alerts}) Tj ET")

        stream_text.append("BT /F1 12 Tf 50 580 Td (RECENT INTERACTION LOG (Last 5 Sessions)) Tj ET")
        y = 560
        for i, item in enumerate(interactions[:5]):
            ts = item['timestamp'][:16].replace("T", " ")
            txt = (item['raw_text'][:40] + '...') if len(item['raw_text']) > 40 else item['raw_text']
            txt = txt.replace("(", "").replace(")", "")
            line = f"[{ts}] {item['primary_emotion'].upper()} (Intensity: {item['intensity_score']:.2f}, Triage: {item['triage_level']}) - \"{txt}\""
            stream_text.append(f"BT /F1 9 Tf 50 {y} Td ({line}) Tj ET")
            y -= 18

        stream_text.append(f"BT /F1 12 Tf 50 {y-15} Td (TOP CLINICAL RECOMMENDATIONS) Tj ET")
        y -= 35
        for i, r in enumerate(recs[:4]):
            title = r['title'].replace("(", "").replace(")", "")
            stream_text.append(f"BT /F1 9 Tf 50 {y} Td ({i+1}. {title} [{r['modality']}] - Score: {r['score']:.2f}) Tj ET")
            y -= 16

        stream_text.append("BT /F1 8 Tf 50 50 Td (CONFIDENTIAL: For personal wellness insights only. Not a formal psychiatric diagnosis.) Tj ET")

        content = "\n".join(stream_text)
        content_len = len(content.encode('utf-8'))
        lines.append(f"4 0 obj << /Length {content_len} >> stream\n{content}\nendstream\nendobj")
        lines.append("xref\n0 6\n0000000000 65535 f \n0000000010 00000 n \n0000000060 00000 n \n0000000117 00000 n \n0000000450 00000 n \n0000000234 00000 n \ntrailer << /Size 6 /Root 1 0 R >>\nstartxref\n500\n%%EOF")

        with open(output_path, "w", encoding="latin-1") as f:
            f.write("\n".join(lines))

        return output_path

    def _build_reportlab_pdf(self, output_path: str, user_id: str, total_sessions: int,
                             dominant_emotion: str, avg_intensity: float,
                             critical_alerts: int, interactions: List[Dict[str, Any]],
                             recs: List[Dict[str, Any]]) -> str:
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas
        c = canvas.Canvas(output_path, pagesize=letter)
        c.setFont("Helvetica-Bold", 16)
        c.drawString(50, 750, "Mood Mentor — Personal Psychological Wellness Report")
        c.setFont("Helvetica", 10)
        c.drawString(50, 730, f"User ID: {user_id} | Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        c.line(50, 720, 560, 720)

        c.setFont("Helvetica-Bold", 12)
        c.drawString(50, 695, "Executive Summary:")
        c.setFont("Helvetica", 10)
        c.drawString(60, 675, f"• Total Check-in Sessions: {total_sessions}")
        c.drawString(60, 655, f"• Dominant Emotion: {dominant_emotion.upper()}")
        c.drawString(60, 635, f"• Average Emotional Intensity: {avg_intensity:.2f} / 1.00")
        c.drawString(60, 615, f"• High/Critical Triage Events: {critical_alerts}")

        c.setFont("Helvetica-Bold", 12)
        c.drawString(50, 580, "Recent Check-in Timeline:")
        c.setFont("Helvetica", 9)
        y = 560
        for item in interactions[:5]:
            ts = item['timestamp'][:16].replace("T", " ")
            c.drawString(60, y, f"[{ts}] {item['primary_emotion'].upper()} ({item['tier']} Intensity, {item['triage_level']} Triage)")
            y -= 18

        c.setFont("Helvetica-Bold", 12)
        c.drawString(50, y - 10, "Top Evidence-Based Wellness Recommendations:")
        y -= 30
        c.setFont("Helvetica", 9)
        for i, r in enumerate(recs[:4]):
            c.drawString(60, y, f"{i+1}. {r['title']} ({r['modality']}) — Score: {r['score']:.2f}")
            y -= 18

        c.setFont("Helvetica-Oblique", 8)
        c.drawString(50, 40, "Disclaimer: Mood Mentor is a self-care support platform, not a substitute for clinical psychiatric care.")
        c.save()
        return output_path
