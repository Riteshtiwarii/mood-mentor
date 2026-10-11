
from milestone4.database.db import get_connection


def generate_summary():
    """Generate a summary of stored telemetry records."""

    with get_connection() as connection:
        total = connection.execute(
            "SELECT COUNT(*) FROM telemetry"
        ).fetchone()[0]

        by_event = connection.execute(
            """
            SELECT event_type, COUNT(*) AS count
            FROM telemetry
            GROUP BY event_type
            ORDER BY event_type
            """
        ).fetchall()

        by_status = connection.execute(
            """
            SELECT status, COUNT(*) AS count
            FROM telemetry
            GROUP BY status
            ORDER BY status
            """
        ).fetchall()

        daily = connection.execute(
            """
            SELECT substr(created_at, 1, 10) AS date,
                   COUNT(*) AS count
            FROM telemetry
            GROUP BY substr(created_at, 1, 10)
            ORDER BY date DESC
            LIMIT 7
            """
        ).fetchall()

    return {
        "total_records": total,
        "by_event": [dict(row) for row in by_event],
        "by_status": [dict(row) for row in by_status],
        "daily_summary": [dict(row) for row in daily],
    }


def print_summary():
    """Print the telemetry report."""

    report = generate_summary()

    print("\n===== MOODMENTOR TELEMETRY REPORT =====")
    print(f"Total records: {report['total_records']}")

    print("\nRecords by event:")
    for row in report["by_event"]:
        print(f"  {row['event_type']}: {row['count']}")

    print("\nRecords by status:")
    for row in report["by_status"]:
        print(f"  {row['status']}: {row['count']}")

    print("\nDaily summary:")
    for row in report["daily_summary"]:
        print(f"  {row['date']}: {row['count']} records")


if __name__ == "__main__":
    print_summary()

import csv
from pathlib import Path


def export_telemetry_csv(output_path="telemetry_report.csv"):
    """Export telemetry records to a CSV file."""

    from milestone4.database.repository import get_recent_telemetry

    records = get_recent_telemetry(limit=100)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "id",
        "event_type",
        "component",
        "status",
        "created_at",
    ]

    with output.open("w", newline="", encoding="utf-8-sig") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    return str(output.resolve())

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)

from milestone4.database.repository import get_recent_telemetry


def export_telemetry_pdf(output_path="telemetry_report.pdf"):
    """Export recent telemetry records as a PDF report."""

    records = get_recent_telemetry(limit=100)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    document = SimpleDocTemplate(str(output), pagesize=A4)
    styles = getSampleStyleSheet()
    elements = []

    elements.append(
        Paragraph("MoodMentor Telemetry Report", styles["Title"])
    )
    elements.append(Spacer(1, 12))
    elements.append(
        Paragraph(f"Total records exported: {len(records)}", styles["Normal"])
    )
    elements.append(Spacer(1, 12))

    table_data = [
        ["ID", "Event", "Component", "Status", "Created At"]
    ]

    for record in records:
        table_data.append([
            str(record["id"]),
            str(record["event_type"]),
            str(record["component"]),
            str(record["status"]),
            str(record["created_at"]),
        ])

    table = Table(
        table_data,
        repeatRows=1,
        colWidths=[30, 105, 90, 55, 170],
    )
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.darkblue),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONTSIZE", (0, 0), (-1, -1), 7),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
    ]))

    elements.append(table)
    document.build(elements)

    return str(output.resolve())
