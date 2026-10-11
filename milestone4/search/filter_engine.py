
from datetime import date

from milestone4.database.db import get_connection


ALLOWED_FILTERS = {
    "event_type": {
        "prediction_completed",
        "prediction_failed",
        "report_generated",
    },
    "component": {
        "database",
        "emotion_model",
        "security",
        "reporting",
    },
    "status": {"success", "failure"},
}


def search_telemetry(
    event_type=None,
    component=None,
    status=None,
    start_date=None,
    end_date=None,
    limit=100,
):
    """Search telemetry records using optional filters."""

    if isinstance(limit, bool) or not isinstance(limit, int):
        raise ValueError("Limit must be an integer.")

    if not 1 <= limit <= 1000:
        raise ValueError("Limit must be between 1 and 1000.")

    conditions = []
    parameters = []

    for field, value in (
        ("event_type", event_type),
        ("component", component),
        ("status", status),
    ):
        if value is not None:
            if not isinstance(value, str):
                raise ValueError(f"{field} must be a string.")

            if value not in ALLOWED_FILTERS[field]:
                raise ValueError(f"Invalid {field}.")

            conditions.append(f"{field} = ?")
            parameters.append(value)

    parsed_dates = {}

    for field, value in (
        ("start_date", start_date),
        ("end_date", end_date),
    ):
        if value is not None:
            if not isinstance(value, str):
                raise ValueError(f"{field} must use YYYY-MM-DD.")

            try:
                date.fromisoformat(value)
            except ValueError as exc:
                raise ValueError(
                    f"{field} must use a valid YYYY-MM-DD date."
                ) from exc

            if len(value) != 10:
                raise ValueError(f"{field} must use YYYY-MM-DD.")

            parsed_dates[field] = value

    if (
        "start_date" in parsed_dates
        and "end_date" in parsed_dates
        and parsed_dates["start_date"] > parsed_dates["end_date"]
    ):
        raise ValueError("start_date cannot be after end_date.")

    if "start_date" in parsed_dates:
        conditions.append("substr(created_at, 1, 10) >= ?")
        parameters.append(parsed_dates["start_date"])

    if "end_date" in parsed_dates:
        conditions.append("substr(created_at, 1, 10) <= ?")
        parameters.append(parsed_dates["end_date"])

    query = """
        SELECT id, event_type, component, status, created_at
        FROM telemetry
    """

    if conditions:
        query += " WHERE " + " AND ".join(conditions)

    query += " ORDER BY id DESC LIMIT ?"
    parameters.append(limit)

    with get_connection() as connection:
        rows = connection.execute(query, parameters).fetchall()

    return [dict(row) for row in rows]
git status