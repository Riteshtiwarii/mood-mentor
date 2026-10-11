
from milestone4.database.db import get_connection
from milestone4.security.validation import validate_telemetry


def add_telemetry(event_type, component, status):
    """Validate and safely save a telemetry record."""

    record = validate_telemetry(
        event_type,
        component,
        status,
    )

    with get_connection() as connection:
        cursor = connection.execute(
            """
            INSERT INTO telemetry (event_type, component, status)
            VALUES (?, ?, ?)
            """,
            (
                record["event_type"],
                record["component"],
                record["status"],
            ),
        )

        return cursor.lastrowid


def get_recent_telemetry(limit=20):
    """Retrieve recent telemetry records."""

    if isinstance(limit, bool) or not isinstance(limit, int):
        raise ValueError("Limit must be an integer.")

    if not 1 <= limit <= 100:
        raise ValueError("Limit must be between 1 and 100.")

    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT id, event_type, component, status, created_at
            FROM telemetry
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

    return [dict(row) for row in rows]
