
import re

ALLOWED_EVENTS = {
    "prediction_completed",
    "prediction_failed",
    "report_generated",
}

ALLOWED_COMPONENTS = {
    "database",
    "emotion_model",
    "security",
    "reporting",
}

ALLOWED_STATUSES = {
    "success",
    "failure",
}


def validate_telemetry(event_type, component, status):
    """Validate telemetry values before storing them."""

    values = {
        "event_type": event_type,
        "component": component,
        "status": status,
    }

    for field, value in values.items():
        if not isinstance(value, str):
            raise ValueError(f"{field} must be a string.")

        if not value or len(value) > 50:
            raise ValueError(f"{field} must contain 1-50 characters.")

        if not re.fullmatch(r"[a-z_]+", value):
            raise ValueError(f"Invalid characters in {field}.")

    if event_type not in ALLOWED_EVENTS:
        raise ValueError("Unsupported event type.")

    if component not in ALLOWED_COMPONENTS:
        raise ValueError("Unsupported component.")

    if status not in ALLOWED_STATUSES:
        raise ValueError("Unsupported status.")

    return values
