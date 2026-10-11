
import unittest

from milestone4.security.validation import validate_telemetry
from milestone4.reporting.reports import generate_summary


class TestMoodMentorModules(unittest.TestCase):

    def test_valid_telemetry(self):
        result = validate_telemetry(
            "prediction_completed",
            "emotion_model",
            "success",
        )
        self.assertEqual(result["status"], "success")

    def test_invalid_event(self):
        with self.assertRaises(ValueError):
            validate_telemetry(
                "unknown_event",
                "emotion_model",
                "success",
            )

    def test_invalid_status(self):
        with self.assertRaises(ValueError):
            validate_telemetry(
                "prediction_completed",
                "emotion_model",
                "unknown",
            )

    def test_report_structure(self):
        report = generate_summary()
        self.assertIn("total_records", report)
        self.assertIn("by_event", report)
        self.assertIn("by_status", report)
        self.assertIn("daily_summary", report)


if __name__ == "__main__":
    unittest.main()
