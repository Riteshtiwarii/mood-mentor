
import unittest

from milestone4.search.filter_engine import search_telemetry


class TestTelemetrySearch(unittest.TestCase):

    def test_filter_by_status(self):
        results = search_telemetry(status="success")
        self.assertTrue(
            all(row["status"] == "success" for row in results)
        )

    def test_filter_by_component(self):
        results = search_telemetry(component="emotion_model")
        self.assertTrue(
            all(row["component"] == "emotion_model" for row in results)
        )

    def test_multiple_filters(self):
        results = search_telemetry(
            event_type="prediction_completed",
            component="emotion_model",
            status="success",
        )
        self.assertTrue(
            all(
                row["event_type"] == "prediction_completed"
                and row["component"] == "emotion_model"
                and row["status"] == "success"
                for row in results
            )
        )

    def test_invalid_status(self):
        with self.assertRaises(ValueError):
            search_telemetry(status="unknown")

    def test_invalid_limit(self):
        with self.assertRaises(ValueError):
            search_telemetry(limit=0)


if __name__ == "__main__":
    unittest.main()
