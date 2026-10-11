
import tempfile
import unittest
import csv
from pathlib import Path

from milestone4.reporting.reports import (
    export_telemetry_csv,
    export_telemetry_pdf,
)


class TestReportExports(unittest.TestCase):

    def test_csv_export(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "report.csv"
            result = export_telemetry_csv(str(output))

            self.assertTrue(Path(result).exists())

            with output.open(
                newline="", encoding="utf-8-sig"
            ) as file:
                rows = list(csv.DictReader(file))

            self.assertIn("event_type", rows[0] if rows else {
                "event_type": ""
            })

    def test_pdf_export(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "report.pdf"
            result = export_telemetry_pdf(str(output))

            self.assertTrue(Path(result).exists())
            self.assertGreater(Path(result).stat().st_size, 0)

            with output.open("rb") as file:
                self.assertTrue(file.read(5).startswith(b"%PDF-"))


if __name__ == "__main__":
    unittest.main()
