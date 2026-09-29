import unittest

from scripts import gc_pdf_png as gc


class GcPdfPngTests(unittest.TestCase):
    def test_apply_requires_archive_as_ocr_input_bucket(self):
        with self.assertRaisesRegex(ValueError, "OCR input bucket"):
            gc.run({"version": 1, "files": {}}, limit=1, checkpoint=0,
                   archive_bucket="melsm/pdf-archive", source_bucket="vomebook/pdf-pages",
                   input_bucket="vomebook/pdf-pages", apply=True,
                   source_token="source", archive_token="archive")

    def test_empty_dry_run_is_non_destructive(self):
        report = gc.run({"version": 1, "files": {}}, limit=1, checkpoint=0,
                        archive_bucket="melsm/pdf-archive", source_bucket="vomebook/pdf-pages",
                        input_bucket="melsm/pdf-archive", apply=False,
                        source_token="source", archive_token="archive")
        self.assertEqual(report, {"selected": 0, "verified": 0, "pngs": 0,
                                  "eligible": 0, "delete": 0, "limit": 1000,
                                  "applied": False, "results": []})


if __name__ == "__main__":
    unittest.main()
