import unittest
from unittest.mock import Mock

from scripts import gc_pdf_production_assets as gc


class GcPdfProductionAssetTests(unittest.TestCase):
    def test_candidate_paths_exclude_png_inputs(self):
        self.assertTrue(gc.candidate_path("objects/aa/x/pages/page-000001.webp"))
        self.assertTrue(gc.candidate_path("objects/aa/x/ocr/page-000001.json.gz"))
        self.assertTrue(gc.candidate_path("objects/aa/x/page-manifest.json"))
        self.assertFalse(gc.candidate_path("objects/aa/x/ocr-input/page-000001.png"))

    def test_apply_requires_ocr_pause(self):
        with self.assertRaisesRegex(ValueError, "PDF_OCR_ENABLED=false"):
            gc.run({"pdf_render_manifest.json": {"version": 1, "files": {}},
                    "pdf_ocr_manifest.json": {"version": 1, "files": {}}},
                   apply=True, ocr_enabled="true", source_token="token", api=Mock())

    def test_empty_dry_run_does_not_delete(self):
        api = Mock()
        api.list_bucket_tree.return_value = []
        report = gc.run({"pdf_render_manifest.json": {"version": 1, "files": {}},
                         "pdf_ocr_manifest.json": {"version": 1, "files": {}}},
                        apply=False, ocr_enabled="false", source_token="token", api=api)
        self.assertEqual(report["delete"], 0)
        self.assertFalse(report["applied"])
        api.batch_bucket_files.assert_not_called()


if __name__ == "__main__":
    unittest.main()
