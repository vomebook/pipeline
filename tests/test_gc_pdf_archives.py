import unittest
from unittest.mock import Mock

from scripts import gc_pdf_archives as gc


class GcPdfArchivesTests(unittest.TestCase):
    def test_candidate_paths_are_scoped_to_selected_archive_type(self):
        self.assertTrue(gc.candidate_path("png", "objects/aa/x/ocr-input/page-000001.png"))
        self.assertFalse(gc.candidate_path("png", "objects/aa/x/pages/page-000001.jxl"))
        self.assertTrue(gc.candidate_path("jxl", "objects/aa/x/pages/page-000001.jxl"))
        self.assertFalse(gc.candidate_path("jxl", "objects/aa/x/ocr-input/page-000001.png"))
        self.assertTrue(gc.candidate_path("jxl", "manifests/sha/book.json"))

    def test_apply_requires_the_expected_bucket_and_png_input_switch(self):
        with self.assertRaisesRegex(ValueError, "PNG archive GC"):
            gc.run({"version": 1, "files": {}}, mode="png", archive_bucket="melsm/pdf-archive",
                   input_bucket="vomebook/pdf-pages", apply=True, source_token="source",
                   archive_token="archive", api=Mock())

    def test_empty_dry_run_is_non_destructive(self):
        api = Mock()
        api.list_bucket_tree.return_value = []
        report = gc.run({"version": 1, "files": {}}, mode="jxl", archive_bucket="melsm/pdf-jxl",
                        input_bucket="melsm/pdf-archive", apply=False, source_token="source",
                        archive_token="archive", api=api)
        self.assertEqual(report["delete"], 0)
        self.assertFalse(report["applied"])
        api.batch_bucket_files.assert_not_called()


if __name__ == "__main__":
    unittest.main()
