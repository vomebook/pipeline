import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock
from unittest.mock import patch

import yaml

from scripts import archive_pdf_derivatives as archive


class ArchivePdfDerivativeTests(unittest.TestCase):
    def test_bucket_validation_rejects_source_bucket(self):
        with self.assertRaises(ValueError):
            archive.validate_bucket("vomebook/pdf-pages")
        self.assertEqual(archive.validate_bucket("melsm/pdf-archive"), "melsm/pdf-archive")

    def test_selection_is_stable_and_checkpointed(self):
        manifest = {"version": 1, "files": {
            f"repo\0{letter}.pdf": {
                "status": "ready", "repo": "repo", "path": f"{letter}.pdf",
                "render_manifest": {"path": "x", "sha256": "a" * 64, "bytes": 1},
            } for letter in "abc"}}
        selected = archive.select_books(manifest, limit=1, checkpoint=1)
        self.assertEqual([key for key, _ in selected], ["repo\0b.pdf"])

    def test_checkpoint_listing_uses_the_full_registry(self):
        manifest = {"version": 1, "files": {
            f"repo\0{number:03d}.pdf": {
                "status": "ready", "repo": "repo", "path": f"{number:03d}.pdf",
                "render_manifest": {"path": "x", "sha256": "a" * 64, "bytes": 1},
            } for number in range(250)}}
        all_books = archive.select_books(manifest, limit=0, checkpoint=0)
        self.assertEqual(len(all_books), 250)
        self.assertEqual(list(range(0, (len(all_books) + 99) // 100)), [0, 1, 2])

    def test_jxl_path_and_report_are_deterministic(self):
        self.assertEqual(
            archive.archive_jxl_path("objects/aa/" + "a" * 64 + "/ocr-input/page-000001.png"),
            "objects/aa/" + "a" * 64 + "/pages/page-000001.jxl")

    def test_archive_book_uploads_only_archive_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            png = b"png-data"
            webp = b"webp-data"
            render = {
                "version": 1, "kind": "pdf-render", "source_sha256": "a" * 64,
                "source_revision": "rev", "pages": [{
                    "p": 1,
                    "i": "objects/aa/" + "a" * 64 + "/ocr-input/page-000001.png",
                    "is": hashlib.sha256(png).hexdigest(), "ib": len(png),
                    "w": "objects/aa/" + "a" * 64 + "/pages/page-000001.webp",
                    "ws": hashlib.sha256(webp).hexdigest(), "wb": len(webp),
                }]}
            render_raw = json.dumps(render).encode()
            entry = {"render_manifest": {"path": "render-manifest.json",
                                          "sha256": hashlib.sha256(render_raw).hexdigest(),
                                          "bytes": len(render_raw)}}

            def download(path, expected_sha, expected_bytes, bucket, token):
                return render_raw if path == "render-manifest.json" else png

            with patch.object(archive, "download_object", side_effect=download):
                result = archive.archive_book("repo\0a.pdf", entry, mode="migrate-png",
                                              source_bucket="vomebook/pdf-pages",
                                              archive_bucket="melsm/pdf-archive", token="token",
                                              api=None, distance=1.5, effort=7, output=output)
            self.assertEqual(result["status"], "ready")
            self.assertEqual(result["copy_paths"], [render["pages"][0]["i"]])
            self.assertTrue(result["archive_manifest"].startswith("manifests/"))

    def test_png_checkpoint_uses_xet_copy_and_one_manifest_add(self):
        path = "objects/aa/" + "a" * 64 + "/ocr-input/page-000001.png"
        api = Mock()
        api.get_bucket_paths_info.return_value = [Mock(path=path, xet_hash="xet-hash")]
        result = {"copy_paths": [path], "archive_manifest": "manifests/book.json",
                  "_archive_payload": {"kind": "pdf-derivative-archive"}}
        archive.publish_png_checkpoint(api, Path("."), [result], "vomebook/pdf-pages",
                                       "melsm/pdf-archive", "token")
        self.assertEqual(api.batch_bucket_files.call_count, 2)
        copy_call, add_call = api.batch_bucket_files.call_args_list
        self.assertEqual(copy_call.kwargs["copy"], [("bucket", "vomebook/pdf-pages", "xet-hash", path)])
        self.assertEqual(add_call.kwargs["add"][0][1], "manifests/book.json")

    def test_jxl_mode_does_not_rearchive_png(self):
        workflow = yaml.safe_load(Path(".github/workflows/archive-pdf-derivatives.yml").read_text())
        inputs = workflow[True]["workflow_dispatch"]["inputs"]
        self.assertEqual(inputs["archive_bucket"]["default"], "")
        self.assertFalse(inputs["apply"]["default"])
        self.assertTrue(inputs["all_checkpoints"]["default"] is False)
        self.assertEqual(workflow["jobs"]["archive"]["strategy"]["max-parallel"], 1)
        text = Path("scripts/archive_pdf_derivatives.py").read_text()
        self.assertIn('archived["png_source"]', text)
        self.assertNotIn('archived["png"] = file_meta(png_target, root)', text.split('if mode == "convert-jxl":', 1)[1])


if __name__ == "__main__":
    unittest.main()
