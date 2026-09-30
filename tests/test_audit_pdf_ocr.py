import json
import tempfile
import unittest
from pathlib import Path

import yaml

from scripts import audit_pdf_ocr as audit


def block(text, confidence, box):
    return {"t": text, "c": confidence, "b": box, "s": "ocr"}


class OcrAuditTests(unittest.TestCase):
    def test_clean_page_is_not_flagged_for_assumed_horizontal_direction(self):
        result = audit.page_audit({
            "page": 1,
            "text": "这是正常的中文扫描页文本内容",
            "blocks": [block("这是正常的中文扫描页文本内容", 0.96, [0.1, 0.1, 0.9, 0.2])],
            "layout": {"review": ["reading-direction-assumed-ltr"]},
        })
        self.assertFalse(result["flagged"])
        self.assertEqual(result["reasons"], [])

    def test_low_confidence_and_empty_pages_are_flagged_and_keep_png(self):
        low = audit.page_audit({
            "page": 2,
            "text": "疑难页面文字",
            "blocks": [block("疑难页面文字", 0.30, [0.1, 0.1, 0.9, 0.2])],
            "layout": {"review": []},
        })
        empty = audit.page_audit({"page": 3, "text": "", "blocks": [], "layout": {"review": []}})
        self.assertIn("low-mean-confidence", low["reasons"])
        self.assertTrue(low["keep_png"])
        self.assertIn("empty-ocr-page", empty["reasons"])
        self.assertTrue(empty["keep_png"])

    def test_overlap_and_high_signal_layout_flags_are_reported(self):
        result = audit.page_audit({
            "page": 4,
            "text": "两段文字重叠",
            "raw_blocks": [
                block("两段文字", 0.95, [0.1, 0.1, 0.7, 0.3]),
                block("重叠", 0.92, [0.5, 0.15, 0.8, 0.25]),
            ],
            "layout": {"review": ["within-block-character-order-unverified"]},
        })
        self.assertIn("overlapping-text-boxes", result["reasons"])
        self.assertIn("layout:within-block-character-order-unverified", result["reasons"])

    def test_book_audit_skips_native_text_pages(self):
        entry = {"key": "repo\0book.pdf", "repo": "repo", "path": "book.pdf"}
        manifest = {"pages": [
            {"p": 1, "source": "native", "o": "native"},
            {"p": 2, "source": "ocr", "o": "ocr"},
        ]}
        summary, flagged = audit.audit_book(
            entry, manifest,
            lambda _descriptor: {"page": 2, "text": "", "blocks": [], "layout": {}},
        )
        self.assertEqual(summary["audited_pages"], 1)
        self.assertEqual([page["page"] for page in flagged], [2])

    def test_select_entries_can_be_limited_to_queue_keys(self):
        manifest = {"files": {
            "repo\0a.pdf": {"status": "ready", "repo": "repo", "path": "a.pdf"},
            "repo\0b.pdf": {"status": "ready", "repo": "repo", "path": "b.pdf"},
            "repo\0c.pdf": {"status": "failed", "repo": "repo", "path": "c.pdf"},
        }}
        selected = audit.select_entries(manifest, keys={"repo\0b.pdf"})
        self.assertEqual([entry["key"] for entry in selected], ["repo\0b.pdf"])

    def test_workflow_audits_triggering_queue_and_uploads_report(self):
        workflow = yaml.load(
            (Path(__file__).resolve().parents[1] / ".github/workflows/audit-pdf-ocr.yml").read_text(),
            Loader=yaml.BaseLoader,
        )
        steps = workflow["jobs"]["audit"]["steps"]
        download = next(step for step in steps if step.get("uses") == "actions/download-artifact@v8")
        self.assertEqual(download["with"]["name"], "pdf-image-ocr-queue")
        self.assertEqual(download["with"]["run-id"], "${{ github.event.workflow_run.id }}")
        upload = next(step for step in steps if step.get("uses") == "actions/upload-artifact@v7")
        self.assertIn("output/pdf-ocr-review", upload["with"]["path"])

    def test_report_summary_handles_empty_review_set(self):
        report = {"summary": {"books": 1, "pages": 12, "flagged_pages": 0, "keep_png_pages": 0},
                  "pages": []}
        self.assertIn("No pages crossed the review thresholds.", audit.report_summary(report))


if __name__ == "__main__":
    unittest.main()
