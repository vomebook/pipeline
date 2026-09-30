#!/usr/bin/env python3
"""Build a conservative page-level PDF OCR review report.

The audit reads immutable OCR page JSON and never changes OCR metadata or
deletes input PNGs.  A flagged page should retain its PNG for a later retry or
manual inspection; an unflagged page is only a candidate for a future retention
policy, not an immediate deletion decision.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

try:
    from . import pdf_ocr, pdf_ocr_stages
except ImportError:
    import pdf_ocr
    import pdf_ocr_stages


REPORT_VERSION = 1
HIGH_SIGNAL_LAYOUT_FLAGS = frozenset({
    "vertical-recognition-needs-sample-validation",
    "within-block-character-order-unverified",
    "blocks-outside-explicit-regions",
})


def non_whitespace_chars(value: str) -> int:
    return len(re.sub(r"\s+", "", str(value or "")))


def block_box(block: dict) -> tuple[float, float, float, float] | None:
    box = block.get("b")
    if not isinstance(box, list) or len(box) != 4:
        return None
    try:
        x0, y0, x1, y1 = (float(value) for value in box)
    except (TypeError, ValueError):
        return None
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1


def box_overlap_ratio(first: tuple[float, float, float, float],
                      second: tuple[float, float, float, float]) -> float:
    x0 = max(first[0], second[0])
    y0 = max(first[1], second[1])
    x1 = min(first[2], second[2])
    y1 = min(first[3], second[3])
    intersection = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    if not intersection:
        return 0.0
    first_area = (first[2] - first[0]) * (first[3] - first[1])
    second_area = (second[2] - second[0]) * (second[3] - second[1])
    return intersection / max(1e-9, min(first_area, second_area))


def overlapping_box_count(blocks: list[dict], minimum_ratio: float) -> int:
    """Count substantial overlaps with a y-sweep instead of O(n^2) pages."""
    boxes = []
    for block in blocks:
        box = block_box(block)
        if box:
            boxes.append(box)
    boxes.sort(key=lambda box: (box[1], box[0]))
    active: list[tuple[float, float, float, float]] = []
    overlaps = 0
    for box in boxes:
        active = [other for other in active if other[3] > box[1]]
        overlaps += sum(box_overlap_ratio(box, other) >= minimum_ratio for other in active)
        active.append(box)
    return overlaps


def page_audit(payload: dict, *, mean_confidence: float = 0.80,
               low_confidence: float = 0.65, low_confidence_ratio: float = 0.35,
               min_chars: int = 8, overlap_ratio: float = 0.50) -> dict:
    """Score one OCR page and return a stable, JSON-serializable summary."""
    raw_blocks = payload.get("raw_blocks")
    blocks = raw_blocks if isinstance(raw_blocks, list) else payload.get("blocks", [])
    blocks = [block for block in blocks if isinstance(block, dict) and block.get("s", "ocr") == "ocr"]
    confidences = []
    for block in blocks:
        try:
            value = float(block.get("c", 0.0))
        except (TypeError, ValueError):
            value = 0.0
        confidences.append(max(0.0, min(1.0, value)))
    text_chars = non_whitespace_chars(payload.get("text", ""))
    mean = sum(confidences) / len(confidences) if confidences else 0.0
    low_count = sum(value < low_confidence for value in confidences)
    low_ratio = low_count / len(confidences) if confidences else 1.0
    minimum = min(confidences) if confidences else 0.0
    layout = payload.get("layout") if isinstance(payload.get("layout"), dict) else {}
    layout_flags = [str(flag) for flag in layout.get("review", []) if isinstance(flag, str)]
    high_signal_flags = sorted(set(layout_flags) & HIGH_SIGNAL_LAYOUT_FLAGS)
    overlap_count = overlapping_box_count(blocks, overlap_ratio)

    reasons = []
    penalties = 0.0
    if not blocks or text_chars == 0:
        reasons.append("empty-ocr-page")
        penalties += 0.45
    if mean < mean_confidence:
        reasons.append("low-mean-confidence")
        penalties += 0.25
    if low_ratio >= low_confidence_ratio:
        reasons.append("many-low-confidence-blocks")
        penalties += 0.20
    if blocks and minimum < low_confidence / 2:
        reasons.append("very-low-confidence-block")
        penalties += 0.10
    if blocks and text_chars < min_chars:
        reasons.append("sparse-ocr-text")
        penalties += 0.15
    if overlap_count:
        reasons.append("overlapping-text-boxes")
        penalties += min(0.25, 0.05 * overlap_count)
    for flag in high_signal_flags:
        reasons.append(f"layout:{flag}")
    score = max(0.0, min(1.0, 1.0 - penalties))
    return {
        "page": payload.get("page"),
        "score": round(score, 4),
        "flagged": bool(reasons),
        "reasons": reasons,
        "chars": text_chars,
        "blocks": len(blocks),
        "mean_confidence": round(mean, 4),
        "min_confidence": round(minimum, 4),
        "low_confidence_ratio": round(low_ratio, 4),
        "overlap_count": overlap_count,
        "layout_flags": layout_flags,
        "keep_png": bool(reasons),
    }


def audit_book(entry: dict, manifest: dict, page_loader, **thresholds) -> tuple[dict, list[dict]]:
    pages = manifest.get("pages")
    if not isinstance(pages, list):
        raise ValueError("OCR manifest pages must be a list")
    page_reports = []
    for descriptor in pages:
        if not isinstance(descriptor, dict) or "o" not in descriptor:
            raise ValueError("OCR manifest page is missing its OCR object")
        if descriptor.get("source") != "ocr":
            continue
        payload = page_loader(descriptor)
        report = page_audit(payload, **thresholds)
        report.update({"key": entry["key"], "source": entry.get("source", "")})
        page_reports.append(report)
    flagged = [page for page in page_reports if page["flagged"]]
    summary = {
        "key": entry["key"],
        "repo": entry.get("repo", ""),
        "path": entry.get("path", ""),
        "page_count": len(page_reports),
        "audited_pages": len(page_reports),
        "flagged_pages": len(flagged),
        "keep_png_pages": len(flagged),
        "min_score": min((page["score"] for page in page_reports), default=1.0),
        "mean_confidence": round(
            sum(page["mean_confidence"] for page in page_reports) / len(page_reports), 4
        ) if page_reports else 0.0,
    }
    return summary, flagged


def select_entries(manifest: dict, *, source_repo: str = "", source_path_prefix: str = "",
                   lane_index: int | None = None, lane_count: int = 4,
                   limit: int = 100, keys: set[str] | None = None) -> list[dict]:
    if limit < 1:
        raise ValueError("limit must be positive")
    if lane_index is not None and not 0 <= lane_index < lane_count:
        raise ValueError("lane index must be within lane count")
    selected = []
    for key, entry in sorted((manifest.get("files") or {}).items()):
        if keys is not None and key not in keys:
            continue
        if not isinstance(entry, dict) or entry.get("status") != "ready":
            continue
        if source_repo and entry.get("repo") != source_repo:
            continue
        if source_path_prefix and not str(entry.get("path", "")).startswith(source_path_prefix):
            continue
        if lane_index is not None and pdf_ocr_stages.ocr_lane_index(key, lane_count) != lane_index:
            continue
        selected.append({"key": key, **entry})
        if len(selected) >= limit:
            break
    return selected


def load_manifest(api, repo: str) -> tuple[str, dict]:
    info = pdf_ocr_stages.retry(lambda: api.repo_info(repo_id=repo, repo_type="dataset"))
    path = pdf_ocr_stages.retry(lambda: api.hf_hub_download(
        repo_id=repo, repo_type="dataset", filename="pdf_ocr_manifest.json", revision=info.sha))
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("version") != 1 or not isinstance(data.get("files"), dict):
        raise ValueError("invalid PDF OCR manifest")
    return info.sha, data


def load_book_manifest(entry: dict) -> dict:
    data = pdf_ocr_stages.read_object({
        "path": entry["ocr_manifest"],
        "sha256": entry["ocr_manifest_sha256"],
        "bytes": entry["ocr_manifest_bytes"],
    }, "/ocr-manifest.json")
    manifest = json.loads(data.decode("utf-8"))
    if manifest.get("kind") != "pdf-ocr" or manifest.get("complete") is not True:
        raise ValueError("incomplete OCR book manifest")
    return manifest


def load_page(entry: dict) -> dict:
    data = pdf_ocr_stages.read_object({
        "path": entry["o"], "sha256": entry["os"], "bytes": entry["ob"],
    }, ".json.gz")
    return json.loads(gzip.decompress(data))


def report_summary(report: dict, max_pages: int = 20) -> str:
    summary = report["summary"]
    lines = [
        "## PDF OCR Review",
        "",
        f"- Books audited: `{summary['books']}`",
        f"- Pages audited: `{summary['pages']}`",
        f"- Flagged pages: `{summary['flagged_pages']}`",
        f"- PNGs to retain: `{summary['keep_png_pages']}`",
        "",
        "### Review Pages",
        "",
    ]
    pages = sorted(report["pages"], key=lambda page: (page["score"], page["key"], page["page"]))
    if not pages:
        lines.append("No pages crossed the review thresholds.")
    else:
        lines.append("| Score | Page | Reason | Source |")
        lines.append("| ---: | ---: | --- | --- |")
        for page in pages[:max_pages]:
            reasons = ", ".join(page["reasons"])
            lines.append(f"| {page['score']:.4f} | {page['page']} | {reasons} | `{page['key']}` |")
        if len(pages) > max_pages:
            lines.append(f"\n_Showing {max_pages} of {len(pages)} flagged pages; see the artifact for all pages._")
    return "\n".join(lines) + "\n"


def build_report(api, *, assets_repo: str, source_repo: str = "", source_path_prefix: str = "",
                 lane_index: int | None = None, limit: int = 100,
                 keys: set[str] | None = None, **thresholds) -> dict:
    revision, registry = load_manifest(api, assets_repo)
    entries = select_entries(registry, source_repo=source_repo,
                             source_path_prefix=source_path_prefix,
                             lane_index=lane_index, limit=limit, keys=keys)
    books, pages = [], []
    for entry in entries:
        manifest = load_book_manifest(entry)
        book, flagged = audit_book(entry, manifest, load_page, **thresholds)
        books.append(book)
        pages.extend(flagged)
    return {
        "version": REPORT_VERSION,
        "kind": "pdf-ocr-review",
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "assets_repo": assets_repo,
        "assets_revision": revision,
        "thresholds": thresholds,
        "summary": {
            "books": len(books),
            "pages": sum(book["audited_pages"] for book in books),
            "flagged_pages": len(pages),
            "keep_png_pages": sum(book["keep_png_pages"] for book in books),
        },
        "books": books,
        "pages": pages,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets-repo", default="vomebook/Reader-Assets")
    parser.add_argument("--source-repo", default="")
    parser.add_argument("--source-path-prefix", default="")
    parser.add_argument("--lane-index", type=int)
    parser.add_argument("--lane-count", type=int, default=4)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--queue-file", type=Path)
    parser.add_argument("--mean-confidence", type=float, default=0.80)
    parser.add_argument("--low-confidence", type=float, default=0.65)
    parser.add_argument("--low-confidence-ratio", type=float, default=0.35)
    parser.add_argument("--min-chars", type=int, default=8)
    parser.add_argument("--overlap-ratio", type=float, default=0.50)
    parser.add_argument("--output", type=Path, default=Path("output/pdf-ocr-review/report.json"))
    parser.add_argument("--summary", type=Path, default=Path("output/pdf-ocr-review/summary.md"))
    args = parser.parse_args()
    if not os.environ.get("HF_TOKEN"):
        raise RuntimeError("HF_TOKEN is required")
    thresholds = {
        "mean_confidence": args.mean_confidence,
        "low_confidence": args.low_confidence,
        "low_confidence_ratio": args.low_confidence_ratio,
        "min_chars": args.min_chars,
        "overlap_ratio": args.overlap_ratio,
    }
    from huggingface_hub import HfApi
    keys = None
    if args.queue_file:
        queue = json.loads(args.queue_file.read_text(encoding="utf-8"))
        if queue.get("kind") != "pdf-image-ocr-queue" or not isinstance(queue.get("books"), list):
            raise ValueError("invalid PDF OCR queue artifact")
        keys = {book["key"] for book in queue["books"]
                if isinstance(book, dict) and isinstance(book.get("key"), str)}
    report = build_report(HfApi(token=os.environ["HF_TOKEN"]),
                          assets_repo=args.assets_repo, source_repo=args.source_repo,
                          source_path_prefix=args.source_path_prefix,
                          lane_index=args.lane_index, lane_count=args.lane_count,
                          limit=args.limit, keys=keys, **thresholds)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
                           encoding="utf-8")
    args.summary.write_text(report_summary(report), encoding="utf-8")
    summary = report["summary"]
    print(f"audited books={summary['books']} pages={summary['pages']} "
          f"flagged={summary['flagged_pages']} keep_png={summary['keep_png_pages']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
