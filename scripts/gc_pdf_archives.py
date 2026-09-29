#!/usr/bin/env python3
"""Garbage-collect unreferenced PNG or JXL archive objects.

The current render registry is the source of truth.  A book whose archive
manifest is missing or has a different render identity blocks deletion under
that book's object prefix, so a partial migration cannot cause data loss.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from urllib.parse import quote

from huggingface_hub import HfApi

try:
    from . import archive_pdf_derivatives as archive
except ImportError:
    import archive_pdf_derivatives as archive


SOURCE_BUCKET = "vomebook/pdf-pages"
PNG_BUCKET = "melsm/pdf-archive"
JXL_BUCKET = "melsm/pdf-jxl"


def read_json_object(path: str, bucket: str, token: str | None) -> dict:
    response = archive.get_session().get(
        f"https://huggingface.co/buckets/{bucket}/resolve/{quote(path, safe='/')}",
        headers={"Authorization": f"Bearer {token}"} if token else {},
        follow_redirects=True, timeout=180)
    archive.hf_raise_for_status(response)
    value = json.loads(response.content.decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"invalid archive manifest: {path}")
    return value


def candidate_path(mode: str, path: str) -> bool:
    if path.startswith("manifests/") and path.endswith(".json"):
        return True
    if mode == "png":
        return "/ocr-input/" in path and path.endswith(".png")
    return "/pages/" in path and path.endswith(".jxl")


def object_prefix(path: str) -> str:
    marker = "/ocr-input/" if "/ocr-input/" in path else "/pages/"
    return path.split(marker, 1)[0] + "/"


def verify_live_references(manifest: dict, mode: str, archive_bucket: str,
                           source_token: str, archive_token: str) -> tuple[set[str], set[str], list[dict]]:
    live: set[str] = set()
    blocked: set[str] = set()
    results = []
    for key, entry in archive.select_books(manifest, limit=0, checkpoint=0):
        try:
            render_meta = entry["render_manifest"]
            render = json.loads(archive.download_object(
                render_meta["path"], render_meta["sha256"], render_meta["bytes"],
                SOURCE_BUCKET, source_token).decode("utf-8"))
            pages = render.get("pages")
            if render.get("kind") != "pdf-render" or not isinstance(pages, list):
                raise ValueError("invalid render manifest")
            for page in pages:
                if mode == "png" and page.get("i"):
                    live.add(page["i"])
                if mode == "jxl" and page.get("j"):
                    live.add(page["j"])
            source_sha = render["source_sha256"]
            manifest_path = archive.archive_manifest_path(key, source_sha)
            saved = read_json_object(manifest_path, archive_bucket, archive_token)
            if (saved.get("kind") != "pdf-derivative-archive"
                    or saved.get("source_sha256") != source_sha
                    or saved.get("render_manifest") != render_meta):
                raise ValueError("archive manifest identity mismatch")
            live.add(manifest_path)
            for page in saved.get("pages", []):
                field = "png" if mode == "png" else "jxl"
                item = page.get(field) if isinstance(page, dict) else None
                if isinstance(item, dict) and isinstance(item.get("path"), str):
                    live.add(item["path"])
            results.append({"key": key, "status": "verified"})
        except Exception as exc:
            render_path = str(entry.get("render_manifest", {}).get("path", ""))
            if render_path.endswith("/render-manifest.json"):
                blocked.add(render_path[:-len("render-manifest.json")])
            results.append({"key": key, "status": "blocked",
                            "error": f"{type(exc).__name__}: {exc}"})
    return live, blocked, results


def run(manifest: dict, *, mode: str, archive_bucket: str, input_bucket: str,
        apply: bool, source_token: str, archive_token: str, api: HfApi) -> dict:
    if mode not in {"png", "jxl"}:
        raise ValueError("mode must be png or jxl")
    expected_bucket = PNG_BUCKET if mode == "png" else JXL_BUCKET
    archive.validate_bucket(archive_bucket)
    if archive_bucket != expected_bucket:
        raise ValueError(f"{mode} archive bucket must be {expected_bucket}")
    if apply and mode == "png" and input_bucket != archive_bucket:
        raise ValueError("PNG archive GC requires OCR input bucket to be the archive bucket")
    live, blocked, results = verify_live_references(
        manifest, mode, archive_bucket, source_token, archive_token)
    files = [getattr(item, "path", "") for item in api.list_bucket_tree(
        archive_bucket, recursive=True, token=archive_token)]
    candidates = sorted(path for path in files if path and candidate_path(mode, path))
    deletions = [path for path in candidates if path not in live
                 and not any(path.startswith(prefix) for prefix in blocked)]
    if apply and any(item["status"] == "blocked" for item in results):
        raise RuntimeError("refusing archive GC because a current book could not be verified")
    if apply and deletions:
        for start in range(0, len(deletions), 1000):
            api.batch_bucket_files(archive_bucket, delete=deletions[start:start + 1000], token=archive_token)
    return {"mode": mode, "archive_bucket": archive_bucket, "candidates": len(candidates),
            "live": len(live), "delete": len(deletions), "applied": apply,
            "blocked": len(blocked), "results": results}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("png", "jxl"), required=True)
    parser.add_argument("--assets-repo", default="vomebook/Reader-Assets")
    parser.add_argument("--archive-bucket", default="")
    parser.add_argument("--input-bucket", default=os.environ.get("PDF_OCR_INPUT_BUCKET", ""))
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("output/pdf-archive-gc/report.json"))
    args = parser.parse_args()
    source_token = os.environ.get("HF_TOKEN")
    archive_token = os.environ.get("ARCHIVE_HF_TOKEN")
    if not source_token or not archive_token:
        raise RuntimeError("HF_TOKEN and ARCHIVE_HF_TOKEN are required")
    if not args.archive_bucket:
        args.archive_bucket = PNG_BUCKET if args.mode == "png" else JXL_BUCKET
    if not args.input_bucket:
        args.input_bucket = PNG_BUCKET
    api = HfApi(token=archive_token)
    report = run(archive.load_registry(HfApi(token=source_token), args.assets_repo),
                 mode=args.mode, archive_bucket=args.archive_bucket,
                 input_bucket=args.input_bucket, apply=args.apply,
                 source_token=source_token, archive_token=archive_token, api=api)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
                           encoding="utf-8")
    print(f"mode={report['mode']} candidates={report['candidates']} live={report['live']} "
          f"delete={report['delete']} blocked={report['blocked']} applied={report['applied']}", flush=True)
    return 1 if report["blocked"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
