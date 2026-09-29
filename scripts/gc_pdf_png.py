#!/usr/bin/env python3
"""Safely remove PNG OCR inputs from the production PDF bucket.

Deletion is allowed only after every selected book has a matching archive
manifest and every archived PNG passes a checksum readback.  The OCR input
bucket must already point at the archive bucket before --apply is accepted.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from huggingface_hub import HfApi

try:
    from . import archive_pdf_derivatives as archive
except ImportError:
    import archive_pdf_derivatives as archive


SOURCE_BUCKET = "vomebook/pdf-pages"
PNG_ARCHIVE_BUCKET = "melsm/pdf-archive"


def read_json_object(path: str, bucket: str, token: str | None) -> dict:
    url = f"https://huggingface.co/buckets/{bucket}/resolve/{path}"
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    response = archive.get_session().get(url, headers=headers, follow_redirects=True, timeout=180)
    archive.hf_raise_for_status(response)
    data = json.loads(response.content.decode("utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"invalid archive manifest: {path}")
    return data


def select_books(manifest: dict, limit: int, checkpoint: int) -> list[tuple[str, dict]]:
    return archive.select_books(manifest, limit=limit, checkpoint=checkpoint)


def verify_book(key: str, entry: dict, archive_bucket: str, archive_token: str) -> list[str]:
    render_meta = entry["render_manifest"]
    render_raw = archive.download_object(render_meta["path"], render_meta["sha256"],
                                         render_meta["bytes"], SOURCE_BUCKET, archive_token)
    render = json.loads(render_raw.decode("utf-8"))
    if render.get("kind") != "pdf-render" or not isinstance(render.get("pages"), list):
        raise ValueError(f"invalid render manifest for {key}")
    archive_path = archive.archive_manifest_path(key, render["source_sha256"])
    saved = read_json_object(archive_path, archive_bucket, archive_token)
    if (saved.get("kind") != "pdf-derivative-archive" or saved.get("mode") != "migrate-png"
            or saved.get("source_sha256") != render.get("source_sha256")
            or saved.get("render_manifest") != render_meta):
        raise ValueError(f"archive identity mismatch for {key}")
    archived = {int(page["p"]): page for page in saved.get("pages", [])
                if isinstance(page, dict) and isinstance(page.get("p"), int)}
    source_paths = []
    for page in render["pages"]:
        if "i" not in page:
            continue
        saved_page = archived.get(page["p"])
        png = saved_page.get("png") if saved_page else None
        if not isinstance(png, dict) or png.get("path") != page["i"]:
            raise ValueError(f"missing archived PNG for {key} page {page['p']}")
        archive.download_object(png["path"], png["sha256"], png["bytes"], archive_bucket, archive_token)
        source_paths.append(page["i"])
    return source_paths


def run(manifest: dict, *, limit: int, checkpoint: int, archive_bucket: str,
        source_bucket: str, input_bucket: str, apply: bool, source_token: str,
        archive_token: str) -> dict:
    if source_bucket != SOURCE_BUCKET:
        raise ValueError("source bucket is fixed to vomebook/pdf-pages")
    archive.validate_bucket(archive_bucket)
    if input_bucket != archive_bucket:
        raise ValueError("OCR input bucket must equal the PNG archive bucket")
    selected = select_books(manifest, limit, checkpoint)
    paths = []
    results = []
    for key, entry in selected:
        try:
            verified = verify_book(key, entry, archive_bucket, archive_token)
            paths.extend(verified)
            results.append({"key": key, "status": "verified", "pngs": len(verified)})
        except Exception as exc:
            results.append({"key": key, "status": "failed",
                            "error": f"{type(exc).__name__}: {exc}"})
    failed = [item for item in results if item["status"] == "failed"]
    if apply and failed:
        raise RuntimeError("refusing PNG deletion because archive verification failed")
    if apply:
        api = HfApi(token=source_token)
        for start in range(0, len(paths), 1000):
            api.batch_bucket_files(source_bucket, delete=paths[start:start + 1000], token=source_token)
    return {"selected": len(selected), "verified": len(results) - len(failed),
            "pngs": len(paths), "applied": apply, "results": results}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets-repo", default="vomebook/Reader-Assets")
    parser.add_argument("--source-bucket", default=SOURCE_BUCKET)
    parser.add_argument("--archive-bucket", default=PNG_ARCHIVE_BUCKET)
    parser.add_argument("--input-bucket", default=os.environ.get("PDF_OCR_INPUT_BUCKET", ""))
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--checkpoint", type=int, default=0)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("output/pdf-png-gc/report.json"))
    args = parser.parse_args()
    source_token = os.environ.get("HF_TOKEN")
    archive_token = os.environ.get("ARCHIVE_HF_TOKEN")
    if not source_token or not archive_token:
        raise RuntimeError("HF_TOKEN and ARCHIVE_HF_TOKEN are required")
    if not args.input_bucket:
        args.input_bucket = PNG_ARCHIVE_BUCKET
    report = run(archive.load_registry(HfApi(token=archive_token), args.assets_repo),
                 limit=args.limit, checkpoint=args.checkpoint, archive_bucket=args.archive_bucket,
                 source_bucket=args.source_bucket, input_bucket=args.input_bucket, apply=args.apply,
                 source_token=source_token, archive_token=archive_token)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
                           encoding="utf-8")
    print(f"selected={report['selected']} verified={report['verified']} pngs={report['pngs']} "
          f"applied={report['applied']}", flush=True)
    return 1 if any(item["status"] == "failed" for item in report["results"]) else 0


if __name__ == "__main__":
    raise SystemExit(main())
