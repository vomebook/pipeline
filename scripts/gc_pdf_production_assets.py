#!/usr/bin/env python3
"""Garbage-collect stale WebP, OCR, and manifest objects in production.

PNG inputs have their own GC.  This collector only removes objects that are
not reachable from the current render/OCR registries, and refuses apply when
any current book cannot be inspected.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from huggingface_hub import HfApi

try:
    from . import archive_pdf_derivatives as archive
    from . import gc_pdf_archives
except ImportError:
    import archive_pdf_derivatives as archive
    import gc_pdf_archives


PRODUCTION_BUCKET = "vomebook/pdf-pages"
MANIFEST_NAMES = ("page-manifest.json", "render-manifest.json", "ocr-manifest.json")


def candidate_path(path: str) -> bool:
    if not path.startswith("objects/"):
        return False
    if "/pages/" in path and path.endswith((".webp", ".jxl")):
        return True
    if "/ocr/" in path and path.endswith(".json.gz"):
        return True
    return path.endswith(MANIFEST_NAMES)


def object_prefix(path: str) -> str:
    for marker in ("/pages/", "/ocr/", "/ocr-input/"):
        if marker in path:
            return path.split(marker, 1)[0] + "/"
    for name in MANIFEST_NAMES:
        suffix = "/" + name
        if path.endswith(suffix):
            return path[:-len(name)]
    return ""


def add_page_manifest(path: str, live: set[str], source_token: str) -> None:
    live.add(path)
    manifest = gc_pdf_archives.read_json_object(path, PRODUCTION_BUCKET, source_token)
    root = path[:-len("page-manifest.json")]
    page_count = manifest.get("page_count")
    if not isinstance(page_count, int) or page_count < 1:
        raise ValueError(f"invalid page manifest: {path}")
    for number in range(1, page_count + 1):
        live.add(f"{root}pages/page-{number:06d}.webp")
    for page in manifest.get("ocr", []):
        if isinstance(page, dict) and isinstance(page.get("o"), str):
            live.add(page["o"])


def collect_live(manifest: dict, source_token: str) -> tuple[set[str], set[str], list[dict]]:
    live: set[str] = set()
    blocked: set[str] = set()
    results = []

    for registry_name in ("pdf_render_manifest.json", "pdf_ocr_manifest.json"):
        try:
            registry = manifest[registry_name]
        except KeyError:
            raise ValueError(f"missing registry {registry_name}")
        for key, entry in sorted(registry.get("files", {}).items()):
            if not isinstance(entry, dict) or entry.get("status") != "ready":
                continue
            try:
                if registry_name == "pdf_render_manifest.json":
                    metadata = entry.get("render_manifest")
                    if not isinstance(metadata, dict):
                        raise ValueError("missing render manifest metadata")
                    raw = archive.download_object(metadata["path"], metadata["sha256"],
                                                  metadata["bytes"], PRODUCTION_BUCKET, source_token)
                    render = json.loads(raw.decode("utf-8"))
                    live.add(metadata["path"])
                    for page in render.get("pages", []):
                        for field in ("w", "j", "o"):
                            if isinstance(page, dict) and isinstance(page.get(field), str):
                                live.add(page[field])
                    page_metadata = entry.get("page_manifest")
                    if isinstance(page_metadata, dict) and isinstance(page_metadata.get("path"), str):
                        add_page_manifest(page_metadata["path"], live, source_token)
                else:
                    path = entry.get("ocr_manifest")
                    if isinstance(path, dict):
                        path = path.get("path")
                    if not isinstance(path, str):
                        raise ValueError("missing OCR manifest path")
                    live.add(path)
                    ocr = gc_pdf_archives.read_json_object(path, PRODUCTION_BUCKET, source_token)
                    for page in ocr.get("pages", []):
                        if isinstance(page, dict) and isinstance(page.get("o"), str):
                            live.add(page["o"])
                    book_text = ocr.get("book_text")
                    if isinstance(book_text, dict) and isinstance(book_text.get("path"), str):
                        live.add(book_text["path"])
                    page_metadata = ocr.get("page_manifest")
                    if isinstance(page_metadata, dict) and isinstance(page_metadata.get("path"), str):
                        add_page_manifest(page_metadata["path"], live, source_token)
                results.append({"registry": registry_name, "key": key, "status": "verified"})
            except Exception as exc:
                metadata = entry.get("render_manifest") if registry_name == "pdf_render_manifest.json" else None
                path = metadata.get("path", "") if isinstance(metadata, dict) else str(entry.get("ocr_manifest", ""))
                prefix = object_prefix(path)
                if prefix:
                    blocked.add(prefix)
                results.append({"registry": registry_name, "key": key, "status": "blocked",
                                "error": f"{type(exc).__name__}: {exc}"})
    return live, blocked, results


def run(registries: dict, *, apply: bool, ocr_enabled: str,
        source_token: str, api: HfApi) -> dict:
    if apply and ocr_enabled.lower() != "false":
        raise ValueError("production asset GC requires PDF_OCR_ENABLED=false")
    live, blocked, results = collect_live(registries, source_token)
    files = [getattr(item, "path", "") for item in api.list_bucket_tree(
        PRODUCTION_BUCKET, recursive=True, token=source_token)]
    candidates = sorted(path for path in files if path and candidate_path(path))
    deletions = [path for path in candidates if path not in live
                 and not any(path.startswith(prefix) for prefix in blocked)]
    if apply and blocked:
        raise RuntimeError("refusing production asset GC because current assets are blocked")
    if apply:
        for start in range(0, len(deletions), 1000):
            api.batch_bucket_files(PRODUCTION_BUCKET, delete=deletions[start:start + 1000], token=source_token)
    return {"candidates": len(candidates), "live": len(live), "delete": len(deletions),
            "blocked": len(blocked), "applied": apply, "results": results}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets-repo", default="vomebook/Reader-Assets")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("output/pdf-production-gc/report.json"))
    args = parser.parse_args()
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is required")
    api = HfApi(token=token)
    registries = {
        "pdf_render_manifest.json": archive.load_registry(api, args.assets_repo),
        "pdf_ocr_manifest.json": archive.load_registry(api, args.assets_repo, "pdf_ocr_manifest.json"),
    }
    report = run(registries, apply=args.apply, ocr_enabled=os.environ.get("PDF_OCR_ENABLED", "true"),
                 source_token=token, api=api)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
                           encoding="utf-8")
    print(f"candidates={report['candidates']} live={report['live']} delete={report['delete']} "
          f"blocked={report['blocked']} applied={report['applied']}", flush=True)
    return 1 if report["blocked"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
