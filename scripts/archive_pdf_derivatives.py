#!/usr/bin/env python3
"""Copy PDF render intermediates to an independent archive bucket.

The archive is deliberately outside the Reader bucket.  This command never
deletes source objects or changes Reader-Assets; it only writes derivative
objects and per-book archive manifests.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from urllib.parse import quote

from huggingface_hub import HfApi, hf_hub_download, sync_bucket
from huggingface_hub.utils import get_session, hf_raise_for_status
from PIL import Image

try:
    from . import pdf_ocr, shared
except ImportError:
    import pdf_ocr, shared


SOURCE_BUCKET = "vomebook/pdf-pages"
DEFAULT_ARCHIVE_BUCKET = "melsm/pdf-archive"
DEFAULT_JXL_BUCKET = "melsm/pdf-jxl"
RENDER_REGISTRY = "pdf_render_manifest.json"
BUCKET_RE = re.compile(r"^[a-z0-9][a-z0-9.-]{0,95}/[a-z0-9][a-z0-9._-]{0,95}$")


def load_registry(api: HfApi, repo: str, name: str = RENDER_REGISTRY) -> dict:
    path = hf_hub_download(repo_id=repo, repo_type="dataset", filename=name,
                            token=os.environ.get("HF_TOKEN"))
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("version") != 1 or not isinstance(data.get("files"), dict):
        raise ValueError(f"invalid {name}")
    return data


def validate_bucket(value: str) -> str:
    if not BUCKET_RE.fullmatch(value) or value == SOURCE_BUCKET:
        raise ValueError("archive bucket must be a different namespace/bucket")
    return value


def select_books(manifest: dict, *, limit: int = 0, checkpoint: int = 0,
                 source_repo: str = "", source_path_prefix: str = "") -> list[tuple[str, dict]]:
    if limit < 0 or checkpoint < 0:
        raise ValueError("limit and checkpoint must be non-negative")
    candidates = []
    for key, entry in sorted(manifest["files"].items()):
        if not isinstance(entry, dict) or entry.get("status") != "ready":
            continue
        if not isinstance(entry.get("render_manifest"), dict):
            continue
        if source_repo and entry.get("repo") != source_repo:
            continue
        if source_path_prefix and not str(entry.get("path", "")).startswith(source_path_prefix):
            continue
        candidates.append((key, entry))
    start = checkpoint * limit if limit else 0
    return candidates[start:start + limit if limit else None]


def download_object(path: str, expected_sha: str, expected_bytes: int,
                    bucket: str, token: str | None) -> bytes:
    pdf_ocr.validate_ocr_object_path(path)
    if not isinstance(expected_sha, str) or len(expected_sha) != 64:
        raise ValueError(f"invalid checksum for {path}")
    if not isinstance(expected_bytes, int) or expected_bytes < 1:
        raise ValueError(f"invalid byte count for {path}")
    url = f"https://huggingface.co/buckets/{bucket}/resolve/{quote(path, safe='/')}"
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    last = None
    for attempt in range(6):
        try:
            response = get_session().get(url, headers=headers, follow_redirects=True, timeout=180)
            hf_raise_for_status(response)
            data = response.content
            if len(data) != expected_bytes or hashlib.sha256(data).hexdigest() != expected_sha:
                raise ValueError(f"object checksum mismatch: {path}")
            return data
        except Exception as exc:
            last = exc
            status = getattr(getattr(exc, "response", None), "status_code", None)
            if attempt == 5 or (status is not None and status not in {408, 429, 500, 502, 503, 504}):
                raise
            time.sleep(min(60, 2 ** attempt))
    raise last or RuntimeError("object download failed")


def page_meta(page: dict, field: str) -> dict:
    return {"path": page[field], "sha256": page[field + "s"], "bytes": page[field + "b"]}


def archive_jxl_path(png_path: str) -> str:
    if "/ocr-input/" not in png_path or not png_path.endswith(".png"):
        raise ValueError(f"cannot derive JXL path from {png_path}")
    return png_path.replace("/ocr-input/", "/pages/")[:-4] + ".jxl"


def archive_manifest_path(key: str, source_sha: str) -> str:
    return f"manifests/{source_sha}/{hashlib.sha256(key.encode()).hexdigest()}.json"


def encode_jxl(png: Path, webp: Path, destination: Path, distance: float, effort: int) -> None:
    with Image.open(png) as source, Image.open(webp) as reference:
        target_size = reference.size
        if source.size != target_size:
            image = source.convert("RGB").resize(target_size, Image.Resampling.LANCZOS)
        else:
            image = source.convert("RGB")
        resized = destination.with_suffix(".png")
        image.save(resized, "PNG")
        image.close()
    destination.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["cjxl", str(resized), str(destination), "-d", str(distance), "-e", str(effort)],
                   check=True, timeout=300)
    resized.unlink(missing_ok=True)
    if not destination.is_file() or destination.stat().st_size < 1:
        raise RuntimeError(f"cjxl produced no output for {destination}")


def file_meta(path: Path, relative_to: Path) -> dict:
    sha, size = shared.hash_file(path)
    return {"path": path.relative_to(relative_to).as_posix(), "sha256": sha, "bytes": size}


def archive_book(key: str, entry: dict, *, mode: str, source_bucket: str,
                archive_bucket: str, token: str, api: HfApi, distance: float,
                effort: int, output: Path) -> dict:
    render_meta = entry["render_manifest"]
    render_raw = download_object(render_meta["path"], render_meta["sha256"], render_meta["bytes"],
                                 source_bucket, token)
    render = json.loads(render_raw.decode("utf-8"))
    if render.get("version") != 1 or render.get("kind") != "pdf-render" or not isinstance(render.get("pages"), list):
        raise ValueError(f"invalid render manifest for {key}")
    with tempfile.TemporaryDirectory(dir=output) as temporary:
        root = Path(temporary)
        pages = []
        includes = []
        for page in render["pages"]:
            if "i" not in page:
                continue
            png = page_meta(page, "i")
            png_data = download_object(png["path"], png["sha256"], png["bytes"], source_bucket, token)
            png_target = root / png["path"]
            png_target.parent.mkdir(parents=True, exist_ok=True)
            png_target.write_bytes(png_data)
            archived = {"p": page["p"]}
            if mode == "migrate-png":
                includes.append(png["path"])
                archived["png"] = file_meta(png_target, root)
            else:
                archived["png_source"] = {**png, "bucket": source_bucket}
            if mode == "convert-jxl":
                webp = page_meta(page, "w")
                webp_target = root / webp["path"]
                webp_target.parent.mkdir(parents=True, exist_ok=True)
                webp_target.write_bytes(download_object(webp["path"], webp["sha256"], webp["bytes"], source_bucket, token))
                jxl_path = page.get("j") or archive_jxl_path(png["path"])
                jxl_target = root / jxl_path
                encode_jxl(png_target, webp_target, jxl_target, distance, effort)
                archived["jxl"] = file_meta(jxl_target, root)
                includes.append(jxl_path)
                webp_target.unlink(missing_ok=True)
            pages.append(archived)
        if not pages:
            return {"key": key, "status": "skipped", "reason": "no PNG render pages"}
        archive_path = archive_manifest_path(key, render["source_sha256"])
        archive_target = root / archive_path
        archive_target.parent.mkdir(parents=True, exist_ok=True)
        archive_target.write_text(json.dumps({
            "version": 1, "kind": "pdf-derivative-archive", "mode": mode,
            "source_sha256": render["source_sha256"], "source_revision": render.get("source_revision", ""),
            "render_manifest": render_meta, "pages": pages,
        }, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        includes.append(archive_path)
        sync_bucket(str(root), f"hf://buckets/{archive_bucket}", include=sorted(set(includes)),
                    token=token, quiet=True)
        return {"key": key, "status": "ready", "mode": mode, "archive_manifest": archive_path,
                "pages": len(pages), "files": len(includes)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("migrate-png", "convert-jxl"), required=True)
    parser.add_argument("--assets-repo", default="vomebook/Reader-Assets")
    parser.add_argument("--source-bucket", default=SOURCE_BUCKET)
    parser.add_argument("--archive-bucket", default=os.environ.get("ARCHIVE_BUCKET", ""))
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--checkpoint", type=int, default=0)
    parser.add_argument("--list-checkpoints", action="store_true")
    parser.add_argument("--source-repo", default="")
    parser.add_argument("--source-path-prefix", default="")
    parser.add_argument("--distance", type=float, default=1.5)
    parser.add_argument("--effort", type=int, default=7)
    parser.add_argument("--output", type=Path, default=Path("output/pdf-derivatives"))
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    if not args.archive_bucket:
        args.archive_bucket = DEFAULT_JXL_BUCKET if args.mode == "convert-jxl" else DEFAULT_ARCHIVE_BUCKET
    validate_bucket(args.archive_bucket)
    if not os.environ.get("HF_TOKEN"):
        raise RuntimeError("HF_TOKEN is required")
    api = HfApi(token=os.environ["HF_TOKEN"])
    selected = select_books(load_registry(api, args.assets_repo), limit=args.limit,
                            checkpoint=args.checkpoint, source_repo=args.source_repo,
                            source_path_prefix=args.source_path_prefix)
    if args.list_checkpoints:
        count = (len(selected) + args.limit - 1) // args.limit if args.limit else 1
        print(json.dumps(list(range(args.checkpoint, args.checkpoint + count)), separators=(",", ":")))
        return 0
    report = {"mode": args.mode, "archive_bucket": args.archive_bucket,
              "selected": len(selected), "applied": args.apply, "results": []}
    args.output.mkdir(parents=True, exist_ok=True)
    if args.apply:
        for key, entry in selected:
            try:
                report["results"].append(archive_book(
                    key, entry, mode=args.mode, source_bucket=args.source_bucket,
                    archive_bucket=args.archive_bucket, token=os.environ["HF_TOKEN"], api=api,
                    distance=args.distance, effort=args.effort, output=args.output))
            except Exception as exc:
                report["results"].append({"key": key, "status": "failed",
                                          "error": f"{type(exc).__name__}: {exc}"})
    report_path = args.output / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
                           encoding="utf-8")
    print(f"selected={len(selected)} applied={args.apply} "
          f"ready={sum(x.get('status') == 'ready' for x in report['results'])} "
          f"failed={sum(x.get('status') == 'failed' for x in report['results'])}", flush=True)
    return 1 if any(x.get("status") == "failed" for x in report["results"]) else 0


if __name__ == "__main__":
    raise SystemExit(main())
