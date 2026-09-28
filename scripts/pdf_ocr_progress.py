"""Bounded OCR progress storage with read-only access to the legacy registry."""

import gzip
import hashlib
import json
import time
from pathlib import Path

import ijson
from huggingface_hub import CommitOperationAdd
from huggingface_hub.errors import HfHubHTTPError

try:
    from . import shared
except ImportError:
    import shared


LEGACY = "pdf_ocr_progress.json"
LEGACY_KEYS = "pdf_ocr_progress_legacy_keys.json.gz"
PROGRESS_PREFIX = "pdf_ocr_progress_v2"


def book_path(key):
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return f"{PROGRESS_PREFIX}/{digest[:2]}/{digest}.json.gz"


def _download(api, repo, name, revision):
    try:
        return api.hf_hub_download(repo_id=repo, repo_type="dataset", filename=name, revision=revision)
    except HfHubHTTPError as exc:
        if shared.hf_status_code(exc) == 404:
            return None
        raise


def load_book(api, repo, key, revision):
    path = _download(api, repo, book_path(key), revision)
    if path is None:
        return None
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        data = json.load(stream)
    if (data.get("version") != 1 or data.get("key") != key
            or not isinstance(data.get("generation"), str) or not isinstance(data.get("pages"), dict)):
        raise ValueError("invalid per-book OCR progress")
    return {"generation": data["generation"], "pages": data["pages"]}


def _legacy_sha(api, repo, revision):
    files = api.get_paths_info(repo_id=repo, repo_type="dataset", revision=revision, paths=[LEGACY])
    if not files:
        return None
    file = files[0]
    if file.path != LEGACY:
        raise ValueError("unexpected OCR progress path")
    sha = getattr(getattr(file, "lfs", None), "sha256", None) or getattr(file, "blob_id", None)
    if not sha:
        raise ValueError("OCR progress identity unavailable")
    return sha


def _load_key_index(api, repo, revision, sha):
    path = _download(api, repo, LEGACY_KEYS, revision)
    if path is None:
        return None
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        data = json.load(stream)
    if (data.get("version") != 1 or not isinstance(data.get("keys"), list)
            or any(not isinstance(key, str) for key in data["keys"])):
        raise ValueError("invalid legacy OCR progress key index")
    return set(data["keys"]) if data.get("legacy_sha256") == sha else None


def _save_key_index(api, repo, sha, keys):
    data = {"version": 1, "legacy_sha256": sha, "keys": sorted(keys)}
    content = gzip.compress(json.dumps(data, ensure_ascii=False, separators=(",", ":")).encode(), mtime=0)
    for attempt in range(10):
        info = api.repo_info(repo_id=repo, repo_type="dataset")
        if _legacy_sha(api, repo, info.sha) != sha:
            return
        try:
            api.create_commit(repo_id=repo, repo_type="dataset", parent_commit=info.sha,
                              commit_message="Index existing OCR progress keys",
                              operations=[CommitOperationAdd(path_in_repo=LEGACY_KEYS, path_or_fileobj=content)])
            return
        except HfHubHTTPError as exc:
            if not shared.is_retryable_hf_status(shared.hf_status_code(exc)) or attempt == 9:
                raise
            time.sleep(shared.hf_retry_delay(attempt))


def load_progress(api, repo, keys, revision):
    keys = set(keys)
    selected = {}
    for key in sorted(keys):
        value = load_book(api, repo, key, revision)
        if value is not None:
            selected[key] = value
    missing = keys - selected.keys()
    if not missing:
        return selected
    sha = _legacy_sha(api, repo, revision)
    if sha is None:
        return selected
    index = _load_key_index(api, repo, revision, sha)
    if index is not None:
        missing &= index
        if not missing:
            return selected
    path = _download(api, repo, LEGACY, revision)
    if path is None:
        raise ValueError("OCR progress disappeared after path lookup")
    found_keys = set()
    with Path(path).open("rb") as stream:
        for key, value in ijson.kvitems(stream, "files", use_float=True):
            if index is None:
                found_keys.add(key)
            if key in missing:
                if not isinstance(value, dict) or not isinstance(value.get("pages"), dict):
                    raise ValueError("invalid legacy OCR book progress")
                selected[key] = value
    if index is None:
        _save_key_index(api, repo, sha, found_keys)
    return selected


def save_progress(api, repo, updates):
    for offset in range(0, len(updates), 10):
        batch = list(updates.items())[offset:offset + 10]
        for attempt in range(20):
            info = api.repo_info(repo_id=repo, repo_type="dataset")
            operations = []
            for key, new in batch:
                old = load_book(api, repo, key, info.sha)
                pages = dict(old["pages"]) if old and old["generation"] == new["generation"] else {}
                pages.update(new["pages"])
                data = {"version": 1, "key": key, "generation": new["generation"], "pages": pages}
                content = gzip.compress(json.dumps(data, ensure_ascii=False, sort_keys=True,
                                                   separators=(",", ":")).encode(), mtime=0)
                operations.append(CommitOperationAdd(path_in_repo=book_path(key), path_or_fileobj=content))
            try:
                api.create_commit(repo_id=repo, repo_type="dataset", parent_commit=info.sha,
                                  commit_message="Save per-book PDF OCR progress", operations=operations)
                break
            except HfHubHTTPError as exc:
                if not shared.is_retryable_hf_status(shared.hf_status_code(exc)) or attempt == 19:
                    raise
                time.sleep(shared.hf_retry_delay(attempt))
