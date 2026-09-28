"""Compact, resumable OCR progress split by book and page range."""

import gzip
import hashlib
import json
import time
from pathlib import Path

from huggingface_hub import CommitOperationAdd
from huggingface_hub.errors import HfHubHTTPError

try:
    from . import shared
except ImportError:
    import shared


PROGRESS_PREFIX = "pdf_ocr_progress_v3"
PAGES_PER_CHUNK = 500
COMPACT_FIELDS = ("p", "source", "i", "is", "ib", "o", "os", "ob")


def _digest(key):
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def book_dir(key):
    digest = _digest(key)
    return f"{PROGRESS_PREFIX}/{digest[:2]}/{digest}"


def book_path(key):
    return f"{book_dir(key)}/index.json"


def chunk_path(key, start):
    return f"{book_dir(key)}/{start:08d}-{start + PAGES_PER_CHUNK - 1:08d}.json.gz"


def _download(api, repo, name, revision):
    try:
        return api.hf_hub_download(repo_id=repo, repo_type="dataset", filename=name, revision=revision)
    except HfHubHTTPError as exc:
        if shared.hf_status_code(exc) == 404:
            return None
        raise


def compact_page(page):
    return {field: page[field] for field in COMPACT_FIELDS if field in page}


def _load_json_gzip(path):
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return json.load(stream)


def _load_v3(api, repo, key, revision):
    index_path = _download(api, repo, book_path(key), revision)
    if index_path is None:
        return None
    index = json.loads(Path(index_path).read_text(encoding="utf-8"))
    if (index.get("version") != 1 or index.get("key") != key
            or not isinstance(index.get("generation"), str)
            or not isinstance(index.get("chunks"), list)):
        raise ValueError("invalid v3 OCR progress index")
    pages = {}
    for chunk in index["chunks"]:
        if (not isinstance(chunk, dict) or not isinstance(chunk.get("path"), str)
                or not isinstance(chunk.get("start"), int) or not isinstance(chunk.get("end"), int)):
            raise ValueError("invalid v3 OCR progress chunk")
        path = _download(api, repo, chunk["path"], revision)
        if path is None:
            raise ValueError("OCR progress chunk disappeared")
        data = _load_json_gzip(path)
        if (data.get("version") != 1 or data.get("key") != key
                or data.get("generation") != index["generation"]
                or not isinstance(data.get("pages"), dict)):
            raise ValueError("invalid v3 OCR progress chunk")
        pages.update(data["pages"])
    return {"generation": index["generation"], "pages": pages}


def load_book(api, repo, key, revision):
    return _load_v3(api, repo, key, revision)


def load_progress(api, repo, keys, revision):
    return {key: value for key in sorted(set(keys))
            if (value := load_book(api, repo, key, revision)) is not None}


def _commit(api, repo, operations, message):
    for attempt in range(20):
        info = api.repo_info(repo_id=repo, repo_type="dataset")
        try:
            api.create_commit(repo_id=repo, repo_type="dataset", parent_commit=info.sha,
                              commit_message=message, operations=operations)
            return
        except HfHubHTTPError as exc:
            if not shared.is_retryable_hf_status(shared.hf_status_code(exc)) or attempt == 19:
                raise
            time.sleep(shared.hf_retry_delay(attempt))


def save_progress(api, repo, updates):
    for offset in range(0, len(updates), 10):
        operations = []
        for key, new in list(updates.items())[offset:offset + 10]:
            old = _load_v3(api, repo, key, None)
            if old is None:
                old = _load_v3(api, repo, key, None) or {"generation": new["generation"], "pages": {}}
            pages = dict(old["pages"]) if old["generation"] == new["generation"] else {}
            pages.update({page: compact_page(value) for page, value in new["pages"].items()})
            grouped = {}
            for page, value in pages.items():
                number = int(page)
                start = number - (number % PAGES_PER_CHUNK)
                grouped.setdefault(start, {})[str(page)] = value
            for start, chunk_pages in grouped.items():
                data = {"version": 1, "key": key, "generation": new["generation"], "start": start,
                        "end": start + PAGES_PER_CHUNK - 1, "pages": chunk_pages}
                content = gzip.compress(json.dumps(data, ensure_ascii=False, sort_keys=True,
                                                   separators=(",", ":")).encode(), mtime=0)
                operations.append(CommitOperationAdd(path_in_repo=chunk_path(key, start),
                                                      path_or_fileobj=content))
            index = {"version": 1, "key": key, "generation": new["generation"],
                     "chunks": [{"start": start, "end": start + PAGES_PER_CHUNK - 1,
                                 "path": chunk_path(key, start)} for start in sorted(grouped)]}
            operations.append(CommitOperationAdd(path_in_repo=book_path(key),
                                                  path_or_fileobj=json.dumps(index, separators=(",", ":")).encode()))
        if operations:
            _commit(api, repo, operations, "Save compact PDF OCR progress")
