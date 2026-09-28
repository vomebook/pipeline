import gzip
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import requests
import yaml
from huggingface_hub.errors import HfHubHTTPError

from scripts import pdf_ocr_progress as progress


class FakeDataset:
    def __init__(self, root, legacy=None):
        self.root = root
        self.files = {}
        if legacy is not None:
            self.files[progress.LEGACY] = (json.dumps({"version": 1, "files": legacy},
                                                     ensure_ascii=False) + "\n").encode()
        self.downloads = []
        self.commits = []
        self.sha = "revision-0"

    def repo_info(self, **_kwargs):
        return SimpleNamespace(sha=self.sha)

    def get_paths_info(self, paths, **_kwargs):
        return [SimpleNamespace(path=path, lfs=SimpleNamespace(
            sha256=hashlib.sha256(self.files[path]).hexdigest())) for path in paths if path in self.files]

    def hf_hub_download(self, filename, **_kwargs):
        self.downloads.append(filename)
        if filename not in self.files:
            response = requests.Response()
            response.status_code = 404
            raise HfHubHTTPError("missing", response=response)
        path = self.root / hashlib.sha256(filename.encode()).hexdigest()
        path.write_bytes(self.files[filename])
        return str(path)

    def create_commit(self, operations, parent_commit, **_kwargs):
        if parent_commit != self.sha:
            raise AssertionError("unpinned commit")
        updates = {operation.path_in_repo: operation.path_or_fileobj for operation in operations}
        self.files.update(updates)
        self.commits.append(updates)
        self.sha = f"revision-{len(self.commits)}"


class ProgressTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_streams_legacy_once_and_indexes_all_keys_without_losing_other_books(self):
        first = "repo\0first.pdf"
        other = "repo\0other.pdf"
        missing = "repo\0new.pdf"
        old = {first: {"generation": "g", "pages": {"1": {"p": 1}}},
               other: {"generation": "h", "pages": {str(n): {"p": n} for n in range(3000)}}}
        api = FakeDataset(self.root, old)
        selected = progress.load_progress(api, "test/repo", [first, missing], api.sha)
        self.assertEqual(selected, {first: old[first]})
        self.assertEqual(len(api.commits), 1)
        indexed = json.loads(gzip.decompress(api.files[progress.LEGACY_KEYS]))
        self.assertEqual(set(indexed["keys"]), {first, other})
        api.downloads.clear()
        self.assertEqual(progress.load_progress(api, "test/repo", [missing], api.sha), {})
        self.assertNotIn(progress.LEGACY, api.downloads)

        progress.save_progress(api, "test/repo", {first: {"generation": "g", "pages": {
            **old[first]["pages"], "2": {"p": 2}}}})
        stored = progress.load_progress(api, "test/repo", [first], api.sha)
        self.assertEqual(stored[first]["pages"], {"1": {"p": 1}, "2": {"p": 2}})
        self.assertEqual(json.loads(api.files[progress.LEGACY])["files"], old)

    def test_per_book_progress_merges_same_generation_and_replaces_old_generation(self):
        key = "repo\0first.pdf"
        api = FakeDataset(self.root)
        progress.save_progress(api, "test/repo", {key: {"generation": "g", "pages": {"1": {"p": 1}}}})
        progress.save_progress(api, "test/repo", {key: {"generation": "g", "pages": {"2": {"p": 2}}}})
        self.assertEqual(progress.load_progress(api, "test/repo", [key], api.sha)[key]["pages"],
                         {"1": {"p": 1}, "2": {"p": 2}})
        progress.save_progress(api, "test/repo", {key: {"generation": "new", "pages": {"3": {"p": 3}}}})
        self.assertEqual(progress.load_progress(api, "test/repo", [key], api.sha)[key]["pages"],
                         {"3": {"p": 3}})
        self.assertNotIn(progress.LEGACY, api.files)

    def test_progress_chunks_keep_object_metadata_but_drop_text_and_layout(self):
        key = "repo\0large.pdf"
        api = FakeDataset(self.root)
        page = {"p": 1, "source": "ocr", "i": "input", "is": "i" * 64, "ib": 10,
                "o": "output", "os": "o" * 64, "ob": 20,
                "text": "重复文本", "text_spans": [{"start": 0}], "layout": {"large": True}}
        progress.save_progress(api, "test/repo", {key: {"generation": "g", "pages": {"1": page}}})
        stored = progress.load_progress(api, "test/repo", [key], api.sha)[key]
        self.assertEqual(stored["pages"]["1"], {field: page[field] for field in progress.COMPACT_FIELDS})
        self.assertNotIn("text", gzip.decompress(api.files[progress.chunk_path(key, 0)]).decode())

    def test_corrupt_per_book_file_fails_closed(self):
        key = "repo\0first.pdf"
        api = FakeDataset(self.root)
        api.files[progress.book_path(key)] = b"not-json"
        with self.assertRaisesRegex((ValueError, UnicodeDecodeError), ""):
            progress.load_progress(api, "test/repo", [key], api.sha)

    def test_index_is_invalidated_if_legacy_changes(self):
        first = "repo\0first.pdf"
        api = FakeDataset(self.root, {first: {"generation": "g", "pages": {}}})
        progress.load_progress(api, "test/repo", [first], api.sha)
        second = "repo\0second.pdf"
        api.files[progress.LEGACY] = json.dumps({"version": 1, "files": {
            first: {"generation": "g", "pages": {}}, second: {"generation": "h", "pages": {}}}}).encode()
        api.downloads.clear()
        self.assertIn(second, progress.load_progress(api, "test/repo", [second], api.sha))
        self.assertIn(progress.LEGACY, api.downloads)

    def test_plan_only_requests_selected_books_progress(self):
        from scripts import pdf_ocr_stages as stages
        lookup = Mock(return_value={})
        self.assertEqual(stages.plan_images({}, {}, lookup)["shard_count"], 0)
        lookup.assert_called_once_with([])

    def test_recovery_workflow_only_publishes_verified_existing_artifacts(self):
        workflow = yaml.load((Path(__file__).resolve().parents[1] / ".github/workflows/pdf-ocr-assets.yml").read_text(),
                             Loader=yaml.BaseLoader)
        self.assertEqual(workflow["permissions"]["actions"], "read")
        self.assertEqual(workflow["on"]["workflow_dispatch"]["inputs"]["recover_run"]["default"], "")
        plan = workflow["jobs"]["plan"]
        build = workflow["jobs"]["build"]
        publish = workflow["jobs"]["publish"]
        self.assertIn("inputs.recover_run == ''", build["if"])
        self.assertEqual([step["if"] for step in plan["steps"] if step.get("name") ==
                          "Plan only published high quality PNG inputs"], ["inputs.recover_run == ''"])
        steps = publish["steps"]
        validation = next(index for index, step in enumerate(steps) if step.get("name") ==
                          "Validate recovery run identity")
        downloads = [index for index, step in enumerate(steps) if step.get("uses") ==
                     "actions/download-artifact@v8"]
        self.assertLess(validation, min(downloads))
        self.assertIn('.path == ".github/workflows/pdf-ocr-assets.yml"', steps[validation]["run"])
        self.assertTrue(all(steps[index]["with"]["run-id"] ==
                            "${{ inputs.recover_run || github.run_id }}" for index in downloads))


if __name__ == "__main__":
    unittest.main()
