import io
import json
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from scripts import dispatch_pdf_stages as controller


REPO = "vomebook/pipeline"


class Response(io.BytesIO):
    def __init__(self, payload=b"", status=200):
        super().__init__(payload)
        self.status = status


class FakeGitHub:
    def __init__(self):
        self.runs = {REPO: [], "anftm/pipeline": []}
        self.jobs = {}
        self.posts = []
        self.hide_new_run = 0

    def open(self, request, timeout):
        if "/actions/runs/" in request.full_url:
            run_id = request.full_url.split("/actions/runs/", 1)[1].split("/", 1)[0]
            jobs = self.jobs.get(run_id, [])
            page = int(request.full_url.split("&page=", 1)[1])
            return Response(json.dumps({"total_count": len(jobs),
                                        "jobs": jobs[(page - 1) * 100:page * 100]}).encode())
        self_path = request.full_url.split("/actions/workflows/")
        repo = self_path[0].split("/repos/", 1)[1]
        if request.get_method() == "POST":
            self.posts.append((repo, self_path[1], json.loads(request.data)))
            self.runs[repo].insert(0, {"id": 123, "status": "queued"})
            return Response(status=204)
        if self_path[1].endswith("/runs?per_page=100"):
            runs = self.runs[repo]
            if self.hide_new_run and self.posts:
                self.hide_new_run -= 1
                runs = [run for run in runs if run["id"] != 123]
            return Response(json.dumps({"workflow_runs": runs}).encode())
        status = self_path[1].split("?status=", 1)[1].split("&", 1)[0]
        matches = [run for run in self.runs[repo] if run["status"] == status]
        return Response(json.dumps({"total_count": len(matches), "workflow_runs": matches[:1]}).encode())


class DispatchPdfStagesTests(unittest.TestCase):
    def test_small_completion_and_schedule_cannot_dispatch_twice(self):
        api = FakeGitHub()
        api.runs[REPO] = [{"id": 100, "status": "completed"}]
        api.jobs["100"] = [{"name": "plan", "conclusion": "success"},
                           {"name": "build (0)", "conclusion": "success"}]
        api.hide_new_run = 2
        with patch.object(controller, "urlopen", side_effect=api.open), \
                patch.object(controller.time, "sleep") as sleep, \
                patch.object(controller, "completed_queue", return_value=None):
            self.assertTrue(controller.dispatch(REPO, "token", "small", "100", "success"))
            self.assertFalse(controller.dispatch(REPO, "token", "small"))
        self.assertEqual(len(api.posts), 1)
        self.assertEqual(api.posts[0][2], {"ref": "main", "inputs": {"render_band": "under16"}})
        self.assertEqual(sleep.call_count, 2)

    def test_empty_or_failed_worker_does_not_trigger_another_immediate_run(self):
        api = FakeGitHub()
        api.jobs["100"] = [{"name": "plan", "conclusion": "success"},
                           {"name": "build", "conclusion": "skipped"},
                           {"name": "publish", "conclusion": "success"}]
        with patch.object(controller, "urlopen", side_effect=api.open):
            self.assertFalse(controller.dispatch(REPO, "token", "small", "100", "success"))
            self.assertFalse(controller.dispatch(REPO, "token", "ocr", "100", "failure"))
        self.assertEqual(api.posts, [])

    def test_missing_job_page_fails_closed(self):
        api = FakeGitHub()
        def truncated(request, timeout):
            if "/jobs?" in request.full_url:
                return Response(b'{"total_count":150,"jobs":[]}')
            return api.open(request, timeout)
        with patch.object(controller, "urlopen", side_effect=truncated):
            with self.assertRaisesRegex(ValueError, "job list"):
                controller.dispatch(REPO, "token", "small", "100", "success")
        self.assertEqual(api.posts, [])

    def test_build_after_first_job_page_still_triggers_next_batch(self):
        api = FakeGitHub()
        api.jobs["100"] = [{"name": "plan", "conclusion": "success"}] * 100 + [
            {"name": "build (100)", "conclusion": "success"}]
        with patch.object(controller, "urlopen", side_effect=api.open), \
                patch.object(controller, "completed_queue", return_value=None):
            self.assertTrue(controller.dispatch(REPO, "token", "ocr", "100", "success"))
        self.assertEqual(len(api.posts), 1)

    def test_completed_ocr_dispatches_bounded_exact_stale_repair(self):
        api = FakeGitHub()
        stale = [{"repo": "source/repo", "path": f"book-{i}.pdf", "page_count": 9}
                 for i in range(controller.STALE_REPAIR_BATCH_SIZE + 3)]
        with patch.object(controller, "urlopen", side_effect=api.open), \
                patch.object(controller, "completed_queue", return_value={"stale_render": stale}), \
                patch.object(controller, "completed_with_work", return_value=True):
            self.assertTrue(controller.dispatch(REPO, "token", "ocr", "100", "success"))
        self.assertEqual(len(api.posts), 1)
        repo, workflow, body = api.posts[0]
        self.assertEqual(workflow, "pdf-render-small-inputs.yml/dispatches")
        self.assertEqual(body["inputs"]["limit"], "20")
        self.assertEqual(body["inputs"]["force_reprobe"], "true")
        self.assertEqual(body["inputs"]["repair_stale"], "true")
        self.assertEqual(json.loads(body["inputs"]["source_items_json"]),
                         [{"repo": item["repo"], "path": item["path"]} for item in stale[:20]])

    def test_scheduled_ocr_check_prioritizes_latest_stale_repair_queue(self):
        api = FakeGitHub()
        stale = [{"repo": "source/repo", "path": f"stale-{i}.pdf"} for i in range(2)]
        render_run = {"id": 455, "created_at": "2026-10-01T10:00:00Z"}
        ocr_run = {"id": 456, "created_at": "2026-10-01T10:05:00Z"}
        with patch.object(controller, "urlopen", side_effect=api.open), \
                patch.object(controller, "latest_successful_run", side_effect=[ocr_run, render_run]), \
                patch.object(controller, "completed_queue", side_effect=[{}, {"stale_render": stale}]):
            self.assertTrue(controller.dispatch(REPO, "token", "ocr"))
        self.assertEqual(api.posts[0][1], "pdf-render-small-inputs.yml/dispatches")
        self.assertEqual(json.loads(api.posts[0][2]["inputs"]["source_items_json"]), stale)

    def test_scheduled_ocr_refreshes_queue_after_newer_repair_publish(self):
        api = FakeGitHub()
        ocr_run = {"id": 10, "created_at": "2026-10-01T10:00:00Z"}
        render_run = {"id": 11, "created_at": "2026-10-01T10:05:00Z"}
        with patch.object(controller, "urlopen", side_effect=api.open), \
                patch.object(controller, "latest_successful_run", side_effect=[ocr_run, render_run]), \
                patch.object(controller, "completed_queue", side_effect=[
                    {"stale_repair": True}, {"stale_render": [{"repo": "r", "path": "p.pdf"}]}
                ]):
            self.assertTrue(controller.dispatch(REPO, "token", "ocr"))
        self.assertEqual(api.posts[0][1], "pdf-ocr-assets.yml/dispatches")

    def test_scheduled_ocr_selects_next_stale_batch_when_ocr_is_newer(self):
        api = FakeGitHub()
        ocr_run = {"id": 10, "created_at": "2026-10-01T10:05:00Z"}
        render_run = {"id": 11, "created_at": "2026-10-01T10:00:00Z"}
        stale = [{"repo": "r", "path": "next.pdf"}]
        with patch.object(controller, "urlopen", side_effect=api.open), \
                patch.object(controller, "latest_successful_run", side_effect=[ocr_run, render_run]), \
                patch.object(controller, "completed_queue", side_effect=[
                    {}, {"stale_render": stale}
                ]):
            self.assertTrue(controller.dispatch(REPO, "token", "ocr"))
        self.assertEqual(api.posts[0][1], "pdf-render-small-inputs.yml/dispatches")
        self.assertEqual(json.loads(api.posts[0][2]["inputs"]["source_items_json"]), stale)

    def test_scheduled_ocr_removes_keys_from_latest_completed_repair(self):
        api = FakeGitHub()
        repaired = {"repo": "r", "path": "done.pdf", "key": "r\\0done.pdf"}
        remaining = {"repo": "r", "path": "next.pdf", "key": "r\\0next.pdf"}
        with patch.object(controller, "urlopen", side_effect=api.open), \
                patch.object(controller, "latest_successful_run", side_effect=[
                    {"id": 10, "created_at": "2026-10-01T10:05:00Z"},
                    {"id": 11, "created_at": "2026-10-01T10:00:00Z"}]), \
                patch.object(controller, "completed_queue", side_effect=[
                    {"stale_repair": True, "books": [repaired]},
                    {"stale_render": [repaired, remaining]}]):
            self.assertTrue(controller.dispatch(REPO, "token", "ocr"))
        self.assertEqual(api.posts[0][1], "pdf-render-small-inputs.yml/dispatches")
        self.assertEqual(json.loads(api.posts[0][2]["inputs"]["source_items_json"]),
                         [{"repo": remaining["repo"], "path": remaining["path"]}])

    def test_completed_repair_render_dispatches_ocr(self):
        api = FakeGitHub()
        with patch.object(controller, "urlopen", side_effect=api.open), \
                patch.object(controller, "completed_queue", return_value={"stale_repair": True}), \
                patch.object(controller, "completed_with_work", return_value=True):
            self.assertTrue(controller.dispatch(REPO, "token", "small", "100", "success"))
        self.assertEqual(api.posts, [(REPO, "pdf-ocr-assets.yml/dispatches",
                                     {"ref": "main", "inputs": {"lane_index": "0"}})])

    def test_repair_loop_ignores_completed_regular_render(self):
        api = FakeGitHub()
        with patch.object(controller, "urlopen", side_effect=api.open), \
                patch.object(controller, "completed_queue", return_value={"books": [{}]}), \
                patch.object(controller, "completed_with_work", return_value=True):
            self.assertFalse(controller.dispatch(
                REPO, "token", "small", "100", "success", repair_loop=True))
        self.assertEqual(api.posts, [])

    def test_ocr_waits_for_upstream_and_its_own_worker(self):
        api = FakeGitHub()
        api.runs["anftm/pipeline"] = [{"id": 10, "status": "pending"}]
        with patch.object(controller, "urlopen", side_effect=api.open):
            self.assertFalse(controller.dispatch(REPO, "token", "ocr"))
            api.runs["anftm/pipeline"] = []
            api.runs[REPO] = [{"id": 11, "status": "in_progress"}]
            self.assertFalse(controller.dispatch(REPO, "token", "ocr"))
            api.runs[REPO][0]["status"] = "completed"
            self.assertTrue(controller.dispatch(REPO, "token", "ocr"))
        self.assertEqual(api.posts, [(REPO, "pdf-ocr-assets.yml/dispatches",
                                     {"ref": "main", "inputs": {"lane_index": "0"}})])

    def test_ocr_uses_separate_upstream_token_for_anftm_status(self):
        api = FakeGitHub()
        seen = []
        def respond(request, timeout):
            if request.full_url.startswith("https://api.github.com/repos/anftm/"):
                seen.append(request.get_header("Authorization"))
            return api.open(request, timeout)
        with patch.object(controller, "urlopen", side_effect=respond):
            self.assertTrue(controller.dispatch(REPO, "repo-token", "ocr", upstream_token="upstream-token"))
        self.assertTrue(seen)
        self.assertEqual(set(seen), {"Bearer upstream-token"})

    def test_dispatch_carries_selected_render_band_and_ocr_lane(self):
        api = FakeGitHub()
        with patch.object(controller, "urlopen", side_effect=api.open):
            self.assertTrue(controller.dispatch(REPO, "token", "small", render_band="32to64"))
        api.runs[REPO].clear()
        with patch.object(controller, "urlopen", side_effect=api.open):
            self.assertTrue(controller.dispatch(REPO, "token", "ocr", ocr_lane_index=3))
        self.assertEqual(api.posts[0][2]["inputs"], {"render_band": "32to64"})
        self.assertEqual(api.posts[1][2]["inputs"], {"lane_index": "3"})

    def test_invalid_render_band_or_ocr_lane_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "render band"):
            controller.dispatch(REPO, "token", "small", render_band="large")
        with self.assertRaisesRegex(ValueError, "lane index"):
            controller.dispatch(REPO, "token", "ocr", ocr_lane_index=4)

    def test_api_error_and_malformed_response_fail_closed(self):
        api = FakeGitHub()
        def fail(request, timeout):
            if "?status=" in request.full_url:
                return Response(b"{}")
            return api.open(request, timeout)
        with patch.object(controller, "urlopen", side_effect=fail):
            with self.assertRaises(ValueError):
                controller.dispatch(REPO, "token", "small")
        with patch.object(controller, "urlopen", side_effect=OSError("API unavailable")):
            with self.assertRaises(OSError):
                controller.dispatch(REPO, "token", "ocr")
        self.assertEqual(api.posts, [])

    def test_dispatch_does_not_claim_success_before_run_is_visible(self):
        api = FakeGitHub()
        api.hide_new_run = 31
        with patch.object(controller, "urlopen", side_effect=api.open), \
                patch.object(controller.time, "sleep") as sleep:
            with self.assertRaisesRegex(RuntimeError, "not visible"):
                controller.dispatch(REPO, "token", "small")
        self.assertEqual(len(api.posts), 1)
        self.assertEqual(sleep.call_count, 30)

    def test_workflows_trigger_on_completion_and_have_separate_serial_queues(self):
        root = Path(__file__).resolve().parents[1] / ".github" / "workflows"
        for name, worker, label, minute in (
            ("scheduled-pdf-render-small.yml", "small", "Render Small PDF OCR Inputs", "2-59/5"),
            ("scheduled-pdf-ocr.yml", "ocr", "Build PDF OCR Assets", "3-59/5"),
        ):
            with self.subTest(name=name):
                workflow = yaml.load((root / name).read_text(), Loader=yaml.BaseLoader)
                self.assertEqual(workflow["on"]["schedule"][0]["cron"], f"{minute} * * * *")
                self.assertEqual(workflow["on"]["workflow_run"],
                                 {"workflows": [label], "types": ["completed"]})
                self.assertIn("workflow_dispatch", workflow["on"])
                self.assertEqual(workflow["concurrency"],
                                 {"group": "schedule-" + ("pdf-ocr" if worker == "ocr" else "small-pdf-render"),
                                  "cancel-in-progress": "false", "queue": "max"})
                self.assertIn(f"dispatch_pdf_stages.py {worker}", workflow["jobs"]["dispatch"]["steps"][-1]["run"])
                self.assertIn("head_branch == 'main'", workflow["jobs"]["dispatch"]["if"])
                self.assertIn("COMPLETED_RUN_ID", workflow["jobs"]["dispatch"]["steps"][-1]["env"])
                env = workflow["jobs"]["dispatch"]["steps"][-1]["env"]
                self.assertEqual(env["GH_TOKEN"], "${{ github.token }}")
                self.assertIn("UPSTREAM_GH_TOKEN", env)
                if worker == "small":
                    self.assertIn("vars.PDF_RENDER_BAND", env["RENDER_BAND"])
                    self.assertIn("vars.PDF_OCR_LANE_INDEX", env["OCR_LANE_INDEX"])
                else:
                    self.assertIn("vars.PDF_OCR_LANE_INDEX", env["OCR_LANE_INDEX"])

    def test_stale_repair_handoff_workflow_is_completion_only(self):
        path = Path(__file__).resolve().parents[1] / ".github/workflows/resume-ocr-after-stale-render.yml"
        workflow = yaml.load(path.read_text(), Loader=yaml.BaseLoader)
        self.assertEqual(workflow["on"]["workflow_run"], {
            "workflows": ["Render Small PDF OCR Inputs"], "types": ["completed"]})
        self.assertIn("workflow_dispatch", workflow["on"])
        command = workflow["jobs"]["resume"]["steps"][-1]["run"]
        self.assertIn("--repair-loop", command)


if __name__ == "__main__":
    unittest.main()
