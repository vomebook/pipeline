#!/usr/bin/env python3
"""Dispatch fork PDF work when both its own worker and any upstream OCR are idle."""

import argparse
import io
import json
import os
import time
import zipfile
from urllib.request import HTTPRedirectHandler, Request, build_opener, urlopen


WORKFLOWS = {"small": "pdf-render-small-inputs.yml", "ocr": "pdf-ocr-assets.yml"}
ACTIVE = ("pending", "requested", "queued", "in_progress", "waiting")
STALE_REPAIR_BATCH_SIZE = 20


def api_json(url, headers):
    with urlopen(Request(url, headers=headers), timeout=30) as response:
        return json.load(response)


def recent_ids(endpoint, headers):
    payload = api_json(endpoint + "/runs?per_page=100", headers)
    runs = payload.get("workflow_runs") if isinstance(payload, dict) else None
    if not isinstance(runs, list) or any(not isinstance(run, dict) or type(run.get("id")) is not int
                                         for run in runs):
        raise ValueError("invalid PDF workflow run list")
    return {run["id"] for run in runs}


def is_active(endpoint, headers):
    for status in ACTIVE:
        payload = api_json(endpoint + f"/runs?status={status}&per_page=1", headers)
        count = payload.get("total_count") if isinstance(payload, dict) else None
        if type(count) is not int or count < 0:
            raise ValueError("invalid PDF workflow status count")
        if count:
            return True
    return False


def completed_with_work(repo, run_id, headers):
    endpoint = f"https://api.github.com/repos/{repo}/actions/runs/{run_id}/jobs"
    page = 1
    while True:
        payload = api_json(endpoint + f"?per_page=100&page={page}", headers)
        jobs = payload.get("jobs") if isinstance(payload, dict) else None
        count = payload.get("total_count") if isinstance(payload, dict) else None
        if not isinstance(jobs, list) or type(count) is not int or count < len(jobs):
            raise ValueError("invalid completed PDF worker job list")
        if not jobs and page * 100 < count:
            raise ValueError("incomplete completed PDF worker job list")
        if any(isinstance(job, dict) and (job.get("name") == "build" or
               str(job.get("name", "")).startswith("build (")) and job.get("conclusion") == "success"
               for job in jobs):
            return True
        if page * 100 >= count:
            return False
        page += 1


def completed_queue(repo, run_id, artifact_name, headers):
    endpoint = f"https://api.github.com/repos/{repo}/actions/runs/{run_id}/artifacts?per_page=100"
    payload = api_json(endpoint, headers)
    artifacts = payload.get("artifacts") if isinstance(payload, dict) else None
    if not isinstance(artifacts, list):
        raise ValueError("invalid completed PDF worker artifact list")
    artifact = next((item for item in artifacts if isinstance(item, dict)
                     and item.get("name") == artifact_name and item.get("expired") is False), None)
    if artifact is None:
        return None

    class NoRedirect(HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, response_headers, newurl):
            return None

    request = Request(artifact.get("archive_download_url", ""), headers=headers)
    github_api = build_opener(NoRedirect)
    try:
        with github_api.open(request, timeout=60) as response:
            archive = response.read()
    except Exception as exc:
        if getattr(exc, "code", None) != 302:
            raise
        download_url = exc.headers.get("Location")
        if not download_url:
            raise ValueError("artifact download redirect is missing Location") from exc
        # The signed storage URL is a separate host; never forward the GitHub token.
        with urlopen(Request(download_url), timeout=120) as response:
            archive = response.read()
    with zipfile.ZipFile(io.BytesIO(archive)) as bundle:
        json_files = [name for name in bundle.namelist() if name.endswith(".json")]
        if len(json_files) != 1:
            raise ValueError("completed PDF queue artifact must contain one JSON file")
        queue = json.loads(bundle.read(json_files[0]))
    if not isinstance(queue, dict):
        raise ValueError("invalid completed PDF queue")
    return queue


def latest_successful_run(endpoint, headers):
    payload = api_json(endpoint + "/runs?per_page=100", headers)
    runs = payload.get("workflow_runs") if isinstance(payload, dict) else None
    if not isinstance(runs, list):
        raise ValueError("invalid PDF workflow run list")
    return next((run for run in runs if isinstance(run, dict)
                 and run.get("status") == "completed" and run.get("conclusion") == "success"
                 and type(run.get("id")) is int), None)


def dispatch_direct(repo, token, workflow, inputs, check_targets=()):
    if not repo or not token or workflow not in WORKFLOWS.values():
        raise ValueError("REPO, GH_TOKEN and a valid worker workflow are required")
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json",
               "X-GitHub-Api-Version": "2022-11-28"}
    endpoint = f"https://api.github.com/repos/{repo}/actions/workflows/{workflow}"
    previous = recent_ids(endpoint, headers)
    for target, target_token in check_targets:
        target_headers = {**headers, "Authorization": f"Bearer {target_token or token}"}
        if is_active(f"https://api.github.com/repos/{target}/actions/workflows/{workflow}", target_headers):
            print(f"PDF handoff worker already pending or active in {target}; skipping dispatch.")
            return False
    request = Request(endpoint + "/dispatches",
                      data=json.dumps({"ref": "main", "inputs": inputs}).encode(),
                      headers={**headers, "Content-Type": "application/json"}, method="POST")
    with urlopen(request, timeout=30) as response:
        if response.status not in (200, 204):
            raise ValueError(f"unexpected PDF dispatch status: {response.status}")
    for _ in range(30):
        if recent_ids(endpoint, headers) - previous:
            print(f"Dispatched PDF handoff {workflow} from main.")
            return True
        time.sleep(2)
    raise RuntimeError("PDF handoff dispatch accepted but new run is not visible")


def dispatch_stale_repair(repo, token, queue_path, render_band="under16"):
    with open(queue_path, encoding="utf-8") as stream:
        queue = json.load(stream)
    stale = queue.get("stale_render", [])
    if not isinstance(stale, list):
        raise ValueError("invalid stale render list in OCR queue")
    batch = [item for item in stale if isinstance(item, dict)
             and isinstance(item.get("repo"), str) and isinstance(item.get("path"), str)][:20]
    if not batch:
        print("OCR queue has no stale render entries.")
        return False
    inputs = {"limit": str(len(batch)), "render_band": render_band, "retry_failed": "true",
              "source_repo": "", "source_path_prefix": "",
              "source_items_json": json.dumps(
                  [{"repo": item["repo"], "path": item["path"]} for item in batch],
                  ensure_ascii=False, separators=(",", ":")),
              "force_reprobe": "true", "repair_stale": "true"}
    return dispatch_direct(repo, token, WORKFLOWS["small"], inputs,
                           check_targets=((repo, token),))


def dispatch_repair_ocr(repo, token, lane_index=0, upstream_token=""):
    if type(lane_index) is not int or not 0 <= lane_index < 4:
        raise ValueError("OCR lane index must be between 0 and 3")
    checks = [(repo, token), ("anftm/pipeline", upstream_token or token)]
    return dispatch_direct(repo, token, WORKFLOWS["ocr"], {"lane_index": str(lane_index)}, checks)


def dispatch(repo, token, worker, completed_run_id="", completed_conclusion="",
             render_band="under16", ocr_lane_index=0, upstream_token="", repair_loop=False):
    if not repo or not token or worker not in WORKFLOWS:
        raise ValueError("REPO, GH_TOKEN and a valid worker are required")
    if worker == "small" and render_band not in {"under16", "16to32", "32to64", "64to100", "under100"}:
        raise ValueError("invalid small PDF render band")
    if worker == "ocr" and (type(ocr_lane_index) is not int or not 0 <= ocr_lane_index < 4):
        raise ValueError("OCR lane index must be between 0 and 3")
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json",
               "X-GitHub-Api-Version": "2022-11-28"}
    dispatch_worker = worker
    stale_items = []
    repair_render = False
    repair_published_after_ocr = False
    if worker == "ocr" and not completed_run_id:
        ocr_endpoint = f"https://api.github.com/repos/{repo}/actions/workflows/{WORKFLOWS['ocr']}"
        repo_headers = {
            "Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28"}
        latest = latest_successful_run(ocr_endpoint, repo_headers)
        render_endpoint = f"https://api.github.com/repos/{repo}/actions/workflows/{WORKFLOWS['small']}"
        latest_render = latest_successful_run(render_endpoint, repo_headers)
        repaired_keys = set()
        if latest_render:
            render_queue = completed_queue(repo, str(latest_render["id"]),
                                           "pdf-render-small-queue", repo_headers) or {}
            if render_queue.get("stale_repair") is True:
                repaired_keys = {item.get("key") for item in render_queue.get("books", [])
                                 if isinstance(item, dict) and isinstance(item.get("key"), str)}
            repair_published_after_ocr = (
                render_queue.get("stale_repair") is True and
                (not latest or latest_render.get("created_at", "") > latest.get("created_at", "")))
            if repair_published_after_ocr:
                # Refresh the OCR queue after a repair before selecting another batch.
                dispatch_worker = "ocr"
        if latest:
            queue = completed_queue(repo, str(latest["id"]), "pdf-image-ocr-queue", repo_headers) or {}
            stale = queue.get("stale_render", [])
            if not isinstance(stale, list):
                raise ValueError("invalid stale render list in OCR queue")
            stale_items = [item for item in stale if isinstance(item, dict)
                           and isinstance(item.get("repo"), str)
                           and isinstance(item.get("path"), str)
                           and item.get("key") not in repaired_keys]
            stale_items = stale_items[:STALE_REPAIR_BATCH_SIZE]
            if stale_items and not repair_published_after_ocr:
                repair_render = True
                dispatch_worker = "small"
    if completed_run_id:
        if not completed_run_id.isdecimal():
            raise ValueError("invalid completed PDF worker run ID")
        if completed_conclusion != "success":
            print(f"PDF {worker} worker did not succeed; waiting for scheduled check.")
            return False
        if worker == "ocr":
            queue = completed_queue(repo, completed_run_id, "pdf-image-ocr-queue", headers) or {}
            stale = queue.get("stale_render", [])
            if not isinstance(stale, list):
                raise ValueError("invalid stale render list in OCR queue")
            stale_items = [item for item in stale if isinstance(item, dict)
                           and isinstance(item.get("repo"), str)
                           and isinstance(item.get("path"), str)][:STALE_REPAIR_BATCH_SIZE]
            if stale_items:
                dispatch_worker = "small"
                repair_render = True
            elif not completed_with_work(repo, completed_run_id, headers):
                print("PDF OCR worker had no work; waiting for scheduled check.")
                return False
        elif worker == "small":
            if not completed_with_work(repo, completed_run_id, headers):
                print("PDF small render had no successful build; waiting for scheduled check.")
                return False
            queue = completed_queue(repo, completed_run_id, "pdf-render-small-queue", headers) or {}
            if queue.get("stale_repair") is True:
                dispatch_worker = "ocr"
            elif repair_loop:
                print("Completed render was not a stale repair; leaving regular render scheduling paused.")
                return False

    workflow = WORKFLOWS[dispatch_worker]
    endpoint = f"https://api.github.com/repos/{repo}/actions/workflows/{workflow}"
    previous = recent_ids(endpoint, headers)
    targets = ((repo, "anftm/pipeline") if dispatch_worker == "ocr" else (repo,))
    for target in targets:
        target_endpoint = f"https://api.github.com/repos/{target}/actions/workflows/{workflow}"
        target_headers = headers
        if target != repo and upstream_token:
            target_headers = {**headers, "Authorization": f"Bearer {upstream_token}"}
        if is_active(target_endpoint, target_headers):
            print(f"PDF {dispatch_worker} worker already pending or active in {target}; skipping dispatch.")
            return False
    if repair_render:
        ocr_endpoint = f"https://api.github.com/repos/{repo}/actions/workflows/{WORKFLOWS['ocr']}"
        upstream_headers = ({**headers, "Authorization": f"Bearer {upstream_token}"}
                            if upstream_token else headers)
        for target in (repo, "anftm/pipeline"):
            target_headers = upstream_headers if target != repo else headers
            if is_active(f"https://api.github.com/repos/{target}/actions/workflows/{WORKFLOWS['ocr']}",
                         target_headers):
                print(f"PDF stale repair waiting for OCR worker to become idle in {target}.")
                return False

    if dispatch_worker == "small":
        inputs = {"render_band": render_band}
        if repair_render:
            inputs.update({"limit": str(len(stale_items)),
                           "source_items_json": json.dumps(
                               [{"repo": item["repo"], "path": item["path"]} for item in stale_items],
                               ensure_ascii=False, separators=(",", ":")),
                           "force_reprobe": "true", "repair_stale": "true"})
    else:
        inputs = {"lane_index": str(ocr_lane_index)}
    body = json.dumps({"ref": "main", "inputs": inputs}).encode()
    request = Request(endpoint + "/dispatches", data=body,
                      headers={**headers, "Content-Type": "application/json"}, method="POST")
    with urlopen(request, timeout=30) as response:
        if response.status not in (200, 204):
            raise ValueError(f"unexpected PDF dispatch status: {response.status}")

    # Keep the controller's concurrency slot until GitHub lists the new run.
    # A simultaneous schedule/completion event must observe it before dispatching.
    for _ in range(30):
        if recent_ids(endpoint, headers) - previous:
            print(f"Dispatched PDF {dispatch_worker} worker from main.")
            return True
        time.sleep(2)
    raise RuntimeError("PDF worker dispatch accepted but new run is not visible")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("worker", choices=(*WORKFLOWS, "repair-stale", "resume-ocr"))
    parser.add_argument("--render-band", choices=("under16", "16to32", "32to64", "64to100", "under100"), default="under16")
    parser.add_argument("--lane-index", type=int, default=0)
    parser.add_argument("--repair-loop", action="store_true")
    parser.add_argument("--queue", default="")
    parser.add_argument("--upstream-token", default=os.environ.get("UPSTREAM_GH_TOKEN", ""))
    args = parser.parse_args()
    repo, token = os.environ.get("REPO"), os.environ.get("GH_TOKEN")
    if args.worker == "repair-stale":
        if not args.queue:
            parser.error("repair-stale requires --queue")
        dispatch_stale_repair(repo, token, args.queue, args.render_band)
    elif args.worker == "resume-ocr":
        dispatch_repair_ocr(repo, token, args.lane_index, args.upstream_token)
    else:
        dispatch(repo, token, args.worker, os.environ.get("COMPLETED_RUN_ID", ""),
                 os.environ.get("COMPLETED_CONCLUSION", ""), args.render_band,
                 args.lane_index, os.environ.get("UPSTREAM_GH_TOKEN", ""), args.repair_loop)
