#!/usr/bin/env python3
"""Dispatch fork PDF work when both its own worker and any upstream OCR are idle."""

import argparse
import json
import os
import time
from urllib.request import Request, urlopen


WORKFLOWS = {"small": "pdf-render-small-inputs.yml", "ocr": "pdf-ocr-assets.yml"}
ACTIVE = ("pending", "requested", "queued", "in_progress", "waiting")


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


def dispatch(repo, token, worker, completed_run_id="", completed_conclusion=""):
    if not repo or not token or worker not in WORKFLOWS:
        raise ValueError("REPO, GH_TOKEN and a valid worker are required")
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json",
               "X-GitHub-Api-Version": "2022-11-28"}
    if completed_run_id:
        if not completed_run_id.isdecimal():
            raise ValueError("invalid completed PDF worker run ID")
        if completed_conclusion != "success" or not completed_with_work(repo, completed_run_id, headers):
            print(f"PDF {worker} worker had no successful build; waiting for scheduled check.")
            return False
    workflow = WORKFLOWS[worker]
    endpoint = f"https://api.github.com/repos/{repo}/actions/workflows/{workflow}"
    previous = recent_ids(endpoint, headers)
    targets = (repo, "anftm/pipeline") if worker == "ocr" else (repo,)
    for target in targets:
        target_endpoint = f"https://api.github.com/repos/{target}/actions/workflows/{workflow}"
        if is_active(target_endpoint, headers):
            print(f"PDF {worker} worker already pending or active in {target}; skipping dispatch.")
            return False

    request = Request(endpoint + "/dispatches", data=b'{"ref":"main"}',
                      headers={**headers, "Content-Type": "application/json"}, method="POST")
    with urlopen(request, timeout=30) as response:
        if response.status not in (200, 204):
            raise ValueError(f"unexpected PDF dispatch status: {response.status}")

    # Keep the controller's concurrency slot until GitHub lists the new run.
    # A simultaneous schedule/completion event must observe it before dispatching.
    for _ in range(30):
        if recent_ids(endpoint, headers) - previous:
            print(f"Dispatched PDF {worker} worker from main.")
            return True
        time.sleep(2)
    raise RuntimeError("PDF worker dispatch accepted but new run is not visible")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("worker", choices=WORKFLOWS)
    args = parser.parse_args()
    dispatch(os.environ.get("REPO"), os.environ.get("GH_TOKEN"), args.worker,
             os.environ.get("COMPLETED_RUN_ID", ""), os.environ.get("COMPLETED_CONCLUSION", ""))
