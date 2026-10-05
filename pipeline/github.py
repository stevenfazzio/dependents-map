"""The GitHub API client the fetch stages share: retries, rate-limit waits, batch splitting
and the log each fetch pass resumes from.

GitHub allows 5,000 REST requests and 5,000 GraphQL points an hour, counted separately.
A long fetch uses all of both, so every response's remaining quota is noted, and a request
that would run a quota out waits for the hour to turn over instead of failing.
"""

import os
import re
import subprocess
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

import requests
from storage import append_jsonl, read_jsonl_log

GRAPHQL_URL = "https://api.github.com/graphql"
REST_URL = "https://api.github.com"
CONCURRENT_REQUESTS = 5
MAX_RETRIES = 5
RETRY_STATUS = {403, 429, 500, 502, 503, 504}  # GitHub signals secondary rate limits with 403
LOG_EVERY_BATCHES = 40
# Requests in flight when a quota runs low still land, so the wait starts a little early.
QUOTA_FLOOR = 40

# A result with one of these statuses is final; anything else is fetched again on a re-run.
SETTLED = {"ok", "not_found", "missing"}

# Committed environments and other people's code: not what the project itself wrote.
VENDORED = re.compile(
    r"(^|/)(site-packages|dist-packages|node_modules|\.?venv[^/]*|\.?env|__pycache__|"
    r"\.ipynb_checkpoints|lib/python[\d.]+|third_party|vendor)(/|$)",
    re.IGNORECASE,
)

_thread = threading.local()
_quota_lock = threading.Lock()
_quota: dict[str, tuple[int, int]] = {}  # resource -> (requests remaining, reset time)
_announced: set[tuple[str, int]] = set()  # the waits already printed


class BatchFailed(RuntimeError):
    """A query kept failing for reasons that are not about any one repository."""


def token() -> str:
    value = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if value:
        return value
    try:
        result = subprocess.run(
            ["gh", "auth", "token"], capture_output=True, text=True, check=True, timeout=30
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        raise SystemExit("Set GITHUB_TOKEN, or log in with `gh auth login`.") from None
    return result.stdout.strip()


def session(token: str) -> requests.Session:
    if not hasattr(_thread, "session"):
        _thread.session = requests.Session()
        _thread.session.headers["Authorization"] = f"Bearer {token}"
    return _thread.session


def note_quota(resp: requests.Response) -> None:
    """Record what a response says is left of the hourly quota it was counted against."""
    try:
        resource = resp.headers["X-RateLimit-Resource"]
        remaining = int(resp.headers["X-RateLimit-Remaining"])
        reset_at = int(resp.headers["X-RateLimit-Reset"])
    except (KeyError, ValueError):
        return
    with _quota_lock:
        known_remaining, known_reset = _quota.get(resource, (remaining, 0))
        if reset_at < known_reset:
            return  # a late response from the hour before
        # Responses arrive out of order, so within one hour the lowest count is the true one.
        if reset_at == known_reset:
            remaining = min(remaining, known_remaining)
        _quota[resource] = (remaining, reset_at)


def wait_for_quota(resource: str) -> None:
    """Sleep until the hour turns over if the last responses left almost nothing."""
    with _quota_lock:
        remaining, reset_at = _quota.get(resource, (QUOTA_FLOOR + 1, 0))
        wait = reset_at - time.time() + 5
        if remaining > QUOTA_FLOOR or wait <= 0:
            # Once the hour has turned, the next response records the new count.
            return
        announce = (resource, reset_at) not in _announced
        _announced.add((resource, reset_at))
    if announce:
        until = datetime.fromtimestamp(reset_at).strftime("%H:%M")
        print(
            f"  GitHub's hourly {resource} quota is used up; "
            f"waiting {wait / 60:.0f} min, until {until}",
            flush=True,
        )
    time.sleep(wait)


def text_of(raw: bytes) -> str:
    """Decode a file fetched as bytes."""
    # PowerShell's `pip freeze > requirements.txt` writes UTF-16, which GitHub calls binary.
    if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):
        return raw.decode("utf-16", errors="replace")
    return raw.decode("utf-8-sig", errors="replace")


def backoff(attempt: int, resp: requests.Response | None) -> int:
    """Seconds to wait before a retry, honouring GitHub's own hint when it sends one."""
    wait = min(2**attempt * 5, 120)
    if resp is not None:
        retry_after = resp.headers.get("Retry-After", "")
        if retry_after.isdigit():
            wait = max(wait, int(retry_after) + 1)
        elif resp.headers.get("X-RateLimit-Remaining") == "0":
            reset_at = int(resp.headers.get("X-RateLimit-Reset", 0))
            wait = max(wait, reset_at - int(time.time()) + 5)
    return wait


def graphql(token: str, query: str) -> tuple[dict, list[dict]]:
    """Run a query and return (data, errors that name a single alias).

    Errors with a path belong to one repository and are returned for the caller to
    record. Anything else (throttling, timeouts, server faults) is retried here.
    """
    reason = "unknown"
    for attempt in range(MAX_RETRIES):
        resp = None
        wait_for_quota("graphql")
        try:
            resp = session(token).post(GRAPHQL_URL, json={"query": query}, timeout=60)
        except (
            requests.Timeout,
            requests.ConnectionError,
            requests.exceptions.ChunkedEncodingError,
        ) as e:
            reason = type(e).__name__
        else:
            note_quota(resp)
            if resp.status_code == 200:
                body = resp.json()
                errors = body.get("errors", [])
                query_errors = [e for e in errors if not e.get("path")]
                if body.get("data") is not None and not query_errors:
                    return body["data"], errors
                first = (query_errors or errors or [{}])[0]
                reason = first.get("type") or first.get("message") or "no data"
            elif resp.status_code in RETRY_STATUS:
                reason = f"HTTP {resp.status_code}"
            else:
                resp.raise_for_status()
        if attempt < MAX_RETRIES - 1:
            wait = backoff(attempt, resp)
            print(f"  {reason}, retrying in {wait}s (attempt {attempt + 1}/{MAX_RETRIES})")
            time.sleep(wait)
    raise BatchFailed(reason)


def rest_get(token: str, path: str, accept: str) -> requests.Response | None:
    """GET from the REST API, waiting out rate limits. None when the retries run out."""
    for attempt in range(MAX_RETRIES):
        resp = None
        wait_for_quota("core")
        try:
            resp = session(token).get(f"{REST_URL}{path}", headers={"Accept": accept}, timeout=120)
        except (
            requests.Timeout,
            requests.ConnectionError,
            requests.exceptions.ChunkedEncodingError,
        ):
            pass
        else:
            note_quota(resp)
            if resp.status_code not in RETRY_STATUS:
                return resp
        if attempt < MAX_RETRIES - 1:
            time.sleep(backoff(attempt, resp))
    return None


def fetch_with_split(fetch: Callable[[list[str]], dict[str, dict]], names: list[str]) -> dict:
    """Run one batch, halving it on failure so one bad repository can't sink the rest."""
    try:
        return fetch(names)
    except BatchFailed as e:
        if len(names) == 1:
            return {names[0]: {"status": "error", "error": f"batch failed: {e}"}}
        half = len(names) // 2
        print(f"  Batch of {len(names)} failed ({e}); splitting")
        return {
            **fetch_with_split(fetch, names[:half]),
            **fetch_with_split(fetch, names[half:]),
        }


def latest_results(log_path: Path) -> dict[str, dict]:
    """Collapse a log to each repository's most recent result."""
    results = {}
    for record in read_jsonl_log(log_path):
        results.update(record["results"])
    return results


def run_pass(
    label: str,
    log_path: Path,
    names: list[str],
    batch_size: int,
    fetch: Callable[[list[str]], dict[str, dict]],
    workers: int = CONCURRENT_REQUESTS,
) -> dict[str, dict]:
    """Fetch every name without a settled result, logging each batch as it completes."""
    results = latest_results(log_path)
    todo = [n for n in names if results.get(n, {}).get("status") not in SETTLED]
    print(f"{label}: {len(names) - len(todo)} already settled, {len(todo)} to fetch")
    if not todo:
        return results

    batches = [todo[i : i + batch_size] for i in range(0, len(todo), batch_size)]
    done = 0
    with (
        open(log_path, "a") as log,
        ThreadPoolExecutor(max_workers=workers) as pool,
    ):
        futures = [pool.submit(fetch_with_split, fetch, batch) for batch in batches]
        for n, future in enumerate(as_completed(futures), start=1):
            batch_results = future.result()
            fetched_at = datetime.now(UTC).isoformat(timespec="seconds")
            append_jsonl(log, {"fetched_at": fetched_at, "results": batch_results})
            results.update(batch_results)
            done += len(batch_results)
            if n % LOG_EVERY_BATCHES == 0 or n == len(batches):
                print(f"{label}: {done}/{len(todo)}", flush=True)
    return results
