"""Fetch metadata and READMEs for every repository among a target's dependents.

That is every repository GitHub's dependents lists name, plus the repositories that
PyPI metadata pointed at for the packages GitHub listed without one.

Three passes. The first two go to GitHub's GraphQL API, batched as aliased
repository() lookups; the third is one REST call per repository, for the few that need it:
  1. Metadata, plus the names of the files in the repository root.
  2. The README blob, by the exact filename pass 1 found.
  3. For repositories with no README in the root but a .github/ or docs/ directory,
     the README GitHub itself recognises there.

Each completed batch is appended to a JSONL log, and a re-run fetches only the
repositories with no settled result yet.

Nothing is filtered here. Every requested repository gets a row in repos.parquet with
a status saying what happened to it; choosing the corpus is the next stage's job.

Usage:
    uv run python pipeline/02_fetch_repos.py umap-learn
    uv run python pipeline/02_fetch_repos.py umap-learn --limit 200
"""

import argparse
import base64
import json
import os
import subprocess
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import requests
from config import ROOT, Target, get_target
from storage import append_jsonl, read_jsonl_log, write_parquet_safely

GRAPHQL_URL = "https://api.github.com/graphql"
REST_URL = "https://api.github.com"
METADATA_BATCH_SIZE = 25
README_BATCH_SIZE = 50
CONCURRENT_REQUESTS = 5
MAX_RETRIES = 5
RETRY_STATUS = {403, 429, 500, 502, 503, 504}  # GitHub signals secondary rate limits with 403
LOG_EVERY_BATCHES = 40

# A result with one of these statuses is final; anything else is fetched again on a re-run.
SETTLED = {"ok", "not_found", "missing"}

METADATA_FRAGMENT = """
fragment Metadata on Repository {
  nameWithOwner
  databaseId
  description
  homepageUrl
  stargazerCount
  forkCount
  isFork
  parent { nameWithOwner }
  isArchived
  isEmpty
  diskUsage
  createdAt
  pushedAt
  primaryLanguage { name }
  licenseInfo { spdxId }
  owner { __typename }
  repositoryTopics(first: 20) { nodes { topic { name } } }
  root: object(expression: "HEAD:") { ... on Tree { entries { name type } } }
}
"""

README_FRAGMENT = """
fragment ReadmeBlob on Blob {
  text
  byteSize
  isBinary
  isTruncated
}
"""

# Preferred README format when a repository root holds several, best first.
README_EXTENSIONS = (".md", ".markdown", ".rst", ".txt", "")
# Directories other than the root where GitHub recognises a README.
README_DIRS = {".github", "docs"}

_thread = threading.local()


class BatchFailed(RuntimeError):
    """A query kept failing for reasons that are not about any one repository."""


def github_token() -> str:
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if token:
        return token
    try:
        result = subprocess.run(
            ["gh", "auth", "token"], capture_output=True, text=True, check=True, timeout=30
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        raise SystemExit("Set GITHUB_TOKEN, or log in with `gh auth login`.") from None
    return result.stdout.strip()


def _session(token: str) -> requests.Session:
    if not hasattr(_thread, "session"):
        _thread.session = requests.Session()
        _thread.session.headers["Authorization"] = f"Bearer {token}"
    return _thread.session


def _backoff(attempt: int, resp: requests.Response | None) -> int:
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
        try:
            resp = _session(token).post(GRAPHQL_URL, json={"query": query}, timeout=60)
        except (
            requests.Timeout,
            requests.ConnectionError,
            requests.exceptions.ChunkedEncodingError,
        ) as e:
            reason = type(e).__name__
        else:
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
            wait = _backoff(attempt, resp)
            print(f"  {reason}, retrying in {wait}s (attempt {attempt + 1}/{MAX_RETRIES})")
            time.sleep(wait)
    raise BatchFailed(reason)


def _repository_alias(i: int, full_name: str, selection: str) -> str:
    owner, name = full_name.split("/", 1)
    # json.dumps output is a valid GraphQL string literal.
    return f"  r{i}: repository(owner: {json.dumps(owner)}, name: {json.dumps(name)}) {selection}"


def _alias_errors(errors: list[dict]) -> dict[str, str]:
    return {e["path"][0]: e.get("type") or e.get("message", "unknown") for e in errors}


def fetch_metadata(token: str, names: list[str]) -> dict[str, dict]:
    aliases = [_repository_alias(i, name, "{ ...Metadata }") for i, name in enumerate(names)]
    query = "query {\n" + "\n".join(aliases) + "\n}\n" + METADATA_FRAGMENT
    data, errors = graphql(token, query)
    alias_errors = _alias_errors(errors)

    results = {}
    for i, name in enumerate(names):
        node = data.get(f"r{i}")
        error = alias_errors.get(f"r{i}")
        if node is not None and error is None:
            results[name] = {"status": "ok", "repo": node}
        elif error == "NOT_FOUND":
            results[name] = {"status": "not_found"}
        else:
            results[name] = {"status": "error", "error": error or "null without an error"}
    return results


def fetch_readmes(token: str, readme_names: dict[str, str]) -> dict[str, dict]:
    """Fetch README blobs, given {repository: README filename}."""
    names = list(readme_names)
    aliases = []
    for i, name in enumerate(names):
        expression = json.dumps(f"HEAD:{readme_names[name]}", ensure_ascii=False)
        blob = f"{{ readme: object(expression: {expression}) {{ ...ReadmeBlob }} }}"
        aliases.append(_repository_alias(i, name, blob))
    data, errors = graphql(token, "query {\n" + "\n".join(aliases) + "\n}\n" + README_FRAGMENT)
    alias_errors = _alias_errors(errors)

    results = {}
    for i, name in enumerate(names):
        node = data.get(f"r{i}")
        error = alias_errors.get(f"r{i}")
        if error is not None or node is None:
            # The repository answered in pass 1, so its disappearance here is not settled.
            results[name] = {"status": "error", "error": error or "null without an error"}
        elif not node.get("readme"):
            results[name] = {"status": "missing", "readme_name": readme_names[name]}
        else:
            results[name] = {"status": "ok", "readme_name": readme_names[name], **node["readme"]}
    return results


def fetch_readme_outside_root(token: str, name: str) -> dict:
    """Ask the REST API for the README GitHub recognises, wherever in the repository it is."""
    reason = "unknown"
    for attempt in range(MAX_RETRIES):
        resp = None
        try:
            resp = _session(token).get(
                f"{REST_URL}/repos/{name}/readme",
                headers={"Accept": "application/vnd.github+json"},
                timeout=60,
            )
        except (requests.Timeout, requests.ConnectionError) as e:
            reason = type(e).__name__
        else:
            if resp.status_code == 200:
                body = resp.json()
                text = base64.b64decode(body["content"]).decode("utf-8", errors="replace")
                return {
                    "status": "ok",
                    "readme_name": body["path"],
                    "text": text,
                    "byteSize": body["size"],
                }
            if resp.status_code == 404:
                return {"status": "missing"}
            if resp.status_code not in RETRY_STATUS:
                return {"status": "error", "error": f"HTTP {resp.status_code}"}
            reason = f"HTTP {resp.status_code}"
        if attempt < MAX_RETRIES - 1:
            time.sleep(_backoff(attempt, resp))
    return {"status": "error", "error": reason}


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
        ThreadPoolExecutor(max_workers=CONCURRENT_REQUESTS) as pool,
    ):
        futures = [pool.submit(fetch_with_split, fetch, batch) for batch in batches]
        for n, future in enumerate(as_completed(futures), start=1):
            batch_results = future.result()
            fetched_at = datetime.now(UTC).isoformat(timespec="seconds")
            append_jsonl(log, {"fetched_at": fetched_at, "results": batch_results})
            results.update(batch_results)
            done += len(batch_results)
            if n % LOG_EVERY_BATCHES == 0 or n == len(batches):
                print(f"{label}: {done}/{len(todo)}")
    return results


def pick_readme(entries: list[dict]) -> str | None:
    """Choose the README among a repository root's files, or None if there isn't one."""
    candidates = []
    for entry in entries:
        parts = entry["name"].lower().split(".")
        if entry["type"] != "blob" or parts[0] != "readme":
            continue
        extension = f".{parts[-1]}" if len(parts) > 1 else ""
        rank = (
            README_EXTENSIONS.index(extension)
            if extension in README_EXTENSIONS
            else len(README_EXTENSIONS)
        )
        # A plain README.md beats a translation such as README.zh-CN.md.
        candidates.append((len(parts) > 2, rank, entry["name"]))
    return min(candidates)[2] if candidates else None


def build_table(
    names: list[str], metadata: dict[str, dict], readmes: dict[str, dict]
) -> pd.DataFrame:
    rows = []
    for name in names:
        meta = metadata.get(name, {"status": "unfetched"})
        row = {"full_name": name, "status": meta["status"], "error": meta.get("error")}
        if meta["status"] == "ok":
            repo = meta["repo"]
            entries = (repo.get("root") or {}).get("entries") or []
            readme = readmes.get(name, {})
            row |= {
                "resolved_name": repo["nameWithOwner"],
                "repo_id": repo["databaseId"],
                "description": repo["description"],
                "homepage_url": repo["homepageUrl"],
                "stars": repo["stargazerCount"],
                "forks": repo["forkCount"],
                "is_fork": repo["isFork"],
                "parent": (repo["parent"] or {}).get("nameWithOwner"),
                "is_archived": repo["isArchived"],
                "is_empty": repo["isEmpty"],
                "disk_usage_kb": repo["diskUsage"],
                "created_at": repo["createdAt"],
                "pushed_at": repo["pushedAt"],
                "language": (repo["primaryLanguage"] or {}).get("name"),
                "license": (repo["licenseInfo"] or {}).get("spdxId"),
                "owner_type": repo["owner"]["__typename"],
                "topics": [n["topic"]["name"] for n in repo["repositoryTopics"]["nodes"]],
                "root_files": [e["name"] for e in entries],
                "readme_name": readme.get("readme_name") or pick_readme(entries),
                "readme_status": readme.get("status"),
                "readme": readme.get("text"),
                "readme_bytes": readme.get("byteSize"),
                "readme_is_truncated": readme.get("isTruncated"),
            }
        rows.append(row)

    df = pd.DataFrame.from_records(rows)
    for column in ("created_at", "pushed_at"):
        df[column] = pd.to_datetime(df[column])
    for column in ("repo_id", "stars", "forks", "disk_usage_kb", "readme_bytes"):
        df[column] = df[column].astype("Int64")
    for column in ("is_fork", "is_archived", "is_empty", "readme_is_truncated"):
        df[column] = df[column].astype("boolean")
    return df


def report(df: pd.DataFrame) -> None:
    print(f"Requested {len(df)} repositories: {df['status'].value_counts().to_dict()}")
    ok = df[df["status"] == "ok"]
    has_text = ok["readme"].fillna("").str.strip().str.len() > 0
    print(
        f"Of {len(ok)} found: {has_text.sum()} have README text, "
        f"{ok['readme_name'].isna().sum()} have no README, "
        f"{(ok['readme_name'].notna() & ~has_text).sum()} have one we got no text from"
    )
    print(
        f"  forks: {ok['is_fork'].sum()}, "
        f"renamed since listed: {(ok['resolved_name'] != ok['full_name']).sum()}, "
        f"listed under more than one name: {ok['repo_id'].duplicated().sum()}"
    )


def load_repository_names(target: Target) -> list[str]:
    """Every repository to look up: those GitHub listed, plus those PyPI pointed at."""
    dependents = pd.read_parquet(target.dependents_parquet)
    listed = dependents["full_name"].dropna()
    pypi = pd.read_parquet(target.pypi_parquet)
    from_pypi = pypi["github_repo"].dropna()

    # GitHub names are case-insensitive, and PyPI metadata often spells them differently.
    by_key = {name.lower(): name for name in from_pypi}
    n_pypi_only = len(by_key.keys() - set(listed.str.lower()))
    by_key |= {name.lower(): name for name in listed}
    names = sorted(by_key.values())
    print(
        f"{len(dependents)} dependent rows → {listed.str.lower().nunique()} listed repositories "
        f"({dependents['full_name'].isna().sum()} rows name no repository, "
        f"{len(listed) - listed.str.lower().nunique()} are repeats)"
    )
    print(
        f"{len(pypi)} PyPI lookups → {len(from_pypi)} point at a repository, "
        f"{n_pypi_only} of them not already listed"
    )
    print(f"{len(names)} repositories to look up")
    return names


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("target", help="target slug from config.TARGETS")
    parser.add_argument("--limit", type=int, help="only the first N repositories (for testing)")
    args = parser.parse_args()
    target: Target = get_target(args.target)
    token = github_token()

    names = load_repository_names(target)
    if args.limit:
        names = names[: args.limit]
        print(f"--limit: keeping the first {len(names)}")

    metadata = run_pass(
        "metadata",
        target.repo_metadata_log,
        names,
        METADATA_BATCH_SIZE,
        lambda batch: fetch_metadata(token, batch),
    )

    readme_names = {}
    for name in names:
        meta = metadata.get(name, {})
        if meta.get("status") == "ok":
            entries = (meta["repo"].get("root") or {}).get("entries") or []
            if readme_name := pick_readme(entries):
                readme_names[name] = readme_name
    run_pass(
        "readmes",
        target.repo_readme_log,
        list(readme_names),
        README_BATCH_SIZE,
        lambda batch: fetch_readmes(token, {n: readme_names[n] for n in batch}),
    )

    elsewhere = []
    for name in names:
        meta = metadata.get(name, {})
        if meta.get("status") == "ok" and name not in readme_names:
            entries = (meta["repo"].get("root") or {}).get("entries") or []
            if any(e["type"] == "tree" and e["name"] in README_DIRS for e in entries):
                elsewhere.append(name)
    # Same log as pass 2: the two passes cover disjoint repositories.
    readmes = run_pass(
        "readmes outside the root",
        target.repo_readme_log,
        elsewhere,
        1,
        lambda batch: {batch[0]: fetch_readme_outside_root(token, batch[0])},
    )

    df = build_table(names, metadata, readmes)
    report(df)
    if args.limit:
        print("--limit run: no parquet written.")
        return
    write_parquet_safely(df, target.repos_parquet)
    print(f"Wrote {len(df)} rows to {target.repos_parquet.relative_to(ROOT)}")
    unsettled = (df["status"] == "error").sum() + (df["readme_status"] == "error").sum()
    if unsettled:
        print(f"WARNING: {unsettled} results are errors. Re-run to retry them.")


if __name__ == "__main__":
    main()
