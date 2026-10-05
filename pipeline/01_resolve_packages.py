"""Look up on PyPI the dependent packages GitHub lists without a repository.

GitHub's PACKAGE list names some dependents only by package name. PyPI usually knows
more: a long description (the package's README), and often a link to the source
repository, which lets the next stage fetch that repository like any other dependent.

Each lookup is appended to a JSONL log, and a re-run fetches only the packages with
no settled result yet.

Usage:
    uv run python pipeline/01_resolve_packages.py umap-learn
    uv run python pipeline/01_resolve_packages.py umap-learn --limit 50
"""

import argparse
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime

import pandas as pd
import requests
from config import ROOT, get_target
from storage import append_jsonl, read_jsonl_log, write_parquet_safely

PYPI_URL = "https://pypi.org/pypi/{package}/json"
USER_AGENT = "dependents-map/0.1 (+https://stevenfazzio.github.io)"
CONCURRENT_REQUESTS = 4
MAX_RETRIES = 4
RETRY_STATUS = {429, 500, 502, 503, 504}
LOG_EVERY = 100

# A result with one of these statuses is final; anything else is fetched again on a re-run.
SETTLED = {"ok", "not_found"}

INFO_FIELDS = (
    "name",
    "version",
    "summary",
    "description",
    "description_content_type",
    "home_page",
    "project_urls",
    "requires_dist",
    "classifiers",
    "keywords",
)

GITHUB_REPO = re.compile(r"github\.com/([A-Za-z0-9][A-Za-z0-9-]*)/([A-Za-z0-9._-]+)", re.I)
# github.com paths whose first segment is not a repository owner.
NOT_AN_OWNER = {"sponsors", "orgs", "users", "apps", "marketplace", "topics"}
# Project-URL labels most likely to point at the source, best first.
URL_LABEL_PRIORITY = ("source", "repository", "code", "github", "homepage", "home")


def fetch_package(package: str) -> dict:
    """Fetch one package's PyPI record, reduced to the fields this project uses."""
    reason = "unknown"
    for attempt in range(MAX_RETRIES):
        wait = min(2**attempt * 5, 60)
        try:
            resp = requests.get(
                PYPI_URL.format(package=package),
                headers={"User-Agent": USER_AGENT},
                timeout=30,
            )
        except (requests.Timeout, requests.ConnectionError) as e:
            reason = type(e).__name__
        else:
            if resp.status_code == 200:
                body = resp.json()
                uploads = [f["upload_time_iso_8601"] for f in body.get("urls", [])]
                return {
                    "status": "ok",
                    "info": {field: body["info"].get(field) for field in INFO_FIELDS},
                    "latest_upload_at": max(uploads) if uploads else None,
                }
            if resp.status_code == 404:
                return {"status": "not_found"}
            if resp.status_code not in RETRY_STATUS:
                return {"status": "error", "error": f"HTTP {resp.status_code}"}
            reason = f"HTTP {resp.status_code}"
            retry_after = resp.headers.get("Retry-After", "")
            if retry_after.isdigit():
                wait = max(wait, int(retry_after) + 1)
        if attempt < MAX_RETRIES - 1:
            time.sleep(wait)
    return {"status": "error", "error": reason}


def github_repo(info: dict) -> str | None:
    """Pick the GitHub repository a package's metadata points at, if any."""
    urls = dict(info.get("project_urls") or {})
    if info.get("home_page"):
        urls.setdefault("Homepage", info["home_page"])

    def priority(label: str) -> int:
        label = label.lower()
        for rank, word in enumerate(URL_LABEL_PRIORITY):
            if word in label:
                return rank
        return len(URL_LABEL_PRIORITY)

    for label in sorted(urls, key=priority):
        match = GITHUB_REPO.search(urls[label] or "")
        if match and match.group(1).lower() not in NOT_AN_OWNER:
            return f"{match.group(1)}/{match.group(2).removesuffix('.git')}"
    return None


def build_table(packages: list[str], results: dict[str, dict]) -> pd.DataFrame:
    rows = []
    for package in packages:
        result = results.get(package, {"status": "unfetched"})
        row = {"package": package, "status": result["status"], "error": result.get("error")}
        if result["status"] == "ok":
            info = result["info"]
            row |= {
                "pypi_name": info["name"],
                "version": info["version"],
                "summary": info["summary"],
                "description": info["description"],
                "description_content_type": info["description_content_type"],
                "github_repo": github_repo(info),
                "requires_dist": info["requires_dist"] or [],
                "classifiers": info["classifiers"] or [],
                "keywords": info["keywords"],
                "latest_upload_at": result["latest_upload_at"],
            }
        rows.append(row)
    df = pd.DataFrame.from_records(rows)
    df["latest_upload_at"] = pd.to_datetime(df["latest_upload_at"])
    return df


def report(df: pd.DataFrame) -> None:
    print(f"Looked up {len(df)} packages: {df['status'].value_counts().to_dict()}")
    ok = df[df["status"] == "ok"]
    has_text = ok["description"].fillna("").str.strip().str.len() > 0
    print(
        f"Of {len(ok)} on PyPI: {has_text.sum()} have a description, "
        f"{ok['github_repo'].notna().sum()} point at a GitHub repository "
        f"({ok['github_repo'].str.lower().nunique()} unique)"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("target", help="target slug from config.TARGETS")
    parser.add_argument("--limit", type=int, help="only the first N packages (for testing)")
    args = parser.parse_args()
    target = get_target(args.target)

    dependents = pd.read_parquet(target.dependents_parquet)
    is_package = dependents["dependent_type"] == "PACKAGE"
    unlinked = dependents[is_package & dependents["full_name"].isna()]
    packages = sorted(unlinked["package"].dropna().unique())
    print(
        f"{is_package.sum()} package rows → {len(packages)} to look up "
        f"({(is_package & dependents['full_name'].notna()).sum()} already name a repository, "
        f"{len(unlinked) - len(packages)} are repeats or unnamed)"
    )
    if args.limit:
        packages = packages[: args.limit]
        print(f"--limit: keeping the first {len(packages)}")

    results = {r["package"]: r for r in read_jsonl_log(target.pypi_log)}
    todo = [p for p in packages if results.get(p, {}).get("status") not in SETTLED]
    print(f"{len(packages) - len(todo)} already settled, {len(todo)} to fetch")

    with (
        open(target.pypi_log, "a") as log,
        ThreadPoolExecutor(max_workers=CONCURRENT_REQUESTS) as pool,
    ):
        futures = {pool.submit(fetch_package, package): package for package in todo}
        for n, future in enumerate(as_completed(futures), start=1):
            fetched_at = datetime.now(UTC).isoformat(timespec="seconds")
            record = {"package": futures[future], "fetched_at": fetched_at, **future.result()}
            append_jsonl(log, record)
            results[record["package"]] = record
            if n % LOG_EVERY == 0 or n == len(todo):
                print(f"{n}/{len(todo)}")

    df = build_table(packages, results)
    report(df)
    if args.limit:
        print("--limit run: no parquet written.")
        return
    write_parquet_safely(df, target.pypi_parquet)
    print(f"Wrote {len(df)} rows to {target.pypi_parquet.relative_to(ROOT)}")
    if errors := (df["status"] == "error").sum():
        print(f"WARNING: {errors} lookups are errors. Re-run to retry them.")


if __name__ == "__main__":
    main()
