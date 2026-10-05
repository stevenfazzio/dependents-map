"""Scrape a target's dependents from GitHub's dependency-graph pages.

GitHub has no API for dependents, so this walks the HTML list at
/<repo>/network/dependents, 30 rows a page, following each page's "Next" cursor. It
crawls both lists GitHub keeps for a package: PACKAGE and REPOSITORY.

Every fetched page is appended to a JSONL log before the crawl moves on, and a re-run
resumes from the last logged cursor. Once both logs are complete their rows are
written to dependents.parquet.

Usage:
    uv run python pipeline/00_scrape_dependents.py umap-learn
    uv run python pipeline/00_scrape_dependents.py umap-learn --max-pages 20
"""

import argparse
import re
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from urllib.parse import parse_qs, urlencode, urlparse

import pandas as pd
import requests
from bs4 import BeautifulSoup
from config import ROOT, Target, get_target
from storage import append_jsonl, read_jsonl_log, write_parquet_safely

GITHUB = "https://github.com"
DEPENDENT_TYPES = ("PACKAGE", "REPOSITORY")
USER_AGENT = "dependents-map/0.1 (+https://stevenfazzio.github.io)"

# GitHub sends no rate-limit headers on these pages, so the delay is set by trial. Observed
# 2026-10-04: at 1 s it answered 429 with Retry-After: 120 after roughly 40 requests.
REQUEST_DELAY_S = 3.0
RETRY_STATUS = {429, 500, 502, 503, 504}
MAX_RETRIES = 6
EMPTY_PAGE_RETRIES = 3
LOG_EVERY_PAGES = 25


class PageShapeError(RuntimeError):
    """The response was HTML, but not a dependents list we know how to read."""


@dataclass
class Page:
    rows: list[dict]
    next_url: str | None
    stated_counts: dict[str, int]  # GitHub's own (approximate) totals, per dependent type


def fetch_html(session: requests.Session, url: str) -> str:
    for attempt in range(MAX_RETRIES):
        wait = min(2**attempt * 15, 300)
        try:
            resp = session.get(url, timeout=30)
        except (requests.Timeout, requests.ConnectionError) as e:
            reason = type(e).__name__
        else:
            if resp.status_code == 200:
                return resp.text
            if resp.status_code not in RETRY_STATUS:
                resp.raise_for_status()
            reason = f"HTTP {resp.status_code}"
            retry_after = resp.headers.get("Retry-After", "")
            if retry_after.isdigit():
                wait = max(wait, int(retry_after) + 1)
        if attempt < MAX_RETRIES - 1:
            print(f"  {reason}, retrying in {wait}s (attempt {attempt + 1}/{MAX_RETRIES})")
            time.sleep(wait)
    raise RuntimeError(f"Gave up after {MAX_RETRIES} attempts: {url}")


def _count(row, icon_class: str) -> int:
    icon = row.select_one(f"svg.{icon_class}")
    if icon is None:
        raise PageShapeError(f"dependent row has no {icon_class} count")
    return int(icon.parent.get_text(strip=True).replace(",", ""))


def parse_page(html: str) -> Page:
    soup = BeautifulSoup(html, "html.parser")
    box = soup.select_one("#dependents")
    if box is None:
        raise PageShapeError("no #dependents container")

    rows = []
    for row in box.select("div.Box-row"):
        repo_link = row.select_one('a[data-hovercard-type="repository"]')
        if repo_link is None:
            # A package GitHub lists with no linked repository: a bare name, no counts.
            name = row.select_one("span.text-bold")
            if name is None:
                raise PageShapeError("dependent row has neither a repository link nor a name")
            rows.append(
                {
                    "full_name": None,
                    "package": name.get_text(strip=True),
                    "stars": None,
                    "forks": None,
                }
            )
            continue
        package = row.select_one("small")
        rows.append(
            {
                "full_name": repo_link["href"].strip("/"),
                "package": (package.get_text(strip=True) or None) if package else None,
                "stars": _count(row, "octicon-star"),
                "forks": _count(row, "octicon-repo-forked"),
            }
        )

    next_url = None
    for link in box.select(".paginate-container a"):
        if link.get_text(strip=True) == "Next":
            next_url = link["href"]

    stated_counts = {}
    text = box.get_text(" ", strip=True)
    for dependent_type, pattern in (
        ("REPOSITORY", r"([\d,]+)\s+Repositor"),
        ("PACKAGE", r"([\d,]+)\s+Package"),
    ):
        if match := re.search(pattern, text):
            stated_counts[dependent_type] = int(match.group(1).replace(",", ""))

    return Page(rows=rows, next_url=next_url, stated_counts=stated_counts)


def fetch_page(session: requests.Session, url: str) -> Page:
    """Fetch and parse one page, re-fetching if it comes back with no rows.

    An empty page with no Next link is indistinguishable from the end of the list, so
    it is only accepted as the end after the retries also come back empty.
    """
    for attempt in range(EMPTY_PAGE_RETRIES + 1):
        page = parse_page(fetch_html(session, url))
        if page.rows:
            return page
        if attempt < EMPTY_PAGE_RETRIES:
            print(f"  Page had no rows, re-fetching ({attempt + 1}/{EMPTY_PAGE_RETRIES})")
            time.sleep(10 * (attempt + 1))
    if page.next_url is not None:
        raise PageShapeError(f"page stays empty but has a Next link: {url}")
    print(f"  Accepting an empty page as the end of the list: {url}")
    return page


def resolve_package_id(session: requests.Session, target: Target) -> str:
    """Look up the package's id in the dependency-graph page's package selector."""
    html = fetch_html(session, f"{GITHUB}/{target.repo}/network/dependents")
    soup = BeautifulSoup(html, "html.parser")
    available = {}
    for link in soup.select('a[href*="package_id="]'):
        ids = parse_qs(urlparse(link["href"]).query).get("package_id")
        if ids:
            available[link.get_text(strip=True)] = ids[0]
    if target.package not in available:
        raise SystemExit(
            f"{target.repo} lists no package named {target.package!r}. "
            f"Packages on the page: {sorted(available)}"
        )
    return available[target.package]


def crawl(
    session: requests.Session,
    target: Target,
    dependent_type: str,
    package_id: str,
    max_pages: int | None,
) -> bool:
    """Crawl one list from wherever its log left off. Returns True once it is complete."""
    log_path = target.dependents_log(dependent_type)
    pages = read_jsonl_log(log_path)
    n_pages = len(pages)
    n_rows = sum(len(p["rows"]) for p in pages)

    if pages and pages[-1]["next_url"] is None:
        print(f"{dependent_type}: already complete ({n_pages} pages, {n_rows} rows)")
        return True
    if pages:
        url = pages[-1]["next_url"]
        print(f"{dependent_type}: resuming after {n_pages} pages, {n_rows} rows")
    else:
        query = urlencode({"dependent_type": dependent_type, "package_id": package_id})
        url = f"{GITHUB}/{target.repo}/network/dependents?{query}"
        print(f"{dependent_type}: starting")

    fetched = 0
    with open(log_path, "a") as log:
        while url is not None and (max_pages is None or fetched < max_pages):
            page = fetch_page(session, url)
            record = {
                "url": url,
                "fetched_at": datetime.now(UTC).isoformat(timespec="seconds"),
                "next_url": page.next_url,
                "stated_counts": page.stated_counts,
                "rows": page.rows,
            }
            append_jsonl(log, record)

            fetched += 1
            n_pages += 1
            n_rows += len(page.rows)
            url = page.next_url
            if n_pages % LOG_EVERY_PAGES == 0 or url is None:
                stated = page.stated_counts.get(dependent_type, "?")
                print(f"{dependent_type}: {n_pages} pages, {n_rows} rows (GitHub states {stated})")
            if url is not None:
                time.sleep(REQUEST_DELAY_S)

    return url is None


def build_table(target: Target) -> pd.DataFrame:
    records = []
    for dependent_type in DEPENDENT_TYPES:
        position = 0
        for page in read_jsonl_log(target.dependents_log(dependent_type)):
            for row in page["rows"]:
                records.append(
                    {
                        "dependent_type": dependent_type,
                        **row,
                        "list_position": position,
                        "fetched_at": page["fetched_at"],
                    }
                )
                position += 1
    df = pd.DataFrame.from_records(records)
    df["stars"] = df["stars"].astype("Int64")
    df["forks"] = df["forks"].astype("Int64")
    df["fetched_at"] = pd.to_datetime(df["fetched_at"])
    return df


def report(target: Target, df: pd.DataFrame) -> None:
    for dependent_type in DEPENDENT_TYPES:
        sub = df[df["dependent_type"] == dependent_type]
        stated = read_jsonl_log(target.dependents_log(dependent_type))[-1]["stated_counts"]
        with_repo = sub["full_name"].notna()
        print(
            f"{dependent_type}: {len(sub)} rows scraped (GitHub states "
            f"{stated.get(dependent_type, '?')}), {with_repo.sum()} with a repository "
            f"({sub.loc[with_repo, 'full_name'].nunique()} unique), "
            f"{(~with_repo).sum()} without"
        )
    print(f"Unique repositories across both lists: {df['full_name'].nunique()}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("target", help="target slug from config.TARGETS")
    parser.add_argument(
        "--max-pages", type=int, help="stop each list after this many new pages (for testing)"
    )
    args = parser.parse_args()
    target = get_target(args.target)

    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT
    package_id = resolve_package_id(session, target)

    complete = [
        crawl(session, target, dependent_type, package_id, args.max_pages)
        for dependent_type in DEPENDENT_TYPES
    ]
    if not all(complete):
        print("Crawl incomplete; re-run to resume. No parquet written.")
        return

    df = build_table(target)
    write_parquet_safely(df, target.dependents_parquet)
    report(target, df)
    print(f"Wrote {len(df)} rows to {target.dependents_parquet.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
