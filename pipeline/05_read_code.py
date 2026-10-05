"""Read each dependent's Python source and report what it does with the target library.

Being listed as a dependent says that a dependency file names the package (stage 04), not
that any code uses it. This stage reads every Python file and notebook of every repository
and records, per project, whether the code imports the library, what it calls from it and
with which arguments, which methods it calls on the objects it builds, and which other
libraries' functions of the same name it uses instead. It also notes the projects that
reach the library without naming it, through another library that runs it for them
(config.Carrier).

Three steps, each resumable:
  1. The file listing of every repository, one REST call each.
  2. The source of every Python file and notebook listed, into a store that all targets
     share. The store is keyed by git's hash of the file, so a file that many copies of a
     project have in common is fetched once. Files under about 500 KB come 100 to a
     GraphQL query; larger ones, nearly all of them notebooks with their outputs embedded,
     take a REST call each. Only a notebook's code cells are kept.
  3. The reading (code_usage.py), which works from the store and needs no network.

Committed environments (site-packages, venv and the like) are skipped. They hold other
people's code, the target library's own included.

Steps 1 and 2 take hours on a full corpus, at a pace set by GitHub's hourly quotas: the
run uses all of the account's REST and GraphQL allowance while it lasts.

Usage:
    uv run python pipeline/05_read_code.py umap-learn --limit 200
    uv run python pipeline/05_read_code.py umap-learn
"""

import argparse
import json
import os
import re
import sqlite3
import time
import zlib
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import NamedTuple

import github
import pandas as pd
from code_usage import CELL_BREAK, imported_modules, read_source
from config import DATA_DIR, ROOT, Target, get_target
from storage import write_parquet_safely

STORE_PATH = DATA_DIR / "code_blobs.sqlite"
CODE_SUFFIXES = (".py", ".ipynb")
# GraphQL returns a file's text whole up to about 512 KB and cut short beyond it.
GRAPHQL_MAX_BYTES = 500_000
MAX_BYTES = 50_000_000
FILES_PER_QUERY = 100
BYTES_PER_QUERY = 3_000_000
# Three workers of each kind use up an hour's quota well inside the hour, and stay clear of
# GitHub's limits on how fast one account may ask.
WORKERS = 3
PROGRESS_EVERY_S = 60

# A file with one of these statuses is final; anything else is fetched again on a re-run.
SETTLED = {"ok", "not a notebook", "binary", "too big", "missing"}

USES = [
    "Calls it",
    "Imports it, calls nothing",
    "Another library's",
    "Names it only",
    "No mention",
    "No code read",
]


class File(NamedTuple):
    oid: str  # git's hash of the contents
    repo: str  # one repository that has it
    path: str
    size: int


class SourceStore:
    """Python source by git blob hash, compressed."""

    def __init__(self, path: Path = STORE_PATH):
        self.db = sqlite3.connect(path)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS source (oid TEXT PRIMARY KEY, status TEXT, text BLOB)"
        )

    def statuses(self) -> dict[str, str]:
        return dict(self.db.execute("SELECT oid, status FROM source"))

    def put(self, rows: list[tuple[str, str, bytes | None]]) -> None:
        with self.db:
            self.db.executemany("INSERT OR REPLACE INTO source VALUES (?, ?, ?)", rows)

    def get(self, oids: list[str]) -> dict[str, str]:
        """The stored source of each of these files that has any."""
        found = {}
        for i in range(0, len(oids), 500):
            chunk = oids[i : i + 500]
            marks = ",".join("?" * len(chunk))
            rows = self.db.execute(
                f"SELECT oid, text FROM source WHERE status = 'ok' AND oid IN ({marks})", chunk
            )
            for oid, text in rows:
                found[oid] = zlib.decompress(text).decode("utf-8", errors="replace")
        return found


def fetch_tree(token: str, name: str) -> dict:
    """List a repository's Python files and notebooks, outside any committed environment."""
    resp = github.rest_get(
        token, f"/repos/{name}/git/trees/HEAD?recursive=1", "application/vnd.github+json"
    )
    if resp is None:
        return {"status": "error", "error": "retries exhausted"}
    if resp.status_code == 404:
        return {"status": "not_found"}
    if resp.status_code == 409:  # an empty repository
        return {"status": "missing"}
    if resp.status_code != 200:
        return {"status": "error", "error": f"HTTP {resp.status_code}"}
    body = resp.json()
    # Mode 120000 is a symbolic link, whose contents are a path.
    blobs = [e for e in body["tree"] if e["type"] == "blob" and e.get("mode") != "120000"]
    code = [e for e in blobs if e["path"].lower().endswith(CODE_SUFFIXES) and e.get("size")]
    kept = [e for e in code if not github.VENDORED.search(e["path"])]
    return {
        "status": "ok",
        "truncated": body["truncated"],
        "n_blobs": len(blobs),
        "n_vendored": len(code) - len(kept),
        "files": [[e["path"], e["size"], e["sha"]] for e in kept],
    }


def notebook_code(text: str) -> str | None:
    """A notebook's code cells joined into one text, or None if this is not a notebook."""
    try:
        notebook = json.loads(text)
    except (json.JSONDecodeError, RecursionError):
        return None
    if not isinstance(notebook, dict):
        return None
    cells = notebook.get("cells")
    if cells is None:  # format 3 kept its cells in worksheets, with code under "input"
        sheets = notebook.get("worksheets") or []
        cells = [c for sheet in sheets if isinstance(sheet, dict) for c in sheet.get("cells", [])]
    sources = []
    for cell in cells if isinstance(cells, list) else []:
        if isinstance(cell, dict) and cell.get("cell_type") == "code":
            source = cell.get("source", cell.get("input", ""))
            sources.append("".join(map(str, source)) if isinstance(source, list) else str(source))
    return CELL_BREAK.join(sources)


def as_row(file: File, text: str) -> tuple[str, str, bytes | None]:
    if file.path.lower().endswith(".ipynb"):
        text = notebook_code(text)
        if text is None:
            return file.oid, "not a notebook", None
    return file.oid, "ok", zlib.compress(text.encode("utf-8", errors="replace"))


def fetch_large(token: str, file: File) -> tuple[str, str, bytes | None]:
    resp = github.rest_get(
        token, f"/repos/{file.repo}/git/blobs/{file.oid}", "application/vnd.github.raw+json"
    )
    if resp is None:
        return file.oid, "error", None
    if resp.status_code == 404:
        return file.oid, "missing", None
    if resp.status_code != 200:
        return file.oid, "error", None
    raw = resp.content
    if b"\x00" in raw[:8000] and raw[:2] not in (b"\xff\xfe", b"\xfe\xff"):
        return file.oid, "binary", None
    return as_row(file, github.text_of(raw))


def fetch_small(token: str, files: list[File]) -> list[tuple[str, str, bytes | None]]:
    """Fetch files through one GraphQL query."""
    by_repo: dict[str, list[File]] = {}
    for file in files:
        by_repo.setdefault(file.repo, []).append(file)
    aliases = []
    for i, (name, repo_files) in enumerate(by_repo.items()):
        owner, repo = name.split("/", 1)
        objects = " ".join(
            f'f{j}: object(oid: "{file.oid}") {{ ...Source }}' for j, file in enumerate(repo_files)
        )
        aliases.append(
            f"  r{i}: repository(owner: {json.dumps(owner)}, name: {json.dumps(repo)}) "
            f"{{ {objects} }}"
        )
    query = (
        "query {\n" + "\n".join(aliases) + "\n}\n"
        "fragment Source on Blob { isBinary isTruncated text }"
    )
    data, errors = github.graphql(token, query)
    failed = {e["path"][0]: e.get("type") for e in errors if len(e["path"]) == 1}

    rows = []
    for i, repo_files in enumerate(by_repo.values()):
        node = data.get(f"r{i}")
        for j, file in enumerate(repo_files):
            if node is None or f"r{i}" in failed:
                # A repository deleted since it was listed takes its files with it.
                gone = failed.get(f"r{i}") == "NOT_FOUND"
                rows.append((file.oid, "missing" if gone else "error", None))
            elif not (blob := node.get(f"f{j}")):
                rows.append((file.oid, "missing", None))
            elif blob["isTruncated"]:
                rows.append(fetch_large(token, file))
            elif blob["text"] is None:
                rows.append((file.oid, "binary", None))
            else:
                rows.append(as_row(file, blob["text"]))
    return rows


def fetch_small_with_split(token: str, files: list[File]) -> list[tuple[str, str, bytes | None]]:
    """Run one query, halving it on failure so one bad file can't sink the rest."""
    try:
        return fetch_small(token, files)
    except github.BatchFailed as e:
        if len(files) == 1:
            return [(files[0].oid, "error", None)]
        half = len(files) // 2
        print(f"  Query for {len(files)} files failed ({e}); splitting")
        return [
            *fetch_small_with_split(token, files[:half]),
            *fetch_small_with_split(token, files[half:]),
        ]


def pack_queries(files: list[File]) -> list[list[File]]:
    """Group small files into queries, a repository's files together."""
    queries, size = [[]], 0
    for file in sorted(files, key=lambda f: (f.repo, f.path)):
        if queries[-1] and (
            len(queries[-1]) >= FILES_PER_QUERY or size + file.size > BYTES_PER_QUERY
        ):
            queries.append([])
            size = 0
        queries[-1].append(file)
        size += file.size
    return [q for q in queries if q]


def fetch_sources(token: str, store: SourceStore, files: list[File]) -> None:
    """Fetch every file into the store, small ones and large ones side by side."""
    too_big = [f for f in files if f.size > MAX_BYTES]
    store.put([(f.oid, "too big", None) for f in too_big])
    large = [f for f in files if GRAPHQL_MAX_BYTES < f.size <= MAX_BYTES]
    queries = pack_queries([f for f in files if f.size <= GRAPHQL_MAX_BYTES])
    n_small = sum(len(q) for q in queries)
    print(
        f"  {n_small} files in {len(queries)} GraphQL queries "
        f"(about {len(queries) / 5000:.1f} h of quota), and {len(large)} larger files by REST "
        f"({sum(f.size for f in large) / 1e9:.1f} GB, about {len(large) / 5000:.1f} h of quota); "
        f"{len(too_big)} over {MAX_BYTES / 1e6:.0f} MB skipped"
    )

    total, done, read_bytes = n_small + len(large), 0, 0
    started = last_report = time.monotonic()
    # The two pools draw on separate quotas, so neither waits for the other's hour.
    with (
        ThreadPoolExecutor(WORKERS) as small_pool,
        ThreadPoolExecutor(WORKERS) as large_pool,
    ):
        pending = {
            small_pool.submit(fetch_small_with_split, token, q): sum(f.size for f in q)
            for q in queries
        }
        pending |= {large_pool.submit(lambda f=f: [fetch_large(token, f)]): f.size for f in large}
        while pending:
            finished, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in finished:
                # Popped, so the fetched text is freed once it is in the store.
                read_bytes += pending.pop(future)
                rows = future.result()
                store.put(rows)
                done += len(rows)
            if time.monotonic() - last_report >= PROGRESS_EVERY_S or not pending:
                last_report = time.monotonic()
                minutes = (last_report - started) / 60
                print(
                    f"  sources: {done}/{total} files, {read_bytes / 1e9:.2f} GB read, "
                    f"{minutes:.0f} min",
                    flush=True,
                )


def canonical(name: str) -> str:
    """A class under the module's own name, wherever in the package it was imported from."""
    module, last = name.split(".", 1)[0], name.rsplit(".", 1)[-1]
    return f"{module}.{last}" if last[:1].isupper() else name


def read_repository(target: Target, files: list[tuple[str, str]], sources: dict[str, str]) -> dict:
    """What one repository's code does with the module, from the source of its files."""
    module = target.module
    carriers = [(carrier, re.compile(carrier.pattern)) for carrier in target.carriers]
    carried: set[str] = set()
    modules: set[str] = set()
    names: set[str] = set()
    called: set[str] = set()
    methods: set[str] = set()
    uncertain_methods: set[str] = set()
    attributes: set[str] = set()
    elsewhere: set[str] = set()
    kwargs: dict[str, set] = {}
    calling_files = []
    counts = Counter()
    # The project's own modules and packages, which are not somebody else's library.
    own = {part.removesuffix(".py") for path, _ in files for part in path.split("/")}
    for path, oid in files:
        text = sources.get(oid)
        if text is None:
            counts["unread"] += 1
            continue
        modules |= imported_modules(text)
        carried |= {c.label for c, pattern in carriers if pattern.search(text)}
        if module not in text.lower():
            continue
        counts["naming"] += 1
        try:
            usage = read_source(text, module)
        except Exception as e:  # one unreadable file must not end a run of a million
            counts["failed"] += 1
            print(f"  could not read {path} ({oid}): {type(e).__name__}: {e}")
            continue
        counts["unparsed"] += bool(usage["unparsed"])
        counts["imports"] += usage["imports"]
        counts["star_import"] += usage["star_import"]
        names |= {canonical(name) for name in usage["names"]}
        attributes |= {name.rsplit(".", 1)[-1] for name in usage["attributes"]}
        elsewhere |= {name for name in usage["elsewhere"] if name.split(".")[0] not in own}
        for call in usage["calls"]:
            name = canonical(call["name"])
            called.add(name)
            if name.rsplit(".", 1)[-1][:1].isupper():
                counts["constructor_calls"] += 1
                counts["star_kwargs"] += call["star"]
                for key, value in call["kwargs"].items():
                    # Kept as JSON so 2, 2.0 and "2" stay apart.
                    kwargs.setdefault(key, set()).add(json.dumps(value))
        for method in usage["methods"]:
            if method["uncertain"]:
                uncertain_methods.add(method["method"])
                continue
            methods.add(method["method"])
            if method["method"] in ("fit", "fit_transform"):
                counts["fits_with_target"] += method["args"] >= 2 or "y" in method["kwargs"]
        if usage["calls"] or usage["methods"]:
            calling_files.append(path)

    uncertain_methods -= methods
    # A library that runs the package for its caller: its pattern only counts where the
    # project imports that library.
    through = sorted(c.label for c, _ in carriers if c.label in carried and c.module in modules)
    if called or methods or uncertain_methods:
        use = "Calls it"
    elif counts["imports"]:
        use = "Imports it, calls nothing"
    elif elsewhere or through:
        use = "Another library's"
    elif counts["naming"]:
        use = "Names it only"
    else:
        use = "No mention"
    return {
        "use": use,
        "n_files": len(files),
        "n_files_unread": counts["unread"],
        "n_files_naming": counts["naming"],
        "n_files_part_unparsed": counts["unparsed"],
        "n_files_failed": counts["failed"],
        "imports": bool(counts["imports"]),
        "star_import": bool(counts["star_import"]),
        "called": sorted(called),
        "n_constructor_calls": counts["constructor_calls"],
        "star_kwargs": bool(counts["star_kwargs"]),
        "kwargs": json.dumps({key: sorted(values) for key, values in sorted(kwargs.items())}),
        "methods": sorted(methods),
        "uncertain_methods": sorted(uncertain_methods),
        "fits_with_target": bool(counts["fits_with_target"]),
        "attributes": sorted(attributes),
        "names": sorted(names),
        "elsewhere": sorted(elsewhere),
        "through": through,
        "modules": sorted(modules),
        "calling_files": calling_files[:10],
    }


def build_table(
    target: Target, documents: pd.DataFrame, trees: dict, store: SourceStore
) -> pd.DataFrame:
    rows = []
    started = time.monotonic()
    for n, row in enumerate(documents.itertuples(), start=1):
        tree = trees.get(row.name, {}) if row.source == "github" else {}
        out = {"doc_id": row.doc_id, "use": "No code read"}
        if tree.get("status") == "ok":
            files = [(path, oid) for path, _, oid in tree["files"]]
            sources = store.get(sorted({oid for _, oid in files}))
            out |= read_repository(target, files, sources)
            out["n_files_vendored"] = tree["n_vendored"]
            out["listing_truncated"] = tree["truncated"]
        rows.append(out)
        if n % 1000 == 0:
            minutes = (time.monotonic() - started) / 60
            print(f"  read {n}/{len(documents)} projects, {minutes:.1f} min", flush=True)
    table = pd.DataFrame.from_records(rows)
    table["use"] = pd.Categorical(table["use"], USES)
    lists = ("called", "methods", "uncertain_methods", "attributes", "names", "elsewhere")
    for column in (*lists, "through", "modules", "calling_files"):
        table[column] = table[column].map(lambda v: v if isinstance(v, list) else [])
    for column in ("imports", "star_import", "star_kwargs", "fits_with_target"):
        table[column] = table[column].astype("boolean")
    for column in [c for c in table.columns if c.startswith("n_")]:
        table[column] = table[column].astype("Int64")
    return table


def tally(lists: pd.Series) -> Counter:
    """How many projects each item appears in."""
    return Counter(item for items in lists for item in items)


def report(table: pd.DataFrame, target: Target) -> None:
    print(f"\n{len(table)} documents:")
    print(table["use"].value_counts().reindex(USES).to_string())
    read = table[table["use"] != "No code read"]
    print(
        f"\n{read['n_files'].sum()} files listed in {len(read)} repositories; "
        f"{read['n_files_unread'].sum()} not read, {read['n_files_naming'].sum()} name "
        f"{target.module}, {read['n_files_part_unparsed'].sum()} of those not fully parsed "
        f"and {read['n_files_failed'].sum()} not readable"
    )
    print(
        f"{read['n_files_vendored'].sum()} files skipped in committed environments; "
        f"{read['listing_truncated'].eq(True).sum()} listings GitHub cut short"
    )
    callers = read[read["use"] == "Calls it"]
    print(f"\nAmong the {len(callers)} projects that call it:")
    print(f"  what they call: {tally(callers['called']).most_common(25)}")
    print(f"  methods on its objects: {tally(callers['methods']).most_common(15)}")
    print(f"  less certainly: {tally(callers['uncertain_methods']).most_common(10)}")
    print(f"  attributes read: {tally(callers['attributes']).most_common(10)}")
    print(f"  fit with a target: {callers['fits_with_target'].sum()}")
    print(f"  keyword arguments passed through **: {callers['star_kwargs'].sum()}")
    kwargs = [json.loads(k) for k in callers["kwargs"]]
    print(f"  constructor arguments: {Counter(key for k in kwargs for key in k).most_common(30)}")
    for key in ("n_components", "metric", "n_neighbors", "min_dist", "random_state"):
        values = Counter(value for k in kwargs for value in k.get(key, []))
        print(f"    {key}: {values.most_common(12)}")
    print(f"\nOther libraries' {target.module}: {tally(read['elsewhere']).most_common(20)}")
    print(f"Through a library that runs it unnamed: {tally(read['through']).most_common()}")
    print(f"Imported alongside, by callers: {tally(callers['modules']).most_common(40)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("target", help="target slug from config.TARGETS")
    parser.add_argument("--limit", type=int, help="only a random N repositories (for testing)")
    args = parser.parse_args()
    target: Target = get_target(args.target)
    token = github.token()

    documents = pd.read_parquet(target.candidates_parquet).drop(columns=["text"])
    repos = documents[documents["source"] == "github"]
    if args.limit:
        repos = repos.sample(min(args.limit, len(repos)), random_state=0)
        documents = documents[documents["doc_id"].isin(repos["doc_id"])]
        print(f"--limit: a random {len(repos)} repositories")
    names = sorted(repos["name"])

    trees = github.run_pass(
        "file listings",
        target.code_tree_log,
        names,
        1,
        lambda batch: {batch[0]: fetch_tree(token, batch[0])},
        workers=WORKERS,
    )
    trees = {name: trees[name] for name in names if name in trees}
    listed = {name: tree for name, tree in trees.items() if tree["status"] == "ok"}

    # One fetch per distinct file, from the first repository that has it.
    files: dict[str, File] = {}
    n_listed = 0
    for name, tree in listed.items():
        for path, size, oid in tree["files"]:
            n_listed += 1
            files.setdefault(oid, File(oid, name, path, size))
    store = SourceStore()
    statuses = store.statuses()
    todo = [f for f in files.values() if statuses.get(f.oid) not in SETTLED]
    print(
        f"sources: {n_listed} files listed in {len(listed)} repositories, {len(files)} distinct; "
        f"{len(files) - len(todo)} already stored, {len(todo)} to fetch "
        f"({sum(f.size for f in todo) / 1e9:.1f} GB)"
    )
    if todo:
        try:
            fetch_sources(token, store, todo)
        except KeyboardInterrupt:
            # Worker threads may be asleep waiting for quota; what was fetched is stored.
            print("\nStopped. Re-run to resume.")
            os._exit(130)
    statuses = store.statuses()
    print(f"  stored: {dict(Counter(statuses[oid] for oid in files if oid in statuses))}")

    table = build_table(target, documents, trees, store)
    report(table, target)
    if args.limit:
        print("--limit run: no parquet written.")
        return
    write_parquet_safely(table, target.code_usage_parquet)
    print(f"Wrote {len(table)} rows to {target.code_usage_parquet.relative_to(ROOT)}")
    unsettled = sum(statuses.get(oid) not in SETTLED for oid in files)
    unlisted = sum(tree["status"] == "error" for tree in trees.values()) + len(names) - len(trees)
    if unsettled or unlisted:
        print(
            f"WARNING: {unsettled} files and {unlisted} listings were not fetched. "
            "Re-run to retry them."
        )


if __name__ == "__main__":
    main()
