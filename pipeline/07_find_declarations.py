"""Find where each dependent names the target package, and whether it asked for it.

GitHub lists a repository as a dependent when a dependency file in it names the package,
and lock files count. A lock file names everything that gets installed, so a project can
be listed for a package that one of its own dependencies pulled in. This stage reads each
repository's dependency files and sorts it into:

  declared    a file the project wrote names the package, or a lock file records it as
              the project's own requirement
  lock only   only lock files name it; what the lock file says required it is kept as `via`
  not found   no dependency file names it

Three passes, each logged and resumable like stage 02's:
  1. The dependency files in the repository root, whose names stage 02 already listed.
  2. For repositories the root did not settle, the full file listing (one REST call each).
  3. For those, the dependency files outside the root, plus a raw copy of any file
     GraphQL would not return whole: it cuts text off at about 512 KB, which lock files
     exceed, and gives no text for UTF-16.

Packages known only from PyPI are sorted by the requirement in their PyPI metadata.

Usage:
    uv run python pipeline/07_find_declarations.py umap-learn --limit 200
    uv run python pipeline/07_find_declarations.py umap-learn
"""

import argparse
import json
import re
import time
import tomllib
from collections import Counter
from importlib import import_module

import pandas as pd
import requests
from config import ROOT, Target, get_target
from storage import write_parquet_safely

# Stage 02's GitHub client: retries, rate-limit waits, batch splitting and the pass log.
# Its file name starts with a digit, so a plain import statement cannot name it.
github = import_module("02_fetch_repos")

ROOT_BATCH_SIZE = 40
REMAINING_BATCH_SIZE = 10
# GraphQL returns up to about 512 KB of text per file, so a query is sized by weight.
QUERY_WEIGHT = 80
LOCK_FILE_WEIGHT = 10
MAX_NESTED_FILES = 300
MAX_RAW_BYTES = 30_000_000
# Text is kept in the log only for files that name the package and are this small, so
# the reading below can be redone without fetching again.
KEEP_TEXT_CHARS = 200_000

DEPENDENCY_FILE = re.compile(
    r"""(^|/)(
        [^/]*(require|reqs|constraint)[^/]*\.(txt|in|lock)
      | (requirements?|reqs)/[^/]+\.(txt|in)
      | pyproject\.toml | setup\.py | setup\.cfg | pipfile | pipfile\.lock
      | poetry\.lock | uv\.lock | pdm\.lock | pixi\.toml | pixi\.lock
      | [^/]*(environment|conda)[^/]*\.ya?ml | env[^/]*\.ya?ml | meta\.ya?ml
    )$""",
    re.IGNORECASE | re.VERBOSE,
)
# Committed environments and other people's code: not this project's dependency files.
VENDORED = re.compile(
    r"(^|/)(site-packages|dist-packages|node_modules|\.?venv[^/]*|\.?env|__pycache__|"
    r"\.ipynb_checkpoints|lib/python[\d.]+|third_party|vendor)(/|$)",
    re.IGNORECASE,
)
TOML_LOCKS = {"uv.lock", "poetry.lock", "pdm.lock"}
TOML_MANIFESTS = {"pyproject.toml", "pipfile", "pixi.toml"}
SETUP_SECTIONS = {
    "install_requires": "required",
    "extras_require": "optional",
    "tests_require": "dev",
    "setup_requires": "dev",
}
SCOPES = ["required", "optional", "dev"]  # strongest first

STATUSES = ["declared", "lock only", "not found", "unread"]


def normalize(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def name_pattern(target: Target) -> re.Pattern:
    """Matches the package as a requirement, or its repository as a direct URL requirement."""
    name = "[-_.]".join(re.escape(part) for part in re.split(r"[-_.]+", target.package))
    return re.compile(
        rf"(?<![\w.-]){name}(?!\w|-[A-Za-z])|github\.com[/:]{re.escape(target.repo)}(?![\w-])",
        re.IGNORECASE,
    )


def is_lock_name(base: str) -> bool:
    return base in TOML_LOCKS or bool(re.search(r"lock|freeze|frozen|constraint", base))


def strip_comment(line: str) -> str:
    return re.sub(r"(^|\s)#.*$", "", line)


def via_notes(lines: list[str], i: int) -> list[str]:
    """What a compiled requirements file says required the entry on line i."""
    notes = []
    if inline := re.search(r"#\s*via\b(.*)$", lines[i]):
        notes.append(inline.group(1))
    for line in lines[i + 1 :]:
        comment = re.match(r"\s+#(.*)$", line)
        if not comment:
            break
        notes.append(re.sub(r"^\s*via\b", "", comment.group(1)))
    return [item.strip() for note in notes for item in note.split(",") if item.strip()]


def read_lines(base: str, text: str, pattern: re.Pattern) -> dict:
    """Read a requirements file, a conda environment, setup.py or setup.cfg line by line."""
    lines = text.splitlines()
    hits = [i for i, line in enumerate(lines) if pattern.search(strip_comment(line))]
    if not hits:
        return {"named": False}
    out = {"named": True, "direct": True, "lines": [lines[i].strip()[:200] for i in hits[:5]]}

    if base.startswith("setup."):
        before = "\n".join(lines[: hits[0] + 1])
        sections = re.findall("|".join(SETUP_SECTIONS), before)
        out["scope"] = SETUP_SECTIONS[sections[-1]] if sections else "required"
        return out

    if re.search(r"dev|test|doc|lint|bench", base):
        out["scope"] = "dev"
    elif re.search(r"optional|extra", base):
        out["scope"] = "optional"
    else:
        out["scope"] = "required"

    # How much the file looks like a dump of an environment rather than a written list.
    if base.endswith((".yml", ".yaml")):
        entries = [m.group(1) for line in lines if (m := re.match(r"\s*-\s+(\S.*)", line))]
        pinned = [e for e in entries if "=" in e]
    else:
        stripped = (strip_comment(line).strip() for line in lines)
        entries = [s for s in stripped if s and not re.match(r"-{1,2}[A-Za-z]", s)]
        pinned = [e for e in entries if "==" in e or " @ " in e]
    listed = " ".join(entries).lower()
    out["entries"] = len(entries)
    out["pinned"] = len(pinned)
    # Nobody lists these by hand: they are what the package itself pulls in.
    out["lists_pynndescent"] = bool(re.search(r"(?<![\w.-])pynndescent\b", listed))
    out["lists_llvmlite"] = bool(re.search(r"(?<![\w.-])llvmlite\b", listed))

    compiled = "autogenerated by" in text[:600].lower()
    if compiled or is_lock_name(base):
        notes = [n for i in hits for n in via_notes(lines, i) if not n.startswith("-c")]
        own = r"^-r\b|\.in\b|\.txt\b|pyproject\.toml|setup\.(py|cfg)"
        if any(re.search(own, note) for note in notes):
            out["direct"] = True
        elif notes:
            out["direct"] = False
            out["via"] = sorted({normalize(note.split()[0]) for note in notes})
        else:
            out["direct"] = None  # a lock file that does not say what required each entry
    return out


def declaration_scope(path: tuple[str, ...]) -> str | None:
    """How a TOML key path declares a dependency, or None if it does not declare one."""
    joined = ".".join(path).lower()
    if re.search(r"(^|\.)(sources?|constraint-dependencies|override-dependencies)(\.|$)", joined):
        return None
    if not re.search(r"dependenc|requires|packages|extras", joined):
        return None
    if re.search(r"optional|extras|feature", joined):
        return "optional"
    if re.search(r"dev|group|test|doc|build-system|envs", joined):
        return "dev"
    return "required"


def read_toml_manifest(text: str, target: Target, pattern: re.Pattern) -> dict:
    """Read pyproject.toml, Pipfile or pixi.toml: every key path that names the package."""
    wanted = normalize(target.package)
    found: dict[str, str] = {}

    def walk(node, path: tuple[str, ...]) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                if normalize(key) == wanted and (scope := declaration_scope(path)):
                    optional = isinstance(value, dict) and value.get("optional")
                    found[".".join(path)] = "optional" if optional else scope
                walk(value, (*path, key))
        elif isinstance(node, list):
            for item in node:
                walk(item, path)
        elif isinstance(node, str) and pattern.search(node):
            if scope := declaration_scope(path):
                found[".".join(path)] = scope

    walk(tomllib.loads(text), ())
    if not found:
        return {"named": False}
    scope = min(found.values(), key=SCOPES.index)
    return {"named": True, "direct": True, "scope": scope, "where": sorted(found)}


def requirement_names(package: dict) -> set[str]:
    """Everything one lock-file entry depends on, in uv's, Poetry's or PDM's layout."""
    names = set()

    def add(item) -> None:
        if isinstance(item, dict) and "name" in item:  # uv
            names.add(normalize(item["name"]))
        elif isinstance(item, str):  # PDM: requirement strings
            if match := re.match(r"\s*([A-Za-z0-9][A-Za-z0-9._-]*)", item):
                names.add(normalize(match.group(1)))

    dependencies = package.get("dependencies", [])
    if isinstance(dependencies, dict):  # Poetry: a table keyed by name
        names |= {normalize(key) for key in dependencies}
    else:
        for item in dependencies:
            add(item)
    for group in ("optional-dependencies", "dev-dependencies"):
        for items in package.get(group, {}).values():
            for item in items:
                add(item)
    return names


def read_toml_lock(text: str, target: Target) -> dict:
    """Read uv.lock, poetry.lock or pdm.lock: which locked packages require the target."""
    wanted = normalize(target.package)
    packages = [p for p in tomllib.loads(text).get("package", []) if isinstance(p, dict)]
    if not any(normalize(p.get("name", "")) == wanted for p in packages):
        return {"named": False}
    parents = [p for p in packages if wanted in requirement_names(p)]
    # The project itself and its workspace members, which uv locks beside what they install.
    own = [p for p in parents if {"editable", "virtual", "directory"} & set(p.get("source", {}))]
    via = sorted({normalize(p["name"]) for p in parents if p not in own})
    # Locked but required by nothing else in the file: the project asked for it.
    return {"named": True, "direct": bool(own) or not parents, "via": via}


def analyze(target: Target, path: str, text: str) -> dict:
    """What one dependency file says about the target package."""
    pattern = name_pattern(target)
    if not pattern.search(text):
        return {"named": False}
    base = path.rsplit("/", 1)[-1].lower()
    try:
        if base in TOML_LOCKS:
            return read_toml_lock(text, target)
        if base in TOML_MANIFESTS:
            return read_toml_manifest(text, target, pattern)
        if base == "pipfile.lock":
            sections = json.loads(text)
            wanted = normalize(target.package)
            named = any(
                normalize(key) == wanted
                for section in ("default", "develop")
                for key in sections.get(section, {})
            )
            return {"named": named, "direct": None}  # Pipfile.lock records no requirers
    except (tomllib.TOMLDecodeError, json.JSONDecodeError, AttributeError, TypeError):
        pass  # malformed, or cut short: fall back to reading it as lines
    out = read_lines(base, text, pattern)
    if out["named"] and base in TOML_LOCKS | {"pipfile.lock"}:
        out |= {"direct": None, "unparsed": True}
    return out


def file_record(
    target: Target, path: str, size: int, oid: str, text: str | None, whole: bool
) -> dict:
    record = {"path": path, "size": size, "oid": oid, "whole": whole, "named": False}
    if text is not None:
        record |= analyze(target, path, text)
        if record["named"] and len(text) <= KEEP_TEXT_CHARS:
            record["text"] = text
    return record


def has_direct(files: list[dict]) -> bool:
    return any(f.get("direct") is True for f in files)


def fetch_blobs(token: str, wanted: list[tuple[str, str, str, int]]) -> tuple[dict, dict]:
    """Fetch blobs through GraphQL, given (repository, key, object selector, weight).

    Returns ({(repository, key): blob}, {repository: error}) with a blob only for the
    selectors that resolved to a file.
    """
    chunks, weight = [[]], 0
    for item in wanted:
        if chunks[-1] and weight + item[3] > QUERY_WEIGHT:
            chunks.append([])
            weight = 0
        chunks[-1].append(item)
        weight += item[3]

    blobs, repo_errors = {}, {}
    for chunk in chunks:
        if not chunk:
            continue
        by_repo: dict[str, list[tuple[str, str]]] = {}
        for name, key, selector, _ in chunk:
            by_repo.setdefault(name, []).append((key, selector))
        aliases = []
        for i, (name, selectors) in enumerate(by_repo.items()):
            owner, repo = name.split("/", 1)
            objects = " ".join(
                f"f{j}: object({selector}) {{ ...File }}"
                for j, (_, selector) in enumerate(selectors)
            )
            aliases.append(
                f"  r{i}: repository(owner: {json.dumps(owner)}, name: {json.dumps(repo)}) "
                f"{{ {objects} }}"
            )
        query = (
            "query {\n" + "\n".join(aliases) + "\n}\n"
            "fragment File on Blob { oid byteSize isBinary isTruncated text }"
        )
        data, errors = github.graphql(token, query)
        failed = {e["path"][0]: e.get("type") or "unknown" for e in errors if len(e["path"]) == 1}
        for i, (name, selectors) in enumerate(by_repo.items()):
            node = data.get(f"r{i}")
            if node is None or f"r{i}" in failed:
                repo_errors[name] = failed.get(f"r{i}", "null without an error")
                continue
            for j, (key, _) in enumerate(selectors):
                # None: no such path. Empty: the path is a directory.
                if blob := node.get(f"f{j}"):
                    blobs[name, key] = blob
    return blobs, repo_errors


def record_from_blob(target: Target, path: str, blob: dict) -> dict:
    whole = not blob["isTruncated"] and blob["text"] is not None
    return file_record(target, path, blob["byteSize"], blob["oid"], blob["text"], whole)


def repo_error(error: str) -> dict:
    if error == "NOT_FOUND":
        return {"status": "not_found"}
    return {"status": "error", "error": error}


def fetch_root_files(
    token: str, target: Target, candidates: dict[str, list[str]], names: list[str]
) -> dict[str, dict]:
    wanted = []
    for name in names:
        for path in candidates[name]:
            selector = f"expression: {json.dumps('HEAD:' + path, ensure_ascii=False)}"
            weight = LOCK_FILE_WEIGHT if is_lock_name(path.lower()) else 1
            wanted.append((name, path, selector, weight))
    blobs, errors = fetch_blobs(token, wanted)
    results = {}
    for name in names:
        if name in errors:
            results[name] = repo_error(errors[name])
            continue
        files = [
            record_from_blob(target, path, blobs[name, path])
            for path in candidates[name]
            if (name, path) in blobs
        ]
        results[name] = {"status": "ok", "files": files}
    return results


def rest_get(token: str, path: str, accept: str) -> requests.Response | None:
    """GET from the REST API, waiting out rate limits. None when the retries run out."""
    for attempt in range(github.MAX_RETRIES):
        resp = None
        try:
            resp = github._session(token).get(
                f"{github.REST_URL}{path}", headers={"Accept": accept}, timeout=120
            )
        except (
            requests.Timeout,
            requests.ConnectionError,
            requests.exceptions.ChunkedEncodingError,
        ):
            pass
        else:
            if resp.status_code not in github.RETRY_STATUS:
                return resp
        if attempt < github.MAX_RETRIES - 1:
            time.sleep(github._backoff(attempt, resp))
    return None


def fetch_tree(token: str, target: Target, name: str) -> dict:
    """List a repository's files, keeping the dependency files outside the root."""
    resp = rest_get(
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
    blobs = [e for e in body["tree"] if e["type"] == "blob"]
    nested = [
        e
        for e in blobs
        if "/" in e["path"] and DEPENDENCY_FILE.search(e["path"]) and not VENDORED.search(e["path"])
    ]
    nested.sort(key=lambda e: (e["path"].count("/"), e["path"]))
    installed = normalize(target.package).replace("-", "_")
    vendored_copy = re.compile(rf"(^|/){installed}-[^/]+\.(dist|egg)-info/", re.IGNORECASE)
    return {
        "status": "ok",
        "truncated": body["truncated"],
        "n_blobs": len(blobs),
        "n_nested": len(nested),
        "vendored_copy": any(vendored_copy.search(e["path"]) for e in blobs),
        "files": [[e["path"], e.get("size", 0), e["sha"]] for e in nested[:MAX_NESTED_FILES]],
    }


def decode(raw: bytes) -> str:
    # PowerShell's `pip freeze > requirements.txt` writes UTF-16, which GitHub calls binary.
    if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):
        return raw.decode("utf-16", errors="replace")
    return raw.decode("utf-8-sig", errors="replace")


def fetch_remaining(
    token: str, target: Target, trees: dict, root: dict, names: list[str]
) -> dict[str, dict]:
    """Read the dependency files outside the root, then raw copies of the incomplete ones."""
    wanted = [
        (name, path, f'oid: "{sha}"', 1 + min(size, 512_000) // 50_000)
        for name in names
        for path, size, sha in trees[name]["files"]
    ]
    blobs, errors = fetch_blobs(token, wanted)
    results = {}
    for name in names:
        if name in errors:
            results[name] = repo_error(errors[name])
            continue
        nested = [
            record_from_blob(target, path, blobs[name, path])
            for path, _, _ in trees[name]["files"]
            if (name, path) in blobs
        ]
        files = {f["path"]: f for f in root[name]["files"]} | {f["path"]: f for f in nested}
        result = {"status": "ok", "files": nested, "reread": []}
        # Raw copies cost a REST call each, so they stop once a file settles the repository.
        partial = [f for f in files.values() if not f["whole"] and f["size"] <= MAX_RAW_BYTES]
        partial.sort(key=lambda f: (is_lock_name(f["path"].rsplit("/", 1)[-1].lower()), f["path"]))
        for f in partial:
            if has_direct(list(files.values())):
                break
            resp = rest_get(
                token, f"/repos/{name}/git/blobs/{f['oid']}", "application/vnd.github.raw+json"
            )
            if resp is None or resp.status_code != 200:
                result = {"status": "error", "error": f"raw copy of {f['path']} failed"}
                break
            reread = file_record(target, f["path"], f["size"], f["oid"], decode(resp.content), True)
            files[f["path"]] = reread
            result["reread"].append(reread)
        results[name] = result
    return results


def classify(target: Target, files: list[dict]) -> dict:
    """Sort one repository by what its dependency files say."""
    # Kept text is read again, so a change to the reading above needs no new fetch.
    files = [
        file_record(target, f["path"], f["size"], f["oid"], f["text"], f["whole"])
        if "text" in f
        else f
        for f in files
    ]
    named = [f for f in files if f["named"]]
    direct = [f for f in named if f.get("direct") is True]
    locks = [f for f in named if f.get("direct") is not True]
    unread = [f for f in files if not f["whole"]]
    if direct:
        status = "declared"
    elif named:
        status = "lock only"
    elif unread:
        status = "unread"
    else:
        status = "not found"

    listings = [f for f in direct if "entries" in f]
    longest = max(listings, key=lambda f: f["entries"], default={})
    return {
        "status": status,
        "declared_in": [f["path"] for f in direct],
        "locked_in": [f["path"] for f in locks],
        "via": sorted({v for f in locks for v in f.get("via", [])}),
        "scope": min((f["scope"] for f in direct if "scope" in f), key=SCOPES.index, default=None),
        # Declared only in requirement listings that also name the package's own
        # dependencies: an environment written out, not a list someone wrote.
        "dumped": bool(direct)
        and len(listings) == len(direct)
        and all(f["lists_llvmlite"] or f["lists_pynndescent"] for f in listings),
        "listing_entries": longest.get("entries"),
        "listing_pinned": longest.get("pinned"),
        "n_files_read": len(files),
        "n_files_unread": len(unread),
    }


def build_table(
    target: Target, documents: pd.DataFrame, root: dict, trees: dict, remaining: dict
) -> pd.DataFrame:
    rows = []
    for row in documents.itertuples():
        out = {"doc_id": row.doc_id, "basis": "repository files"}
        if row.source != "github":
            out["basis"] = "PyPI metadata"
            out["status"] = "not found" if row.target_dependency == "absent" else "declared"
            out["scope"] = None if row.target_dependency == "absent" else row.target_dependency
        elif root.get(row.name, {}).get("status") != "ok":
            out["status"] = "unread"
        else:
            tree = trees.get(row.name, {})
            rest = remaining.get(row.name, {})
            files = {f["path"]: f for f in root[row.name]["files"]}
            if rest.get("status") == "ok":
                files |= {f["path"]: f for f in rest["files"] + rest["reread"]}
            out |= classify(target, list(files.values()))
            # Settled by the root alone, or by every pass having run.
            searched = out["status"] == "declared" or rest.get("status") == "ok"
            if not searched:
                out["status"] = "unread"
            out["searched"] = "root" if not tree else "whole repository"
            out["n_nested_candidates"] = tree.get("n_nested")
            out["listing_truncated"] = tree.get("truncated")
            out["vendored_copy"] = tree.get("vendored_copy")
        rows.append(out)
    table = pd.DataFrame.from_records(rows)
    for column in ("declared_in", "locked_in", "via"):
        table[column] = table[column].map(lambda v: v if isinstance(v, list) else [])
    table["status"] = pd.Categorical(table["status"], STATUSES)
    for column in ("dumped", "listing_truncated", "vendored_copy"):
        table[column] = table[column].astype("boolean")
    counts = ("listing_entries", "listing_pinned", "n_files_read", "n_files_unread")
    for column in (*counts, "n_nested_candidates"):
        table[column] = table[column].astype("Int64")
    return table


def report(table: pd.DataFrame) -> None:
    print(f"\n{len(table)} documents:")
    counts = table.groupby(["basis", "status"], observed=True).size()
    print(counts.to_string())
    repos = table[table["basis"] == "repository files"]
    declared = repos[repos["status"] == "declared"]
    print(f"\nDeclared ({len(declared)}):")
    print(f"  scope: {declared['scope'].value_counts(dropna=False).to_dict()}")
    print(f"  only in a dumped environment: {declared['dumped'].sum()}")
    kinds = Counter(
        p.rsplit("/", 1)[-1].lower() for paths in declared["declared_in"] for p in paths
    )
    print(f"  files naming it: {kinds.most_common(8)}")
    locked = repos[repos["status"] == "lock only"]
    print(f"\nLock only ({len(locked)}):")
    print(f"  say what required it: {(locked['via'].map(len) > 0).sum()}")
    via = Counter(v for names in locked["via"] for v in names)
    print(f"  required by: {via.most_common(15)}")
    kinds = Counter(p.rsplit("/", 1)[-1].lower() for paths in locked["locked_in"] for p in paths)
    print(f"  files naming it: {kinds.most_common(8)}")
    missing = repos[repos["status"] == "not found"]
    print(f"\nNot found ({len(missing)}):")
    print(f"  with no dependency file at all: {(missing['n_files_read'] == 0).sum()}")
    print(f"  with a committed copy of the package: {missing['vendored_copy'].eq(True).sum()}")
    capped = (repos["n_nested_candidates"] > MAX_NESTED_FILES).sum()
    print(f"\nRepositories with more than {MAX_NESTED_FILES} nested dependency files: {capped}")
    print(f"Listings GitHub cut short: {repos['listing_truncated'].eq(True).sum()}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("target", help="target slug from config.TARGETS")
    parser.add_argument("--limit", type=int, help="only a random N repositories (for testing)")
    args = parser.parse_args()
    target: Target = get_target(args.target)
    token = github.github_token()

    documents = pd.read_parquet(target.documents_parquet)
    repos = documents[documents["source"] == "github"]
    if args.limit:
        repos = repos.sample(min(args.limit, len(repos)), random_state=0)
        documents = documents[documents["doc_id"].isin(repos["doc_id"])]
        print(f"--limit: a random {len(repos)} repositories")
    candidates = {
        row.name: [f for f in row.root_files if DEPENDENCY_FILE.search(f)]
        for row in repos.itertuples()
    }
    names = sorted(candidates)
    n_files = sum(len(files) for files in candidates.values())
    print(f"{len(names)} repositories with {n_files} dependency files in their roots")

    root = github.run_pass(
        "root files",
        target.declaration_log("root"),
        names,
        ROOT_BATCH_SIZE,
        lambda batch: fetch_root_files(token, target, candidates, batch),
    )
    read = [n for n in names if root.get(n, {}).get("status") == "ok"]
    unsettled = [n for n in read if not has_direct(root[n]["files"])]
    print(
        f"{len(read) - len(unsettled)} of {len(read)} repositories declare {target.package} "
        f"in a root file; {len(unsettled)} need their full listing"
    )

    trees = github.run_pass(
        "file listings",
        target.declaration_log("tree"),
        unsettled,
        1,
        lambda batch: {batch[0]: fetch_tree(token, target, batch[0])},
    )
    listed = [n for n in unsettled if trees.get(n, {}).get("status") == "ok"]
    remaining = github.run_pass(
        "remaining files",
        target.declaration_log("remaining"),
        listed,
        REMAINING_BATCH_SIZE,
        lambda batch: fetch_remaining(token, target, trees, root, batch),
    )

    table = build_table(target, documents, root, trees, remaining)
    report(table)
    if args.limit:
        print("--limit run: no parquet written.")
        return
    write_parquet_safely(table, target.declarations_parquet)
    print(f"Wrote {len(table)} rows to {target.declarations_parquet.relative_to(ROOT)}")
    if unread := (table["status"] == "unread").sum():
        print(f"WARNING: {unread} repositories could not be read. Re-run to retry them.")


if __name__ == "__main__":
    main()
