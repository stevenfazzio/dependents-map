"""Choose the candidates for the map: one document per dependent project.

Stage 06 decides which of these go on the map, once stages 04 and 05 have looked at what
each project does with the package.

Reads what the earlier stages fetched and applies the selection rules in order,
printing how many rows each rule removes:

  1. Repositories GitHub could not find are dropped.
  2. A repository listed under two names counts once.
  3. Forks are represented by their parent. A fork is kept only if it has its own
     following, or if its parent is not itself a listed dependent, in which case the
     most-starred fork stands in for the project.
  4. The README must have at least MIN_TEXT_CHARS characters.
  5. Repositories with byte-identical READMEs collapse to the most-starred copy.
  6. Packages with no usable repository are added from their PyPI description.

Usage:
    uv run python pipeline/03_select_corpus.py umap-learn
"""

import argparse
import hashlib
import re

import pandas as pd
from config import ROOT, Target, get_target
from storage import write_parquet_safely

MIN_TEXT_CHARS = 200
# A fork with at least this many stars has its own following and is kept as its own project.
FORK_KEEP_STARS = 10

REPO_COLUMNS = [
    "repo_id",
    "description",
    "homepage_url",
    "stars",
    "forks",
    "is_fork",
    "parent",
    "is_archived",
    "created_at",
    "pushed_at",
    "language",
    "license",
    "owner_type",
    "topics",
    "root_files",
]


class Funnel:
    """Prints the row count after each selection rule, so no drop goes unexplained."""

    def __init__(self, label: str, n: int):
        self.n = n
        print(f"{label}: {n}")

    def step(self, df: pd.DataFrame, reason: str) -> pd.DataFrame:
        change = len(df) - self.n
        print(f"  {self.n:>6} → {len(df):>6} ({change:+d}: {reason})")
        self.n = len(df)
        return df


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def normalize_package_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def dependency_kind(requires_dist: list[str], package: str) -> str:
    """How a package's latest release depends on `package`: required, optional or absent."""
    target = normalize_package_name(package)
    hits = []
    for requirement in requires_dist:
        match = re.match(r"\s*([A-Za-z0-9][A-Za-z0-9._-]*)", requirement)
        if match and normalize_package_name(match.group(1)) == target:
            hits.append(requirement)
    if not hits:
        return "absent"
    return "optional" if all("extra ==" in hit for hit in hits) else "required"


def strongest(kinds: list[str]) -> str | None:
    for kind in ("required", "optional", "absent"):
        if kind in kinds:
            return kind
    return None


def select_repositories(
    target: Target, dependents: pd.DataFrame, pypi: pd.DataFrame
) -> pd.DataFrame:
    repos = pd.read_parquet(target.repos_parquet)
    repos["key"] = repos["full_name"].str.lower()

    # What each requested name was listed as, and the package names attached to it.
    packages: dict[str, set[str]] = {}
    pypi_rows: dict[str, list] = {}
    package_rows = dependents[
        (dependents["dependent_type"] == "PACKAGE") & dependents["full_name"].notna()
    ]
    for row in package_rows.itertuples():
        names = packages.setdefault(row.full_name.lower(), set())
        if row.package:
            names.add(row.package)
    for row in pypi[pypi["github_repo"].notna()].itertuples():
        packages.setdefault(row.github_repo.lower(), set()).add(row.pypi_name)
        pypi_rows.setdefault(row.github_repo.lower(), []).append(row)
    in_repository_list = set(
        dependents.loc[dependents["dependent_type"] == "REPOSITORY", "full_name"].str.lower()
    )
    in_package_list = set(package_rows["full_name"].str.lower())

    repos["listed_as_repository"] = repos["key"].isin(in_repository_list)
    repos["listed_as_package"] = repos["key"].isin(in_package_list)
    repos["is_package"] = repos["key"].isin(packages.keys())
    repos["packages"] = repos["key"].map(lambda k: sorted(packages.get(k, ())))
    repos["pypi_summary"] = repos["key"].map(
        lambda k: next((r.summary for r in pypi_rows.get(k, []) if r.summary), None)
    )
    repos["target_dependency"] = repos["key"].map(
        lambda k: strongest(
            [dependency_kind(list(r.requires_dist), target.package) for r in pypi_rows.get(k, [])]
        )
    )

    funnel = Funnel("Repositories looked up", len(repos))
    found = funnel.step(repos[repos["status"] == "ok"].copy(), "not found on GitHub")

    # Rule 2: one row per repository, pooling what its several names were listed as.
    pooled = found.groupby("repo_id").agg(
        listed_as_repository=("listed_as_repository", "any"),
        listed_as_package=("listed_as_package", "any"),
        is_package=("is_package", "any"),
        packages=("packages", lambda lists: sorted(set().union(*lists))),
        pypi_summary=("pypi_summary", "first"),
        target_dependency=("target_dependency", lambda kinds: strongest(list(kinds))),
    )
    one = found.drop_duplicates("repo_id").drop(columns=pooled.columns)
    one = funnel.step(one.join(pooled, on="repo_id"), "listed under more than one name")
    one["key"] = one["resolved_name"].str.lower()
    one["is_fork"] = one["is_fork"].astype(bool)

    # Rule 3: forks.
    parent_key = one["parent"].str.lower().fillna(one["key"])
    forks_per_parent = parent_key[one["is_fork"]].value_counts()
    parent_listed = parent_key.isin(set(one.loc[~one["is_fork"], "key"]))
    stand_ins = (
        one[one["is_fork"] & ~parent_listed]
        .assign(parent_key=parent_key)
        .sort_values(["stars", "pushed_at"], ascending=False)
        .drop_duplicates("parent_key")
    )
    own_following = one["is_fork"] & (one["stars"] >= FORK_KEEP_STARS)
    keep = ~one["is_fork"] | own_following | one.index.isin(stand_ins.index)
    one["n_listed_forks"] = 0
    originals = ~one["is_fork"]
    one.loc[originals, "n_listed_forks"] = (
        one.loc[originals, "key"].map(forks_per_parent).fillna(0).astype(int)
    )
    one.loc[stand_ins.index, "n_listed_forks"] = (
        stand_ins["parent_key"].map(forks_per_parent).astype(int) - 1
    )
    n_stand_ins = (~own_following[stand_ins.index]).sum()
    kept = funnel.step(
        one[keep].copy(),
        f"forks represented by their parent or by another fork; kept {own_following.sum()} "
        f"with {FORK_KEEP_STARS}+ stars and {n_stand_ins} standing in for an unlisted parent",
    )

    # Rule 4: enough README to say what the project is.
    kept["text"] = kept["readme"].fillna("").str.strip()
    long_enough = kept["text"].str.len() >= MIN_TEXT_CHARS
    no_text = (kept["text"].str.len() == 0).sum()
    kept = funnel.step(
        kept[long_enough].copy(),
        f"README shorter than {MIN_TEXT_CHARS} characters, {no_text} of them missing or empty",
    )

    # Rule 5: identical READMEs are copies of one project.
    kept["text_hash"] = kept["text"].map(text_hash)
    kept = kept.sort_values(["stars", "created_at"], ascending=[False, True])
    kept["n_identical_copies"] = kept.groupby("text_hash")["text_hash"].transform("size") - 1
    kept = funnel.step(kept.drop_duplicates("text_hash").copy(), "README identical to a kept one")

    kept["doc_id"] = "github:" + kept["resolved_name"]
    kept["source"] = "github"
    kept["name"] = kept["resolved_name"]
    kept["url"] = "https://github.com/" + kept["resolved_name"]
    kept["text_origin"] = kept["readme_name"]
    return kept


def select_pypi_only(
    target: Target, pypi: pd.DataFrame, repositories: pd.DataFrame
) -> pd.DataFrame:
    """Packages not represented by a selected repository, taken from their PyPI description."""
    repos = pd.read_parquet(target.repos_parquet, columns=["full_name", "repo_id"])
    repo_id = dict(zip(repos["full_name"].str.lower(), repos["repo_id"], strict=True))
    selected_ids = set(repositories["repo_id"])

    on_pypi = pypi[pypi["status"] == "ok"].copy()
    funnel = Funnel("Packages found on PyPI", len(on_pypi))
    represented = on_pypi["github_repo"].str.lower().map(repo_id).isin(selected_ids)
    rest = funnel.step(on_pypi[~represented].copy(), "represented by a selected repository")

    rest["text"] = rest["description"].fillna("").str.strip()
    rest = funnel.step(
        rest[rest["text"].str.len() >= MIN_TEXT_CHARS].copy(),
        f"PyPI description shorter than {MIN_TEXT_CHARS} characters",
    )
    rest["text_hash"] = rest["text"].map(text_hash)
    fresh = ~rest["text_hash"].isin(set(repositories["text_hash"]))
    rest = funnel.step(rest[fresh].copy(), "description identical to a selected README")
    rest = funnel.step(
        rest.drop_duplicates("text_hash").copy(), "description identical to another package's"
    )

    rest["doc_id"] = "pypi:" + rest["pypi_name"]
    rest["source"] = "pypi"
    rest["name"] = rest["pypi_name"]
    rest["url"] = "https://pypi.org/project/" + rest["pypi_name"] + "/"
    rest["text_origin"] = "PyPI description"
    rest["is_package"] = True
    rest["listed_as_package"] = True
    rest["listed_as_repository"] = False
    rest["packages"] = rest["pypi_name"].map(lambda name: [name])
    rest["pypi_summary"] = rest["summary"]
    rest["target_dependency"] = rest["requires_dist"].map(
        lambda reqs: dependency_kind(list(reqs), target.package)
    )
    rest["pypi_latest_upload_at"] = rest["latest_upload_at"]
    rest["n_listed_forks"] = 0
    rest["n_identical_copies"] = 0
    return rest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("target", help="target slug from config.TARGETS")
    args = parser.parse_args()
    target = get_target(args.target)

    dependents = pd.read_parquet(target.dependents_parquet)
    pypi = pd.read_parquet(target.pypi_parquet)

    repositories = select_repositories(target, dependents, pypi)
    pypi_only = select_pypi_only(target, pypi, repositories)

    columns = [
        "doc_id",
        "source",
        "name",
        "url",
        "text",
        "text_origin",
        "is_package",
        "listed_as_repository",
        "listed_as_package",
        "packages",
        "pypi_summary",
        "target_dependency",
        "n_listed_forks",
        "n_identical_copies",
        *REPO_COLUMNS,
        "pypi_latest_upload_at",
    ]
    documents = pd.concat([repositories, pypi_only], ignore_index=True).reindex(columns=columns)
    documents = documents.sort_values("doc_id").reset_index(drop=True)
    documents["is_fork"] = documents["is_fork"].astype("boolean")
    documents["is_archived"] = documents["is_archived"].astype("boolean")

    assert documents["doc_id"].is_unique, "duplicate doc_id"
    assert documents["text"].str.len().min() >= MIN_TEXT_CHARS
    print(
        f"Selected {len(documents)} documents: {len(repositories)} repositories + "
        f"{len(pypi_only)} PyPI-only packages; {documents['is_package'].sum()} are packages"
    )
    write_parquet_safely(documents, target.candidates_parquet)
    print(f"Wrote {target.candidates_parquet.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
