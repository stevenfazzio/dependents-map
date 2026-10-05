"""Keep the candidates that show some sign of the package, and leave the rest off the map.

GitHub lists a project as a dependent when any dependency file in it names the package.
For about a third of the candidates that is all there is: the name sits in a dumped
environment or a lock file, and nothing the project itself wrote refers to the package.
A candidate is dropped only when all three of these come up empty:

  1. its code: stage 05 found no mention of the package in any Python file or notebook;
  2. its dependencies: stage 04 found the package in no dependency list the project
     wrote, only in a dumped environment, in a lock file, or nowhere;
  3. its description of itself: the README, description and topics never name the library.

A candidate whose code could not be listed or read in full is kept, since the first check
is then unknown.

Asking for three blanks keeps the rule to a pattern known to be wrong, a `pip freeze` or
a lock file, instead of asking each project to prove that it uses the package. Stage 05
reads only Python on the default branch, and a project dropped by mistake leaves no trace.

Writes documents.parquet, the corpus every later stage works on.

Usage:
    uv run python pipeline/06_filter_corpus.py umap-learn
"""

import argparse
import re

import pandas as pd
from config import ROOT, get_target
from storage import write_parquet_safely

# Stage 05's words for code that never refers to the package, or that was not read.
NOT_IN_CODE = ["No mention", "No code read"]

# The first of these that holds is the reason a candidate is kept.
REASONS = [
    "its code refers to the package",
    "it wrote the package into a dependency list",
    "its README names the library",
    "its code could not be read in full",
]


def flag(series: pd.Series) -> pd.Series:
    """A yes/no column that may have gaps, as plain booleans with the gaps as no."""
    return series.astype("boolean").fillna(False).astype(bool)


def self_description(row) -> str:
    """Everything a project says about itself: what stage 09 shows the model, untruncated."""
    parts = [row.name, row.description, row.pypi_summary, row.text]
    if row.topics is not None:
        parts.extend(row.topics)
    return "\n".join(part for part in parts if isinstance(part, str))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("target", help="target slug from config.TARGETS")
    args = parser.parse_args()
    target = get_target(args.target)

    candidates = pd.read_parquet(target.candidates_parquet)
    declarations = pd.read_parquet(
        target.declarations_parquet, columns=["doc_id", "status", "dumped"]
    )
    code = pd.read_parquet(
        target.code_usage_parquet,
        columns=["doc_id", "use", "listing_truncated", "n_files_unread"],
    )
    df = candidates.merge(declarations, on="doc_id", how="left", validate="one_to_one")
    df = df.merge(code, on="doc_id", how="left", validate="one_to_one")
    assert len(df) == len(candidates), "merge changed the row count"
    assert df["status"].notna().all(), "candidates stage 04 has not read"
    assert df["use"].notna().all(), "candidates stage 05 has not read"

    mention = re.compile(target.mention, re.IGNORECASE)
    in_code = ~df["use"].isin(NOT_IN_CODE)
    wrote_it = (df["status"] == "declared") & ~flag(df["dumped"])
    names_it = pd.Series(
        [bool(mention.search(self_description(row))) for row in df.itertuples()], index=df.index
    )
    # A package known only from PyPI has no code to read, which is not the same as code
    # that was only partly read.
    unread = (
        flag(df["listing_truncated"])
        | df["n_files_unread"].fillna(0).gt(0)
        | ((df["source"] == "github") & (df["use"] == "No code read"))
    )

    print(f"Candidates: {len(df)}")
    kept = pd.Series(False, index=df.index)
    for reason, holds in zip(REASONS, [in_code, wrote_it, names_it, unread], strict=True):
        new = holds & ~kept
        print(f"  {new.sum():>6} kept because {reason}")
        kept |= new
    dropped = df[~kept]
    where = dropped["status"].astype(str).where(~flag(dropped["dumped"]), "dumped")
    print(
        f"  {len(dropped):>6} dropped ({len(dropped) / len(df):.0%}): no sign of the package in "
        f"their code, their own dependency lists or their README. Where it was listed: "
        f"{where.value_counts().to_dict()}"
    )

    documents = candidates[kept.to_numpy()].reset_index(drop=True)
    assert documents["doc_id"].is_unique, "duplicate doc_id"
    assert len(documents) + len(dropped) == len(candidates)
    print(f"{len(candidates)} candidates → {len(documents)} documents")
    write_parquet_safely(documents, target.documents_parquet)
    print(f"Wrote {target.documents_parquet.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
