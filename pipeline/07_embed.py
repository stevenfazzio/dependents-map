"""Embed each document's text with Qwen3-Embedding-8B.

Writes embeddings.npz (doc_ids, vectors), row-aligned with documents.parquet. Vectors
come through embedder.py's disk cache, so a re-run only embeds text it has not seen,
and an interrupted run resumes where it stopped.

Usage:
    uv run python pipeline/07_embed.py umap-learn
    uv run python pipeline/07_embed.py umap-learn --limit 200    # timing pilot
"""

import argparse
import time

import numpy as np
import pandas as pd
from config import ROOT, get_target
from embedder import Embedder
from storage import write_npz_safely


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("target", help="target slug from config.TARGETS")
    parser.add_argument(
        "--limit", type=int, help="embed a random sample of N documents and write nothing"
    )
    args = parser.parse_args()
    target = get_target(args.target)

    documents = pd.read_parquet(target.documents_parquet, columns=["doc_id", "text"])
    if args.limit:
        documents = documents.sample(args.limit, random_state=0)
        print(f"--limit: a random sample of {len(documents)} documents")

    start = time.time()
    vectors = Embedder().encode(documents["text"].tolist(), show_progress_bar=True)
    elapsed = time.time() - start

    assert vectors.shape[0] == len(documents), (vectors.shape, len(documents))
    assert np.isfinite(vectors).all(), "non-finite values in the embeddings"
    norms = np.linalg.norm(vectors, axis=1)
    print(
        f"{len(documents)} documents → {vectors.shape} in {elapsed / 60:.1f} min, "
        f"norms {norms.min():.3f} to {norms.max():.3f}"
    )
    if args.limit:
        print("--limit run: no file written.")
        return

    write_npz_safely(
        target.embeddings_npz,
        doc_ids=documents["doc_id"].to_numpy(dtype=str),
        vectors=vectors,
    )
    print(f"Wrote {target.embeddings_npz.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
