"""Reduce the document embeddings to the 2-d layout the map is drawn on.

This one layout is used twice: Toponymy clusters on it to find the regions it names,
and DataMapPlot plots it. Sharing it is what keeps named regions visually coherent, and
it makes min_dist a clustering parameter: it stays low so regions come out dense enough
for density clustering to find.

Usage:
    uv run python pipeline/05_reduce.py umap-learn
"""

import argparse

import numpy as np
import umap
from config import ROOT, get_target
from storage import write_npz_safely

N_NEIGHBORS = 15
MIN_DIST = 0.05
# A fixed seed keeps the layout identical between runs, at the cost of UMAP running
# single-threaded (it overrides n_jobs to 1 when random_state is set).
RANDOM_STATE = 42


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("target", help="target slug from config.TARGETS")
    args = parser.parse_args()
    target = get_target(args.target)

    with np.load(target.embeddings_npz) as data:
        doc_ids, vectors = data["doc_ids"], data["vectors"]
    print(f"Reducing {vectors.shape} with UMAP (n_neighbors={N_NEIGHBORS}, min_dist={MIN_DIST})")

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=N_NEIGHBORS,
        min_dist=MIN_DIST,
        metric="cosine",
        random_state=RANDOM_STATE,
    )
    coords = reducer.fit_transform(vectors).astype(np.float32)

    # UMAP's orientation is arbitrary and screens are wide, so lay the map on its side
    # when it comes out taller than it is broad. A quarter turn preserves every distance,
    # so the regions found downstream are essentially the same either way (on umap-learn,
    # an adjusted Rand index of 0.99 at the finest layer; not bit-identical).
    spread_x, spread_y = np.ptp(np.percentile(coords, [1, 99], axis=0), axis=0)
    if spread_y > spread_x:
        coords = np.column_stack([coords[:, 1], -coords[:, 0]])
        print(f"Rotated a quarter turn: the layout was {spread_x:.1f} wide by {spread_y:.1f} tall")

    assert coords.shape == (len(doc_ids), 2), coords.shape
    assert np.isfinite(coords).all(), "non-finite coordinates"
    write_npz_safely(target.coords_npz, doc_ids=doc_ids, coords=coords)
    print(f"Wrote {coords.shape} to {target.coords_npz.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
