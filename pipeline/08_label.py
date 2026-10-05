"""Find the map's regions with Toponymy, at several zoom levels, and name them.

Clustering runs on the 2-d layout from stage 05, the same coordinates that get
plotted, so the regions are ones a viewer can see. Regions are named from the
one-sentence summaries of stage 06, not from the raw READMEs.

    uv run python pipeline/08_label.py umap-learn --explore
        Free: fits only the clusterer at a few settings and prints, per layer, how
        many regions it finds and how many documents fall outside any region.

    uv run python pipeline/08_label.py umap-learn --dry-run
        Nearly free: runs the whole fit with placeholder names to collect every prompt
        the real run would send, counts their tokens, and names a few regions for real
        to measure output size. Prints a cost estimate.

    uv run python pipeline/08_label.py umap-learn
        Paid: fits and names the regions.

Writes labels.parquet with x, y and label_layer_0..k, where layer 0 is the FINEST layer
(Toponymy's in-memory order, which DataMapPlot also expects), and topic_names.json.

The embedding endpoint has to be reachable throughout: Toponymy embeds its keyphrases
and the names it generates with the same model that embedded the documents.
"""

import argparse
import asyncio
import json
import random
import warnings
from concurrent.futures import ThreadPoolExecutor

import anthropic
import numpy as np
import pandas as pd
from config import ROOT, Target, get_target
from embedder import Embedder
from namer import MODEL, PRICE_INPUT, PRICE_OUTPUT, ClaudeNamer
from storage import write_parquet_safely
from toponymy import Toponymy, ToponymyClusterer

MIN_CLUSTERS = 4
BASE_MIN_CLUSTER_SIZE = 10
EXPLORE_SIZES = [10, 15, 20, 30, 50]
# Toponymy spreads these across the layers and rounds each onto a name-length tier. With
# four layers, 0.4 to 0.8 asks for 4-8 words at the finest layer and 1-4 at the coarsest.
DETAIL_LEVELS = (0.4, 0.8)
OBJECT_DESCRIPTION = "one-sentence descriptions of software projects"
DRY_RUN_REAL_SAMPLE = 8


def corpus_description(target: Target) -> str:
    return f"collection of open-source projects that depend on the {target.package} library"


def load_vectors(target: Target) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(target.embeddings_npz) as data:
        doc_ids, vectors = data["doc_ids"], data["vectors"]
    with np.load(target.coords_npz) as data:
        assert list(data["doc_ids"]) == list(doc_ids), "layout out of sync with embeddings"
        coords = data["coords"]
    return doc_ids, vectors, coords


def load_summaries(target: Target, doc_ids: np.ndarray) -> list[str]:
    """One short text per document for Toponymy to draw keyphrases and exemplars from."""
    documents = pd.read_parquet(target.documents_parquet, columns=["doc_id", "name", "description"])
    enrichment = pd.read_parquet(target.enrichment_parquet, columns=["doc_id", "status", "summary"])
    merged = documents.merge(enrichment, on="doc_id", how="left", validate="one_to_one")
    assert list(merged["doc_id"]) == list(doc_ids), "documents out of sync with embeddings"

    summary = merged["summary"].where(merged["status"] == "ok")
    description = merged["description"].fillna("").str.strip()
    fallback = description.where(description != "", merged["name"])
    if missing := summary.isna().sum():
        print(f"{missing} documents have no summary; using their description or name instead")
    return summary.fillna(fallback).tolist()


def make_clusterer(base_min_cluster_size: int) -> ToponymyClusterer:
    return ToponymyClusterer(min_clusters=MIN_CLUSTERS, base_min_cluster_size=base_min_cluster_size)


def explore(vectors: np.ndarray, coords: np.ndarray, sizes: list[int]) -> None:
    for size in sizes:
        clusterer = make_clusterer(size)
        # The clusterer takes the low-D layout first; Toponymy.fit takes it second.
        clusterer.fit(clusterable_vectors=coords, embedding_vectors=vectors)
        layers = []
        for layer in clusterer.cluster_layers_:  # finest first
            labels = layer.cluster_labels
            layers.append(f"{labels.max() + 1} ({(labels == -1).mean():.0%} outside)")
        print(
            f"base_min_cluster_size={size}: {len(layers)} layers, finest → coarsest: "
            + " | ".join(layers)
        )


def fit(
    target: Target, namer: ClaudeNamer, size: int, detail: tuple[float, float]
) -> tuple[Toponymy, np.ndarray, np.ndarray]:
    doc_ids, vectors, coords = load_vectors(target)
    summaries = load_summaries(target, doc_ids)
    model = Toponymy(
        llm_wrapper=namer,
        text_embedding_model=Embedder(),
        clusterer=make_clusterer(size),
        object_description=OBJECT_DESCRIPTION,
        corpus_description=corpus_description(target),
        lowest_detail_level=detail[0],
        highest_detail_level=detail[1],
        verbose=False,
    )
    np.random.seed(42)  # exemplar selection draws from NumPy's global generator
    # Toponymy.fit takes (objects, high-D embeddings, low-D layout): the reverse of the
    # clusterer's argument order, so pass by keyword.
    model.fit(summaries, embedding_vectors=vectors, clusterable_vectors=coords)
    return model, doc_ids, coords


def dry_run(target: Target, size: int, detail: tuple[float, float]) -> None:
    namer = ClaudeNamer(dry_run=True)
    with warnings.catch_warnings():
        # Placeholder answers can't be parsed as disambiguation results; that is expected.
        warnings.simplefilter("ignore")
        model, _, _ = fit(target, namer, size, detail)
    n_regions = sum(len(names) for names in model.topic_names_)
    prompts = namer.recorded
    print(f"{n_regions} regions → {len(prompts)} prompts, including disambiguation passes")

    client = anthropic.Anthropic()

    def count(prompt: tuple[str | None, str]) -> int:
        system, user = prompt
        kwargs = {"system": system} if system else {}
        return client.messages.count_tokens(
            model=MODEL, messages=[{"role": "user", "content": user}], **kwargs
        ).input_tokens

    with ThreadPoolExecutor(max_workers=8) as pool:
        input_tokens = sum(pool.map(count, prompts))

    # Output size, thinking included, can only be measured by asking. The finest layer's
    # prompts come first and don't depend on any generated name, so these answers are
    # cached for the real run.
    real = ClaudeNamer()
    finest = prompts[: len(model.topic_names_[0])]
    sample = random.Random(0).sample(finest, min(DRY_RUN_REAL_SAMPLE, len(finest)))

    async def ask() -> list[str]:
        return await asyncio.gather(*(real.create(system, user) for system, user in sample))

    answers = asyncio.run(ask())
    for answer in answers[:4]:
        print(f"  sample answer: {answer.strip()[:160]}")
    if not real.usage["calls"]:
        print("The sample was already cached, so output size was not measured again.")
        return
    output_per_call = real.usage["output_tokens"] / real.usage["calls"]
    output_tokens = output_per_call * len(prompts)
    cost = (input_tokens * PRICE_INPUT + output_tokens * PRICE_OUTPUT) / 1e6
    print(
        f"Estimate for {MODEL}: {input_tokens:,} input tokens + about {output_tokens:,.0f} "
        f"output tokens ({output_per_call:.0f} per call, measured on {real.usage['calls']} "
        f"calls) ≈ ${cost:.2f}. The sample itself cost ${real.cost():.2f}."
    )


def fit_and_name(target: Target, size: int, detail: tuple[float, float]) -> None:
    namer = ClaudeNamer()
    model, doc_ids, coords = fit(target, namer, size, detail)

    names = model.topic_names_
    assert len(names[0]) >= len(names[-1]), "expected layer 0 to be the finest"
    labels = pd.DataFrame({"doc_id": doc_ids, "x": coords[:, 0], "y": coords[:, 1]})
    for i, name_vector in enumerate(model.topic_name_vectors_):
        labels[f"label_layer_{i}"] = name_vector
    write_parquet_safely(labels, target.labels_parquet)
    target.topic_names_json.write_text(json.dumps(names, indent=2))

    for i, layer_names in enumerate(names):
        outside = (labels[f"label_layer_{i}"] == "Unlabelled").mean()
        words = np.median([len(name.split()) for name in layer_names])
        # A failed call does not stop the fit; it leaves an empty name behind.
        empty = sum(not name.strip() for name in layer_names)
        print(
            f"layer {i}: {len(layer_names)} names, median {words:.0f} words, "
            f"{outside:.0%} of documents outside any region"
            + (f", {empty} EMPTY NAMES" if empty else "")
        )
    usage = namer.usage
    print(
        f"naming: {usage['calls']} calls ({usage['cached']} more from cache), "
        f"{usage['input_tokens']:,} input + {usage['output_tokens']:,} output tokens "
        f"≈ ${namer.cost():.2f}"
    )
    print(f"Wrote {target.labels_parquet.relative_to(ROOT)} and {target.topic_names_json.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("target", help="target slug from config.TARGETS")
    parser.add_argument("--explore", action="store_true", help="print layer counts and exit")
    parser.add_argument("--sizes", type=int, nargs="+", default=EXPLORE_SIZES)
    parser.add_argument("--dry-run", action="store_true", help="estimate the naming cost")
    parser.add_argument("--min-cluster-size", type=int, default=BASE_MIN_CLUSTER_SIZE)
    parser.add_argument(
        "--detail",
        type=float,
        nargs=2,
        default=DETAIL_LEVELS,
        metavar=("LOWEST", "HIGHEST"),
        help="Toponymy detail levels for the finest and coarsest layers",
    )
    args = parser.parse_args()
    target = get_target(args.target)

    if args.explore:
        _, vectors, coords = load_vectors(target)
        explore(vectors, coords, args.sizes)
    elif args.dry_run:
        dry_run(target, args.min_cluster_size, tuple(args.detail))
    else:
        fit_and_name(target, args.min_cluster_size, tuple(args.detail))


if __name__ == "__main__":
    main()
