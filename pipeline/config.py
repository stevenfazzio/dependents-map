"""Targets and paths shared by every stage.

Every stage takes a target slug and reads/writes under data/<slug>/, so mapping a
new project's dependents means adding one entry to TARGETS.
"""

from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
DOCS_DIR = ROOT / "docs"  # served by GitHub Pages
# Where docs/ is published; each target's map lives at <base><slug>/.
PUBLIC_BASE_URL = "https://stevenfazzio.com/dependents-map/"
REPO_URL = "https://github.com/stevenfazzio/dependents-map"


@dataclass(frozen=True)
class Carrier:
    """A library that runs the package for its caller, who never has to name it."""

    label: str  # what the map calls this path
    module: str  # the library's import name
    pattern: str  # code that takes the path


@dataclass(frozen=True)
class Signal:
    """Something code does around the package that says how it gets on with it."""

    label: str
    pattern: str  # searched for in the files that name the package


@dataclass(frozen=True)
class Feature:
    """A part of the library beyond building a model and fitting it, and the code that shows it.

    A project has the feature when any of the conditions given here holds.
    """

    label: str
    called: str = ""  # a pattern over the dotted names the project calls
    method: str = ""  # a method called on one of the library's objects
    argument: tuple[str, object] | None = None  # a constructor argument given this value
    fit_with_target: bool = False  # fit or fit_transform handed a target as well


@dataclass(frozen=True)
class Target:
    """A project whose dependents we map."""

    slug: str  # directory name under data/ and docs/
    repo: str  # owner/name of the GitHub repo whose dependency graph lists the dependents
    package: str  # package name as shown in that page's package selector
    module: str  # the name the package is imported under
    name: str  # what the library is called in prose
    mention: str  # a pattern that finds the library's name in a README
    title: str  # the map's title
    # Libraries that run the package for a caller who never names it.
    carriers: tuple[Carrier, ...] = ()
    signals: tuple[Signal, ...] = ()
    features: tuple[Feature, ...] = ()
    # Constructor arguments the map has a colormap for, as (argument, colormap name):
    # coloured by the value a project gives, or by whether it gives one at all.
    value_colormaps: tuple[tuple[str, str], ...] = ()
    given_colormaps: tuple[tuple[str, str], ...] = ()

    @property
    def data_dir(self) -> Path:
        path = DATA_DIR / self.slug
        path.mkdir(parents=True, exist_ok=True)
        return path

    def dependents_log(self, dependent_type: str) -> Path:
        return self.data_dir / f"dependents_{dependent_type.lower()}.pages.jsonl"

    @property
    def dependents_parquet(self) -> Path:
        return self.data_dir / "dependents.parquet"

    @property
    def pypi_log(self) -> Path:
        return self.data_dir / "pypi_packages.jsonl"

    @property
    def pypi_parquet(self) -> Path:
        return self.data_dir / "pypi_packages.parquet"

    @property
    def repo_metadata_log(self) -> Path:
        return self.data_dir / "repo_metadata.batches.jsonl"

    @property
    def repo_readme_log(self) -> Path:
        return self.data_dir / "repo_readmes.batches.jsonl"

    @property
    def repos_parquet(self) -> Path:
        return self.data_dir / "repos.parquet"

    @property
    def candidates_parquet(self) -> Path:
        return self.data_dir / "candidates.parquet"

    @property
    def documents_parquet(self) -> Path:
        return self.data_dir / "documents.parquet"

    @property
    def embeddings_npz(self) -> Path:
        return self.data_dir / "embeddings.npz"

    @property
    def coords_npz(self) -> Path:
        return self.data_dir / "coords.npz"

    @property
    def enrichment_log(self) -> Path:
        return self.data_dir / "enrichment.jsonl"

    @property
    def enrichment_batch_json(self) -> Path:
        return self.data_dir / "enrichment_batch.json"

    @property
    def enrichment_parquet(self) -> Path:
        return self.data_dir / "enrichment.parquet"

    def declaration_log(self, fetch_pass: str) -> Path:
        return self.data_dir / f"declaration_{fetch_pass}.batches.jsonl"

    @property
    def declarations_parquet(self) -> Path:
        return self.data_dir / "declarations.parquet"

    @property
    def code_tree_log(self) -> Path:
        return self.data_dir / "code_trees.batches.jsonl"

    @property
    def code_usage_parquet(self) -> Path:
        return self.data_dir / "code_usage.parquet"

    @property
    def usage_log(self) -> Path:
        return self.data_dir / "usage_descriptions.jsonl"

    @property
    def usage_batch_json(self) -> Path:
        return self.data_dir / "usage_batch.json"

    @property
    def usage_parquet(self) -> Path:
        return self.data_dir / "usage_descriptions.parquet"

    @property
    def labels_parquet(self) -> Path:
        return self.data_dir / "labels.parquet"

    @property
    def topic_names_json(self) -> Path:
        return self.data_dir / "topic_names.json"

    @property
    def map_html(self) -> Path:
        return DOCS_DIR / self.slug / "index.html"

    @property
    def social_preview_png(self) -> Path:
        return DOCS_DIR / self.slug / "social-preview.png"

    @property
    def map_meta_json(self) -> Path:
        return DOCS_DIR / self.slug / "meta.json"


# A call to warnings.filterwarnings or simplefilter, and the arguments up to a given one.
_SILENCE = r"(?:filterwarnings|simplefilter)\s*\("
_ARGS = r"(?:[^()]|\([^()]*\))*?"

TARGETS = {
    t.slug: t
    for t in [
        Target(
            slug="umap-learn",
            repo="lmcinnes/umap",
            package="umap-learn",
            module="umap",
            name="UMAP",
            mention=r"u-?map|uniform manifold approximation",
            # Each checked in the library's own source, October 2026.
            carriers=(
                # Builds a UMAP model unless it is handed another reducer.
                Carrier("BERTopic", "bertopic", r"\bBERTopic\s*\("),
                # Fits a UMAP model on its document vectors.
                Carrier("Top2Vec", "top2vec", r"\bTop2Vec\s*\("),
                # Its default method builds the graph with umap's fuzzy_simplicial_set.
                Carrier("scanpy.pp.neighbors", "scanpy", r"\bpp\.neighbors\s*\("),
                # A speaker model brings in a clustering backend that calls umap.UMAP.
                Carrier("FunASR speaker model", "funasr", r"\bspk_model\b"),
            ),
            # Chosen from what 1,500 callers' files actually silence and set.
            signals=(
                Signal(
                    "silences every warning",
                    _SILENCE + r"\s*(?:action\s*=\s*)?[\"']ignore[\"']\s*\)",
                ),
                Signal(
                    "silences the TensorFlow import warning",
                    _SILENCE + _ARGS + r"Tensorflow not installed",
                ),
                Signal("silences the n_jobs warning", _SILENCE + _ARGS + r"n_jobs value"),
                Signal(
                    "silences other UMAP warnings",
                    _SILENCE
                    + _ARGS
                    + r"(?:module\s*=\s*[\"']umap|precomputed metric|not fully connected"
                    r"|force_all_finite)",
                ),
                Signal("silences numba warnings", _SILENCE + _ARGS + r"Numba\w*Warning"),
                Signal("sets numba variables", r"NUMBA_[A-Z_]{3,}"),
            ),
            features=(
                Feature("Transforms new data", method="transform"),
                Feature("Inverse transform", method="inverse_transform"),
                Feature("Supervised (fit with labels)", fit_with_target=True),
                Feature("Precomputed metric", argument=("metric", "precomputed")),
                Feature("DensMAP", argument=("densmap", True)),
                Feature("Parametric UMAP", called=r"ParametricUMAP"),
                Feature("Aligned UMAP", called=r"AlignedUMAP|^umap\.aligned_umap\."),
                Feature("umap.plot", called=r"^umap\.plot\."),
                # The functions the classes are built from, such as fuzzy_simplicial_set:
                # anything in a submodule that is not a class and not counted above.
                Feature(
                    "Lower-level functions",
                    called=r"^umap\.(?!plot\.|parametric_umap\.|aligned_umap\.|[A-Z])\w+\.\w",
                ),
            ),
            value_colormaps=(("metric", "Distance metric"),),
            given_colormaps=(("random_state", "Random seed"),),
            title="UMAP Dependents Map",
        ),
    ]
}


def get_target(slug: str) -> Target:
    try:
        return TARGETS[slug]
    except KeyError:
        raise SystemExit(f"Unknown target {slug!r}. Known targets: {sorted(TARGETS)}") from None
