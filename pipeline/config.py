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
class Target:
    """A project whose dependents we map."""

    slug: str  # directory name under data/ and docs/
    repo: str  # owner/name of the GitHub repo whose dependency graph lists the dependents
    package: str  # package name as shown in that page's package selector
    title: str  # the map's title

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


TARGETS = {
    t.slug: t
    for t in [
        Target(
            slug="umap-learn",
            repo="lmcinnes/umap",
            package="umap-learn",
            title="UMAP Dependents Map",
        ),
    ]
}


def get_target(slug: str) -> Target:
    try:
        return TARGETS[slug]
    except KeyError:
        raise SystemExit(f"Unknown target {slug!r}. Known targets: {sorted(TARGETS)}") from None
