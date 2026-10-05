# Dependents Map

Interactive maps of the open-source projects that depend on a given package. Each
project is placed by the meaning of its README, and the regions that form are named at
four zoom levels.

The first map is of [`umap-learn`](https://github.com/lmcinnes/umap): 10,509 projects
that GitHub's dependency graph listed as depending on it in October 2026.

**Live map:** https://stevenfazzio.com/dependents-map/umap-learn/

All maps are listed at https://stevenfazzio.com/dependents-map/. GitHub Pages serves
them from `docs/`.

The published page carries only metadata (names, star counts, dates, topics) and
model-written text (one-sentence summaries, categories, region names). No README text
is redistributed. Clicking a point opens the project.

## How it's built

Every stage takes a target (for now, `umap-learn`), reads and writes under
`data/<target>/`, and can be re-run: fetched pages, embeddings and LLM responses are
all kept on disk, so a re-run only does what is missing.

| Stage | Script | What it does |
|---|---|---|
| Scrape | `00_scrape_dependents.py` | Walks GitHub's "Dependents" pages for the package, both the repository list and the package list. GitHub has no API for this |
| Resolve | `01_resolve_packages.py` | Looks up on PyPI the packages GitHub lists without a repository, for their description and a link to their source |
| Fetch | `02_fetch_repos.py` | Metadata and README for every repository, through GitHub's GraphQL API |
| Select | `03_select_corpus.py` | One document per project: drops forks represented by their parent, READMEs under 200 characters and identical copies, and adds packages known only from PyPI |
| Embed | `04_embed.py` | Qwen3-Embedding-8B, served by vLLM on a RunPod serverless endpoint, embeds each README (cut to 8,000 tokens) |
| Reduce | `05_reduce.py` | UMAP to the 2-d layout |
| Enrich | `06_enrich.py` | Claude Haiku 4.5 (Message Batches API) writes a one-sentence summary and labels project type, domain, and what the README says UMAP is used for |
| Label | `07_label.py` | [Toponymy](https://github.com/TutteInstitute/toponymy) finds regions on the layout and names them with Claude Opus 5.5, from the summaries |
| Render | `08_render.py` | [DataMapPlot](https://github.com/TutteInstitute/datamapplot) builds `docs/<target>/index.html` and the data files beside it |
| Card | `09_social_preview.py` | Screenshots the rendered map in Chrome for the link-preview image |

Regions are found on the same 2-d layout that is plotted, so every named region matches
something visible on the map.

## The umap-learn map in numbers

- GitHub listed 13,743 repositories and 1,518 packages. Its page header states 25,638
  repositories; the list itself ends at 13,743.
- 661 of the packages had no linked repository. 608 of them exist on PyPI, and 399 of
  those point at a GitHub repository.
- 14,333 repositories were looked up and 14,251 found. After the selection rules,
  10,286 remain, plus 223 packages known only from their PyPI description.
- The map has 245, 73, 22 and 6 named regions, from the finest layer to the coarsest.
- About 19% of the projects mention UMAP anywhere in what was read. Point size and one
  colormap show GitHub stars; the other colormaps show domain, project type, UMAP use,
  package or repository, and the year of the last push.

## Reproducing

Requires [uv](https://docs.astral.sh/uv/) and:

- a GitHub login for `gh` (or `GITHUB_TOKEN`), for stage 02;
- `RUNPOD_API_KEY` and an endpoint serving Qwen3-Embedding-8B, for stages 04 and 07. The
  endpoint's configuration is at the top of `pipeline/embedder.py`, and its ID is set
  there. The one that built this map has been deleted;
- `ANTHROPIC_API_KEY`, for stages 06 and 07;
- a local Chrome, for stage 09.

```bash
uv run python pipeline/00_scrape_dependents.py umap-learn    # about an hour: GitHub throttles these pages
uv run python pipeline/01_resolve_packages.py umap-learn
uv run python pipeline/02_fetch_repos.py umap-learn
uv run python pipeline/03_select_corpus.py umap-learn
uv run python pipeline/04_embed.py umap-learn --limit 200    # timing pilot
uv run python pipeline/04_embed.py umap-learn                # about 40 minutes on a 48 GB A40
uv run python pipeline/05_reduce.py umap-learn
uv run python pipeline/06_enrich.py umap-learn --pilot 60    # read the outputs before the batch
uv run python pipeline/06_enrich.py umap-learn
uv run python pipeline/07_label.py umap-learn --explore      # region counts at a few settings
uv run python pipeline/07_label.py umap-learn --dry-run      # cost estimate
uv run python pipeline/07_label.py umap-learn
uv run python pipeline/08_render.py umap-learn
uv run python pipeline/09_social_preview.py umap-learn
uv run python pipeline/build_index.py                        # the page that lists the maps
```

For umap-learn the paid steps came to about $21 for the summaries and categories, $2
for the region names, and about $1.30 of RunPod GPU time.

The rendered map loads its data from files beside it, so it has to be served over HTTP
to view locally:

```bash
python3 -m http.server 8765 --bind 127.0.0.1 --directory docs
```

Then open http://127.0.0.1:8765/ for the list of maps, or
http://127.0.0.1:8765/umap-learn/ for the map itself.

Intermediate data lives in `data/` and is not committed.

## Mapping another package

Add a `Target` to `TARGETS` in `pipeline/config.py` with the repository whose dependency
graph lists the dependents, the package name as GitHub's package selector shows it, and
a title. Then run the stages with the new target's slug. Its data goes to
`data/<slug>/` and its map to `docs/<slug>/`, and `build_index.py` adds it to the list
at `docs/index.html`.

Two things are still specific to umap-learn and would need generalising: the enrichment
prompt in `06_enrich.py` (it describes UMAP and asks what UMAP is used for) and the PyPI
lookup in stage 01, which assumes a Python package.

## License

The code is released under the [MIT License](LICENSE). The projects shown on the map
belong to their authors.
