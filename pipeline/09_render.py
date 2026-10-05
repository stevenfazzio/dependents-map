"""Render the interactive map to docs/<target>/index.html, which GitHub Pages serves.

Everything on the page is metadata or model-written text: names, star counts, dates,
the generated one-sentence summary and the categories. README text stays out of the
page; it is in the embeddings, not the HTML.

Usage:
    uv run python pipeline/09_render.py umap-learn
    uv run python pipeline/build_index.py    # refresh the page that lists the maps
"""

import argparse
import colorsys
import html
import json
import os
import re
import tempfile

import datamapplot
import glasbey
import numpy as np
import pandas as pd
from config import PUBLIC_BASE_URL, ROOT, Target, get_target
from matplotlib.colors import to_hex, to_rgb
from storage import write_bytes_safely

FONT = "IBM Plex Sans"
# Frame the central 99% of points on load. A small, far-flung island (on umap-learn, 36
# submissions of one course assignment) otherwise shrinks everything else to fit it in.
INITIAL_ZOOM_FRACTION = 0.99
# DataMapPlot's default of 3 hid four of the six top-level labels on load at 1280 px.
TEXT_COLLISION_SIZE_SCALE = 2
# The page's data goes in files beside it rather than inside it: a multi-megabyte page is
# slow on a phone and too heavy for link-preview crawlers. The price is that the map has
# to be served over HTTP; it no longer opens from a file:// path.
DATA_PREFIX = "map"
STARS_CMAP = "YlGnBu"
STARS_LOG10_CAP = 4

# DataMapPlot sets line-height 0.95 on its panels, so a wrapped title's descenders
# run into the subtitle.
CUSTOM_CSS = """
#main-title {
  line-height: 1.1 !important;
  letter-spacing: -0.01em;
  text-wrap: balance;
}
#main-title + br + span {
  display: inline-block; margin-top: 4px; line-height: 1.3; text-wrap: pretty;
}
.title-pill {
  display: inline-block; margin: 8px 6px 0 0; padding: 3px 10px;
  font-size: 11px; font-weight: 500; letter-spacing: 0.04em; text-transform: uppercase;
  color: #636c76; background: rgba(99, 108, 118, 0.08);
  border: 1px solid rgba(99, 108, 118, 0.15); border-radius: 12px;
}
"""

# Non-categories are pinned to greys so the named categories carry the colour. Where a
# catch-all holds most of the corpus, a pale grey turns the colormap into a picture of
# where the exceptions are.
NEUTRAL_COLORS = {
    "Not mentioned": "#d9d9d9",
    "Not found": "#d9d9d9",
    "Repository": "#d0d0d0",
    "Listed only": "#9a9a9a",
    "Other": "#8c8c8c",
    "Unknown": "#e2e2e2",
}

# glasbey measures distance as a colour-blind viewer sees it, and for the third category of
# "where the package is listed" it chose a magenta that normal vision barely separates
# from the red beside it (33 in CAM02-UCS, where every other pair is over 50). This green
# is the colour farthest from the other three at worst over normal, deuteranomalous and
# protanomalous vision. It is much lighter than the red, which is what keeps the two
# apart for a colour-blind viewer.
LISTED_IN_OVERRIDES = {"Lock file only": "#51d400"}

# Cards are read in quick succession while sweeping the mouse, so every card has the
# same layout: domain pill, name, labelled fields, then the summary and a footer. The
# fields sit above the summary because summaries vary in length; below it they would
# land at a different height on every card.
DOT = (
    "display: inline-block; width: 8px; height: 8px; border-radius: 50%; "
    "margin-right: 6px; box-shadow: inset 0 0 0 1px rgba(0, 0, 0, 0.15); "
    "background: {color};"
)
FIELD_NAME = "color: #6b7280; font-size: 11px; padding-top: 1px;"
HOVER_TEMPLATE = f"""
<div style="max-width: 340px; white-space: normal; font-weight: 400; color: #1f2328;">
  <span style="display: inline-block; padding: 1px 8px; margin-bottom: 5px;
    border-radius: 999px; font-size: 11px; font-weight: 600;
    background: {{pill_bg}}; color: {{pill_fg}};">{{domain}}</span>
  <div style="font-weight: 600; font-size: 15px; line-height: 1.25;
    overflow-wrap: anywhere;">{{title}}</div>
  <div style="display: grid; grid-template-columns: 62px minmax(0, 1fr);
    gap: 3px 10px; margin-top: 6px; font-size: 12px; line-height: 1.35;">
    <div style="{FIELD_NAME}">Type</div>
    <div><span style="{DOT.format(color="{type_dot}")}"></span>{{project_type}}</div>
    <div style="{FIELD_NAME}">UMAP use</div>
    <div><span style="{DOT.format(color="{role_dot}")}"></span>{{umap_role}}</div>
    <div style="{FIELD_NAME}">Listed in</div>
    <div><span style="{DOT.format(color="{listed_dot}")}"></span>{{listed_in}}</div>
  </div>
  <div style="font-size: 13px; margin-top: 8px; padding-top: 7px;
    border-top: 1px solid #e5e7eb; line-height: 1.4;">{{summary}}</div>
  <div style="color: #6b7280; font-size: 11px; margin-top: 7px;">{{footer}}</div>
</div>
"""


def categorical_palette(n_colors: int, avoid: list[str]) -> list[str]:
    """n_colors glasbey colours, each also kept distinct from the `avoid` colours."""
    # Seeding with the greys already in use stops glasbey handing out a muted colour
    # that reads as one of them.
    palette = glasbey.extend_palette(
        avoid,
        palette_size=len(avoid) + n_colors,
        colorblind_safe=True,
        cvd_severity=50.0,  # glasbey's default: moderate red-green deficiency
        # A floor of 35: near-black colours are distinct as swatches but read as one
        # colour when drawn as small dots.
        lightness_bounds=(35, 75),
        chroma_bounds=(20, 90),
    )
    return palette[len(avoid) :]


def color_mapping(values: pd.Series, overrides: dict[str, str] | None = None) -> dict[str, str]:
    """Category -> colour, most frequent first.

    glasbey's palette is greedy, so its earliest colours are the most distinct; handing
    them out by frequency gives the largest categories the clearest colours. `overrides`
    replaces the colour glasbey chose for a category.
    """
    order = values.value_counts().index.tolist()
    named = [c for c in order if c not in NEUTRAL_COLORS]
    neutrals = {c: NEUTRAL_COLORS[c] for c in order if c in NEUTRAL_COLORS}
    palette = categorical_palette(len(named), list(neutrals.values()))
    mapping = dict(zip(named, palette, strict=True))
    mapping.update({c: color for c, color in (overrides or {}).items() if c in mapping})
    mapping.update(neutrals)
    return mapping


def categorical_colormap(
    field: str, description: str, values: pd.Series, overrides: dict[str, str] | None = None
):
    # An explicit "Unknown" rather than dropped points, so a gap in the metadata doesn't
    # read as a hole in the map.
    values = values.fillna("Unknown").astype(str)
    mapping = color_mapping(values, overrides)
    meta = {
        "field": field,
        "description": description,
        "kind": "categorical",
        # An explicit mapping, rather than a palette DataMapPlot assigns itself, so the
        # hovercard can use exactly the colours on the map.
        "color_mapping": mapping,
    }
    return values.to_numpy(), meta, mapping


def relative_luminance(rgb) -> float:
    lin = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in rgb]
    return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]


def contrast(a, b) -> float:
    la, lb = sorted([relative_luminance(a), relative_luminance(b)], reverse=True)
    return (la + 0.05) / (lb + 0.05)


def pill_colors(hex_color: str) -> tuple[str, str]:
    """A pale tint for the pill background and a deep shade of the same hue for text."""
    h, _, s = colorsys.rgb_to_hls(*to_rgb(hex_color))
    bg = colorsys.hls_to_rgb(h, 0.92, min(s, 0.65))
    lightness = 0.35
    fg = colorsys.hls_to_rgb(h, lightness, s)
    while contrast(fg, bg) < 4.5:  # WCAG AA for small text
        lightness -= 0.03
        fg = colorsys.hls_to_rgb(h, lightness, s)
    return to_hex(bg), to_hex(fg)


def shown_label(values: pd.Series, others: pd.Series) -> pd.Series:
    """The category, or the model's own words for it where the category is "Other"."""
    shown = values.copy()
    use_other = (values == "Other") & others.fillna("").str.strip().ne("")
    shown[use_other] = others[use_other].str.strip().map(lambda s: s[:1].upper() + s[1:])
    return shown.fillna("Unknown")


def declaration_category(df: pd.DataFrame) -> pd.Series:
    """Where each project names the target package, from stage 07's reading of its files."""
    declared = df["declaration_status"] == "declared"
    dumped = df["dumped"].fillna(False).astype(bool)
    category = np.select(
        [
            declared & ~dumped,
            declared & dumped,
            df["declaration_status"] == "lock only",
            df["declaration_status"] == "not found",
        ],
        ["Dependency list", "Environment dump", "Lock file only", "Not found"],
        default="Unknown",
    )
    return pd.Series(category, index=df.index)


def declaration_detail(row, category: str) -> str:
    """The category with the specifics a card has room for: which file, or what pulled it in."""
    if category == "Lock file only" and len(row.via):
        more = f" +{len(row.via) - 2}" if len(row.via) > 2 else ""
        return f"{category}, via {', '.join(row.via[:2])}{more}"
    if category in ("Dependency list", "Environment dump"):
        if row.declaration_basis == "PyPI metadata":
            where = "PyPI metadata"
        else:
            # The shallowest file that names it, and a list the project wrote before a lock.
            paths = sorted(row.declared_in, key=lambda p: (p.endswith(".lock"), p.count("/"), p))
            where = paths[0].rsplit("/", 1)[-1]
        scope = {"optional": ", optional", "dev": ", development"}.get(row.scope, "")
        return f"{category} ({where}{scope})"
    return category


def footer(row) -> str:
    parts = []
    if pd.notna(row.stars):
        parts.append(f"★ {int(row.stars):,}")
    if isinstance(row.language, str):
        parts.append(row.language)
    if pd.notna(row.last_active):
        parts.append(f"active {row.last_active.year}")
    if row.is_package:
        names = ", ".join(row.packages[:2])
        parts.append(f"package {names}" if names else "package")
    if row.n_listed_forks:
        parts.append(f"{row.n_listed_forks} forks also listed")
    return " · ".join(parts)


def title_pills(crawled: pd.Timestamp) -> str:
    # The crawl date: the list of dependents is a snapshot of GitHub on that day.
    pills = [f"{crawled:%B %Y}"]
    spans = "".join(f'<span class="title-pill">{html.escape(p)}</span>' for p in pills)
    return f"<div>{spans}</div>"


def open_graph_tags(target: Target, n_documents: int) -> str:
    url = f"{PUBLIC_BASE_URL}{target.slug}/"
    og = {
        "og:title": target.title,
        "og:description": (
            f"{n_documents:,} open-source projects that GitHub lists as depending on "
            f"{target.package}, laid out by what their READMEs say and named at four zoom levels. "
            "Pan, zoom, hover and search."
        ),
        "og:type": "website",
        "og:url": url,
        # Declared unconditionally: the image is a screenshot of this page, so it is
        # always produced after the HTML that names it.
        "og:image": url + "social-preview.png",
    }
    tags = [
        f'<meta property="{key}" content="{html.escape(value, quote=True)}">'
        for key, value in og.items()
    ]
    tags.append(f'<meta name="description" content="{html.escape(og["og:description"])}">')
    tags.append('<meta name="twitter:card" content="summary_large_image">')
    return "\n".join(tags)


def load(target: Target) -> pd.DataFrame:
    documents = pd.read_parquet(target.documents_parquet).drop(columns=["text"])
    enrichment = pd.read_parquet(target.enrichment_parquet).drop(columns=["status"])
    labels = pd.read_parquet(target.labels_parquet)
    df = documents.merge(labels, on="doc_id", how="left", validate="one_to_one")
    df = df.merge(enrichment, on="doc_id", how="left", validate="one_to_one")
    declarations = pd.read_parquet(
        target.declarations_parquet,
        columns=["doc_id", "status", "basis", "dumped", "declared_in", "via", "scope"],
    ).rename(columns={"status": "declaration_status", "basis": "declaration_basis"})
    df = df.merge(declarations, on="doc_id", how="left", validate="one_to_one")
    assert len(df) == len(documents), "merge changed the row count"
    assert df["declaration_status"].notna().all(), "documents stage 07 has not read"
    assert df["x"].notna().all(), "documents without coordinates"
    print(f"{len(df)} documents; {df['summary'].isna().sum()} without a summary")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("target", help="target slug from config.TARGETS")
    args = parser.parse_args()
    target = get_target(args.target)
    df = load(target)

    domain_label = shown_label(df["domain"], df["domain_other"])
    type_label = shown_label(df["project_type"], df["project_type_other"])
    summary = df["summary"].fillna("").str.strip()
    unterminated = summary.ne("") & ~summary.str.endswith((".", "?", "!"))
    summary[unterminated] = summary[unterminated] + "."
    # A package without a repository has no push date; its last release stands in.
    df["last_active"] = df["pushed_at"].fillna(df["pypi_latest_upload_at"])

    # Colormaps use the category; the hovercard and search keep the specific text.
    domain_values, domain_meta, domain_colors = categorical_colormap(
        "domain", "Domain", df["domain"]
    )
    type_values, type_meta, type_colors = categorical_colormap(
        "project_type", "Project type", df["project_type"]
    )
    role_values, role_meta, role_colors = categorical_colormap(
        "umap_role", "UMAP use (per README)", df["umap_role"]
    )
    listed_values, listed_meta, listed_colors = categorical_colormap(
        "listed_in",
        f"Where {target.package} is listed",
        declaration_category(df),
        LISTED_IN_OVERRIDES,
    )
    kind_values, kind_meta, _ = categorical_colormap(
        "listed_as",
        "Package or repository",
        pd.Series(np.where(df["is_package"], "Package", "Repository"), index=df.index),
    )
    year = df["last_active"].dt.year
    year = year.fillna(year.min()).to_numpy().astype(float)
    pills = {category: pill_colors(color) for category, color in domain_colors.items()}

    extra = pd.DataFrame(
        {
            "title": df["name"].map(html.escape),
            "summary": summary.map(html.escape),
            "domain": domain_label.map(html.escape),
            "pill_bg": [pills[v][0] for v in domain_values],
            "pill_fg": [pills[v][1] for v in domain_values],
            "project_type": type_label.map(html.escape),
            "type_dot": [type_colors[v] for v in type_values],
            "umap_role": pd.Series(role_values).map(html.escape),
            "role_dot": [role_colors[v] for v in role_values],
            "listed_in": [
                html.escape(declaration_detail(row, category))
                for row, category in zip(df.itertuples(), listed_values, strict=True)
            ],
            "listed_dot": [listed_colors[v] for v in listed_values],
            "footer": [html.escape(footer(row)) for row in df.itertuples()],
            "url": df["url"],
            # Search the composed text, not just names: the summary and the categories
            # are where a project's content shows up.
            "search_text": [
                " ".join(
                    [
                        row.name,
                        " ".join(row.packages),
                        text,
                        domain,
                        kind,
                        " ".join(row.topics) if row.topics is not None else "",
                        # What a lock file says pulled the package in, so searching for
                        # that library finds the projects it carries.
                        " ".join(row.via) if row.declaration_status == "lock only" else "",
                    ]
                )
                for row, text, domain, kind in zip(
                    df.itertuples(), summary, domain_label, type_label, strict=True
                )
            ],
        }
    )

    label_columns = sorted(
        (c for c in df.columns if c.startswith("label_layer_")),
        key=lambda c: int(c.rsplit("_", 1)[1]),  # numeric: _10 must sort after _2
    )
    label_layers = [df[c].fillna("Unlabelled").to_numpy() for c in label_columns]
    n_finest = pd.Series(label_layers[0]).nunique()
    n_coarsest = pd.Series(label_layers[-1]).nunique()
    assert n_finest >= n_coarsest, "label layers must be finest-first for DataMapPlot"

    # Stars set the point size and are also offered as a colormap. That repeats the
    # encoding on purpose: where points crowd together, their sizes are hard to tell apart.
    # A package with no repository has no star count and is drawn as zero.
    stars = np.log10(1 + df["stars"].fillna(0).to_numpy(dtype=float))
    marker_sizes = 3 + 15 * stars / stars.max()

    output = target.map_html
    output.parent.mkdir(parents=True, exist_ok=True)
    # Data files from an earlier build can outlive it if the chunking changes.
    for stale in output.parent.glob(f"{DATA_PREFIX}_*"):
        stale.unlink()

    fig = datamapplot.create_interactive_plot(
        df[["x", "y"]].to_numpy(),
        *label_layers,
        hover_text=df["name"].to_numpy(),
        inline_data=False,
        # The page refers to these files by name alone, so they sit in its directory.
        offline_data_path=str(output.parent / DATA_PREFIX),
        extra_point_data=extra,
        hover_text_html_template=HOVER_TEMPLATE,
        on_click="window.open(`{url}`)",
        enable_search=True,
        search_field="search_text",
        # Domain first: it is the split the layout is organised around.
        colormap_rawdata=[
            domain_values,
            type_values,
            role_values,
            listed_values,
            kind_values,
            # DataMapPlot spaces five legend ticks evenly over the range, so capping at
            # 10,000 stars puts them on whole powers of ten. On umap-learn, 25 projects
            # exceed the cap.
            np.minimum(stars, STARS_LOG10_CAP),
            year,
        ],
        colormap_metadata=[
            domain_meta,
            type_meta,
            role_meta,
            listed_meta,
            kind_meta,
            {
                "field": "stars",
                "description": f"Stars, log10 ({STARS_LOG10_CAP} = 10k+)",
                "kind": "continuous",
                # Pale to dark: half the corpus has no stars, so the starred projects
                # stand out against a faint background.
                "cmap": STARS_CMAP,
            },
            {
                "field": "last_active",
                "description": "Year last active",
                "kind": "continuous",
                "cmap": "viridis",
            },
        ],
        marker_size_array=marker_sizes,
        cvd_safer=True,
        title=target.title,
        sub_title=(
            # "GitHub lists": its dependency graph counts lock files and dumped
            # environments, so some of these projects never asked for the package.
            f"{len(df):,} projects GitHub lists as depending on {target.package}."
            # DataMapPlot inserts the subtitle as HTML, so the break is explicit: two
            # short lines keep the panel narrower than one long wrapped one.
            "<br />"
            # Nothing else on the page explains point size, or says clicking works:
            # DataMapPlot has no size legend and deck.gl shows a grab cursor on points.
            "Mapped by README. Point size shows GitHub stars. Click to open."
        ),
        # The defaults (36/18) make the title panel cover the top of the map on
        # laptop-width screens.
        title_font_size=24,
        sub_title_font_size=13,
        font_family=FONT,
        tooltip_font_family=FONT,
        tooltip_font_weight=400,
        custom_css=CUSTOM_CSS,
        initial_zoom_fraction=INITIAL_ZOOM_FRACTION,
        text_collision_size_scale=TEXT_COLLISION_SIZE_SCALE,
        noise_label="Unlabelled",
    )

    crawled = pd.read_parquet(target.dependents_parquet, columns=["fetched_at"])["fetched_at"]
    fd, tmp = tempfile.mkstemp(dir=output.parent, suffix=".html")
    os.close(fd)
    try:
        fig.save(tmp)
        with open(tmp, encoding="utf-8") as f:
            text = f.read()
        # After the charset declaration, which browsers only honour within the first
        # 1024 bytes of the document.
        anchor = re.search(r"<meta[^>]*charset[^>]*>", text) or re.search("<head>", text)
        assert anchor is not None, "no <head> in the rendered page"
        at = anchor.end()
        text = text[:at] + "\n" + open_graph_tags(target, len(df)) + text[at:]
        # Pills go inside the title panel, after the subtitle; the panel holds only
        # spans, so its first closing </div> is its own.
        text, n = re.subn(
            r'(<div\s+id="title-container"[^>]*>.*?)(</div>)',
            lambda m: m.group(1) + title_pills(crawled.max()) + m.group(2),
            text,
            count=1,
            flags=re.DOTALL,
        )
        assert n == 1, "title panel not found for the pills"
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(text)
        os.chmod(tmp, 0o644)
        os.replace(tmp, output)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
    # A few facts about this build, for the index page that lists the maps.
    meta = {
        "title": target.title,
        "package": target.package,
        "documents": len(df),
        "crawled": f"{crawled.max():%Y-%m}",
    }
    write_bytes_safely(target.map_meta_json, (json.dumps(meta, indent=2) + "\n").encode())

    data_files = sorted(output.parent.glob(f"{DATA_PREFIX}_*"))
    assert data_files, "no data files were written beside the page"
    print(f"Wrote {output.relative_to(ROOT)} ({output.stat().st_size / 1e3:.0f} KB) and:")
    for path in data_files:
        print(f"  {path.name} ({path.stat().st_size / 1e3:.0f} KB)")
    total = output.stat().st_size + sum(path.stat().st_size for path in data_files)
    print(
        f"Total {total / 1e6:.2f} MB. Serve {output.parent.relative_to(ROOT)}/ over HTTP to view."
    )


if __name__ == "__main__":
    main()
