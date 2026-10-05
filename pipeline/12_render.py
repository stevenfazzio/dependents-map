"""Render the interactive map to docs/<target>/index.html, which GitHub Pages serves.

The page holds metadata (names, star counts, dates), model-written text (the one-sentence
summary, the categories, and a sentence on how each project uses the library), and a few
facts read from each project's code and dependency files: the call it makes most often,
the names of what it calls, and the version it asks for. README text stays out of the
page; it is in the embeddings, not the HTML.

Usage:
    uv run python pipeline/12_render.py umap-learn
    uv run python pipeline/build_index.py    # refresh the page that lists the maps
"""

import argparse
import colorsys
import hashlib
import html
import json
import os
import re
import tempfile
from collections import Counter
from pathlib import Path

import datamapplot
import glasbey
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import usage_fields as usage
from code_usage import EXPRESSION, class_signature
from config import PUBLIC_BASE_URL, ROOT, Target, get_target
from matplotlib.colors import to_hex, to_rgb
from storage import write_bytes_safely

FONT = "IBM Plex Sans"
# Frame the central 99% of points on load. A small, far-flung island (on umap-learn, 36
# submissions of one course assignment) otherwise shrinks everything else to fit it in.
INITIAL_ZOOM_FRACTION = 0.99
# The screen the page is laid out for, and the panels in its top-left corner as
# (right edge, bottom edge) in pixels: the title panel, and the search box below it.
DESIGN_SCREEN = (1280, 800)
CORNER_PANELS = [(612, 143), (195, 215)]
# The share of the screen's tighter dimension that the framed layout fills on load,
# measured on this page.
FRAME_SHARE = 0.69
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
    "Not in a dependency list": "#d9d9d9",
    "Repository": "#d0d0d0",
    "Listed only": "#9a9a9a",
    "No code read": "#9a9a9a",
    "Other": "#8c8c8c",
    "Unknown": "#e2e2e2",
}

# The colormaps drawn from a project's calls have three non-answers, here palest first:
# no call to colour, a call that settles for the usual, and a call whose arguments this
# reading cannot see. A legend entry selects points by their colour, so within one
# colormap every grey has to be a different one.
NO_CALL = "No direct call"
NO_FEATURE = "None of these"
UNREADABLE = "Not readable"
SEVERAL = "Several values"
CALL_NEUTRALS = {NO_CALL: "#e2e2e2", NO_FEATURE: "#c4c4c4", UNREADABLE: "#a8a8a8"}
DEFAULT_GREY = CALL_NEUTRALS[NO_FEATURE]
# The same for the two colormaps drawn from stage 10's reading of the code, where the
# non-answers are no call and a call whose purpose or data the code does not show.
CANT_TELL = usage.Purpose.UNKNOWN.value
USE_NEUTRALS = {NO_CALL: "#e2e2e2", CANT_TELL: "#a8a8a8"}
# What a README can say of the library's use that is worth a card's space.
README_SILENT = ("Not mentioned", "Listed only")
# A value that fewer projects than this give shares "Other" with the rest of the rare ones.
MIN_VALUE_PROJECTS = 20
GIVEN_LABELS = {
    usage.EVERY: "Set in every call",
    usage.SOME: "Set in some calls",
    usage.NEVER: "Not set",
    usage.UNREADABLE: UNREADABLE,
}
# A range and a bare ceiling both stop the package at some version.
VERSION_LABELS = {
    "Any version": "Any version",
    "Lower bound": "Lower bound",
    "One version": "One exact version",
    "Compatible releases": "Compatible releases",
    "Bounded range": "Has an upper bound",
    "Upper bound": "Has an upper bound",
}
# What a card shows of one call.
MAX_SHOWN_ARGUMENTS = 6
# How a version specifier begins, as against "" for any version and "from source".
SPEC_STARTS = ("=", "<", ">", "~", "^", "!")
MAX_SHOWN_VALUE = 24

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
# land at a different height on every card. "In code" is the last field for the same
# reason: it runs from one line to seven, with the sentence on what the call is for.
DOT = (
    "display: inline-block; width: 8px; height: 8px; border-radius: 50%; "
    "margin-right: 6px; box-shadow: inset 0 0 0 1px rgba(0, 0, 0, 0.15); "
    "background: {color};"
)
FIELD_NAME = "color: #6b7280; font-size: 11px; padding-top: 1px;"
CODE = (
    "font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; "
    "font-size: 11px; overflow-wrap: anywhere;"
)
# Under a row's value and in line with its text, past the dot.
NOTE = "color: #6b7280; font-size: 11px; margin-left: 14px;"
SENTENCE = "margin: 4px 0 0 14px;"
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
    <div><span style="{DOT.format(color="{use_dot}")}"></span>{{umap_use}}</div>
    <div style="{FIELD_NAME}">Data</div>
    <div><span style="{DOT.format(color="{data_dot}")}"></span>{{data}}</div>
    <div style="{FIELD_NAME}">Listed in</div>
    <div><span style="{DOT.format(color="{listed_dot}")}"></span>{{listed_in}}</div>
    <div style="{FIELD_NAME}">In code</div>
    <div><span style="{DOT.format(color="{code_dot}")}"></span>{{in_code}}</div>
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
        # Left unset, glasbey takes the hue range from a palette of four or more colours,
        # and the range of four greys holds no colours at all.
        hue_bounds=(0, 360),
    )
    return palette[len(avoid) :]


def color_mapping(
    values: pd.Series,
    overrides: dict[str, str] | None = None,
    neutrals: dict[str, str] | None = None,
) -> dict[str, str]:
    """Category -> colour, most frequent first.

    glasbey's palette is greedy, so its earliest colours are the most distinct; handing
    them out by frequency gives the largest categories the clearest colours. `overrides`
    replaces the colour glasbey chose for a category, and `neutrals` names this colormap's
    own greys, beside the ones every colormap shares.
    """
    greys = NEUTRAL_COLORS | (neutrals or {})
    order = values.value_counts().index.tolist()
    named = [c for c in order if c not in greys]
    used = {c: greys[c] for c in order if c in greys}
    palette = categorical_palette(len(named), list(used.values()))
    mapping = dict(zip(named, palette, strict=True))
    mapping.update({c: color for c, color in (overrides or {}).items() if c in mapping})
    mapping.update(used)
    # Clicking a legend entry selects the points drawn in its colour.
    assert len(set(mapping.values())) == len(mapping), f"two categories share a colour: {mapping}"
    return mapping


def categorical_colormap(
    field: str,
    description: str,
    values: pd.Series,
    overrides: dict[str, str] | None = None,
    neutrals: dict[str, str] | None = None,
):
    # An explicit "Unknown" rather than dropped points, so a gap in the metadata doesn't
    # read as a hole in the map.
    values = values.fillna("Unknown").astype(str)
    mapping = color_mapping(values, overrides, neutrals)
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
    """Where each project names the target package, from stage 04's reading of its files."""
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
    """The category with the specifics a card has room for, as HTML: which file and which
    versions, or what pulled the package in."""
    if category == "Lock file only" and len(row.via):
        more = f" +{len(row.via) - 2}" if len(row.via) > 2 else ""
        return html.escape(f"{category}, via {', '.join(row.via[:2])}{more}")
    if category in ("Dependency list", "Environment dump"):
        if row.declaration_basis == "PyPI metadata":
            where = "PyPI metadata"
        else:
            # The shallowest file that names it, and a list the project wrote before a lock.
            paths = sorted(row.declared_in, key=lambda p: (p.endswith(".lock"), p.count("/"), p))
            where = paths[0].rsplit("/", 1)[-1]
        scope = {"optional": ", optional", "dev": ", development"}.get(row.scope, "")
        detail = html.escape(f"{category} ({where}{scope})")
        # The specifier alone: the row is about one package, and the card is narrow.
        spec = row.version_spec
        if spec == "":
            return f"{detail}, any version"
        if spec == "from source":
            return f"{detail}, from source"
        if isinstance(spec, str):
            return f'{detail} <code style="{CODE}">{html.escape(spec)}</code>'
        return detail
    return html.escape(category)


def version_category(df: pd.DataFrame, listed: pd.Series) -> pd.Series:
    """Which versions each project asks for, where it wrote the package into a list."""
    asked = df["version_kind"].astype(object).map(VERSION_LABELS)
    # From source, or a specifier stage 04 could not read.
    asked = asked.where(asked.notna(), "Other")
    category = np.select(
        # A dumped environment records the version that was installed, which nobody chose.
        [listed == "Dependency list", listed == "Environment dump"],
        [asked, "Environment dump"],
        default="Not in a dependency list",
    )
    return pd.Series(category, index=df.index)


def code_category(df: pd.DataFrame, name: str) -> pd.Series:
    """What each project's code does with the library, from stage 05's reading of it."""
    # Importing without calling and naming without importing are one category here: both
    # are code the library appears in and never runs from. The hovercard says which.
    categories = {
        "Calls it": f"Calls {name}",
        "Another library's": f"Another library's {name}",
        "Imports it, calls nothing": f"Mentions {name} only",
        "Names it only": f"Mentions {name} only",
        "No mention": "Not mentioned",
        "No code read": "No code read",
    }
    category = df["code_use"].astype(str).map(categories)
    assert category.notna().all(), "a code use the map has no category for"
    return category


def call_text(call: dict) -> str:
    """One constructor call as its author might have written it, cut to fit a card."""

    def shown(value) -> str:
        if value == EXPRESSION:
            return "…"
        text = repr(value)
        return text if len(text) <= MAX_SHOWN_VALUE else text[: MAX_SHOWN_VALUE - 1] + "…"

    parts = [f"{key}={shown(value)}" for key, value in call["kwargs"].items()]
    if call["star"]:
        parts.append("**…")
    if len(parts) > MAX_SHOWN_ARGUMENTS:
        parts = [*parts[:MAX_SHOWN_ARGUMENTS], f"+{len(parts) - MAX_SHOWN_ARGUMENTS} more"]
    return f"{call['name'].rsplit('.', 1)[-1]}({', '.join(parts)})"


def code_detail(row, category: str, calls: list[dict], features: list[str], sentence: str) -> str:
    """The category with what a card has room to add, as HTML."""
    if row.code_use == "Another library's":
        # The function that computes a layout, before the one that plots it; or, for a
        # project that never names the library, the one that runs it for the project.
        names = [*row.elsewhere] or [*row.through]
        shown = min(names, key=lambda name: (".pl." in name, name))
        return html.escape(f"{category} ({shown})")
    if row.code_use == "Imports it, calls nothing":
        return html.escape(f"{category} (imported, never called)")
    if row.code_use == "Names it only":
        return html.escape(f"{category} (in names, strings or comments)")
    if row.code_use != "Calls it":
        return html.escape(category)
    notes = []
    if calls:
        # The call it makes most often says more than the category would, and holds the
        # arguments that have no colormap of their own.
        shown = f'<code style="{CODE}">{html.escape(call_text(calls[0]))}</code>'
        if len(calls) > 1:
            others = len(calls) - 1
            notes.append(f"+{others} other configuration{'s' if others > 1 else ''}")
    else:
        # Functions only, or methods of an object this reading never saw built.
        shown = html.escape(f"{category} ({row.called[0]})" if len(row.called) else category)
    if features:
        notes.append("Features: " + ", ".join(features))
    shown += "".join(f'<div style="{NOTE}">{html.escape(note)}</div>' for note in notes)
    if sentence:
        # Stage 10's account of what goes in and what becomes of the result.
        shown += f'<div style="{SENTENCE}">{html.escape(sentence)}</div>'
    return shown


def use_detail(row, shown: str, purposes: list[str]) -> str:
    """Every purpose the code shows, the one the map colours by first."""
    if row.code_use != "Calls it":
        # No call to read a purpose from. What the README says of it is the next best.
        stated = isinstance(row.umap_role, str) and row.umap_role not in README_SILENT
        return f"{row.umap_role} (per README)" if stated else shown
    if shown == usage.PLOTTED_AND_CLUSTERED:
        return shown
    return ", ".join([shown, *(p for p in purposes if p != shown)])


def data_detail(row, shown: str) -> str:
    """The kind of data with the form it arrives in, or the model's own words for a kind
    the list lacks."""
    if row.code_use != "Calls it" or row.usage_status != "ok":
        return shown
    if shown == usage.InputData.OTHER.value and row.input_detail.strip():
        detail = row.input_detail.strip()
        return detail[:1].upper() + detail[1:]
    silent = (usage.InputForm.ANY.value, usage.InputForm.UNKNOWN.value)
    if shown in (usage.InputData.ANY.value, CANT_TELL) or row.input_form in silent:
        return shown
    return f"{shown} · {row.input_form.lower()}"


def call_categories(
    target: Target, calls: list[list[dict]], calls_it: np.ndarray, parameter: str
) -> tuple[pd.Series, dict[str, str]]:
    """Each project by the value its calls give one argument, and this colormap's greys."""
    answers = [
        usage.value_given(project, parameter) if direct else None
        for project, direct in zip(calls, calls_it, strict=True)
    ]
    # The default of the class the projects build most, read from the installed library.
    classes = Counter(call["name"].rsplit(".", 1)[-1] for project in calls for call in project)
    signature = class_signature(target.module, classes.most_common(1)[0][0]) if classes else None
    default = signature[1].get(parameter) if signature else None
    default_label = "Default" if default is None else f"Default ({default})"
    given = Counter(json.dumps(a[1]) for a in answers if a is not None and a[0] == usage.VALUE)

    def label(answer) -> str:
        if answer is None:
            return NO_CALL
        kind, value = answer
        if kind == usage.SEVERAL:
            return SEVERAL
        if kind == usage.DEFAULT:
            return default_label
        if kind == usage.UNREADABLE:
            return UNREADABLE
        if given[json.dumps(value)] < MIN_VALUE_PROJECTS:
            return "Other"
        # Beside "Default (euclidean)", a bare "euclidean" would look like the same thing.
        return f"{value}, written out" if value == default else str(value)

    return pd.Series([label(a) for a in answers]), CALL_NEUTRALS | {default_label: DEFAULT_GREY}


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
    # "Use or declare": what stage 06 keeps a project for.
    url = f"{PUBLIC_BASE_URL}{target.slug}/"
    og = {
        "og:title": target.title,
        "og:description": (
            f"{n_documents:,} open-source projects that use or declare {target.package}, "
            "laid out by what their READMEs say and named at four zoom levels. "
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


def version_data_urls(text: str, data_files: list[Path]) -> str:
    """Give each data file's URL a query string that names its contents.

    The page fetches its data separately. A browser that has both cached revalidates the
    page on reload but can go on using its copy of the data, so a rebuilt page runs on the
    previous build's data: a new colormap is in the menu and selecting it does nothing. A
    URL that changes with the contents cannot be answered from an older copy.
    """
    for path in data_files:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()[:12]
        # The page writes each URL as a template string ending in the file's name.
        text, n = re.subn(rf"/{re.escape(path.name)}(?=`)", f"/{path.name}?v={digest}", text)
        assert n == 1, f"the page refers to {path.name} {n} times, not once"
    return text


def orient(coords: np.ndarray) -> np.ndarray:
    """Mirror the layout so that as little of it as possible lies under the title panel.

    Which way round a layout comes out is arbitrary, and a mirror image keeps every
    distance, so the regions are the same ones. Left as it falls, a dense corner can land
    under the panel, which then hides the region names there.
    """
    width, height = DESIGN_SCREEN
    low, high = np.percentile(coords, [0.5, 99.5], axis=0)
    scale = FRAME_SHARE * min(width / (high[0] - low[0]), height / (high[1] - low[1]))
    centred = (coords - (low + high) / 2) * scale
    best = None
    for flip_y in (False, True):
        for flip_x in (False, True):
            # Where each point is drawn: x from the left edge, y from the top.
            x = width / 2 + centred[:, 0] * (-1 if flip_x else 1)
            y = height / 2 - centred[:, 1] * (-1 if flip_y else 1)
            covered = sum(
                int(((x < right) & (y < bottom)).sum()) for right, bottom in CORNER_PANELS
            )
            # Ties go to the fewest mirrorings, so a layout that already fits is left alone.
            if best is None or covered < best[0]:
                best = (covered, flip_x, flip_y)
    covered, flip_x, flip_y = best
    if flip_x or flip_y:
        axes = " and ".join(name for name, flip in (("x", flip_x), ("y", flip_y)) if flip)
        print(f"Mirrored the layout in {axes}; {covered} points are left under the title panel")
    return coords * np.array([-1.0 if flip_x else 1.0, -1.0 if flip_y else 1.0])


def load(target: Target) -> pd.DataFrame:
    documents = pd.read_parquet(target.documents_parquet).drop(columns=["text"])
    enrichment = pd.read_parquet(target.enrichment_parquet).drop(columns=["status"])
    labels = pd.read_parquet(target.labels_parquet)
    # Labels carry the layout, so labels left from an earlier corpus would draw this one in
    # the wrong places without any error.
    assert set(labels["doc_id"]) == set(documents["doc_id"]), (
        "labels.parquet is for a different set of documents; run stage 11 again"
    )
    df = documents.merge(labels, on="doc_id", how="left", validate="one_to_one")
    df = df.merge(enrichment, on="doc_id", how="left", validate="one_to_one")
    declarations = pd.read_parquet(
        target.declarations_parquet,
        columns=[
            "doc_id",
            "status",
            "basis",
            "dumped",
            "declared_in",
            "via",
            "scope",
            "version_spec",
            "version_kind",
        ],
    ).rename(columns={"status": "declaration_status", "basis": "declaration_basis"})
    df = df.merge(declarations, on="doc_id", how="left", validate="one_to_one")
    code = pd.read_parquet(
        target.code_usage_parquet,
        columns=[
            "doc_id",
            "use",
            "called",
            "elsewhere",
            "through",
            "methods",
            "fits_with_target",
            "calls",
            "import_forms",
            "signals",
        ],
    ).rename(columns={"use": "code_use"})
    df = df.merge(code, on="doc_id", how="left", validate="one_to_one")
    # A project whose code was not read has no answer here, which is not a yes.
    df["fits_with_target"] = df["fits_with_target"].astype("boolean").fillna(False).astype(bool)
    described = pd.read_parquet(
        target.usage_parquet,
        columns=[
            "doc_id",
            "status",
            "input_data",
            "input_form",
            "input_detail",
            "purposes",
            "usage_summary",
        ],
    ).rename(columns={"status": "usage_status"})
    df = df.merge(described, on="doc_id", how="left", validate="one_to_one")
    # Stage 10 describes the repositories whose code calls the library, and no others.
    calling = (df["code_use"] == "Calls it") & (df["source"] == "github")
    assert df.loc[calling, "usage_status"].notna().all(), (
        "calling projects stage 10 has not described; run it again"
    )
    assert len(df) == len(documents), "merge changed the row count"
    assert df["declaration_status"].notna().all(), "documents stage 04 has not read"
    assert df["code_use"].notna().all(), "documents stage 05 has not read"
    assert df["x"].notna().all(), "documents without coordinates"
    print(f"{len(df)} documents; {df['summary'].isna().sum()} without a summary")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("target", help="target slug from config.TARGETS")
    args = parser.parse_args()
    target = get_target(args.target)
    df = load(target)
    df[["x", "y"]] = orient(df[["x", "y"]].to_numpy())
    n_listed = pq.read_metadata(target.candidates_parquet).num_rows

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
    # What the code does with the library's output, and what it feeds it, as stage 10 read
    # them. One purpose a project; the card and the search text have them all.
    calls_it = (df["code_use"] == "Calls it").to_numpy()
    described = (df["usage_status"] == "ok").to_numpy()
    purposes = [list(p) if ok else [] for p, ok in zip(df["purposes"], described, strict=True)]
    use_category = pd.Series(
        [
            (one or CANT_TELL) if direct else NO_CALL
            for one, direct in zip(usage.shown_purpose(purposes), calls_it, strict=True)
        ]
    )
    use_values, use_meta, use_colors = categorical_colormap(
        "umap_use", f"{target.name} use", use_category, neutrals=USE_NEUTRALS
    )
    data_values, data_meta, data_colors = categorical_colormap(
        "input_data",
        f"Data {target.name} reduces",
        pd.Series(np.where(calls_it, df["input_data"].fillna(CANT_TELL), NO_CALL)),
        neutrals=USE_NEUTRALS,
    )
    sentences = df["usage_summary"].fillna("").str.strip().where(described, "")
    code_values, code_meta, code_colors = categorical_colormap(
        "in_code", f"{target.name} in the code", code_category(df, target.name)
    )
    listed_values, listed_meta, listed_colors = categorical_colormap(
        "listed_in",
        f"Where {target.package} is listed",
        declaration_category(df),
        LISTED_IN_OVERRIDES,
    )
    version_values, version_meta, _ = categorical_colormap(
        "version_asked",
        f"{target.package} version asked for",
        version_category(df, pd.Series(listed_values, index=df.index)),
    )

    # How the code calls the library, for the projects whose code calls it at all.
    calls = [usage.constructor_calls(project) for project in df["calls"]]
    features = [
        usage.features_used(
            target.features, list(row.called), list(row.methods), bool(row.fits_with_target), c
        )
        if direct
        else []
        for row, c, direct in zip(df.itertuples(), calls, calls_it, strict=True)
    ]
    # One feature a project, the rarest it uses; the card and the search text have them all.
    shown_feature = [
        (feature or NO_FEATURE) if direct else NO_CALL
        for feature, direct in zip(
            usage.rarest(features, [f.label for f in target.features]), calls_it, strict=True
        )
    ]
    call_values, call_metas = [], []
    values, meta, _ = categorical_colormap(
        "features", f"{target.name} features used", pd.Series(shown_feature), neutrals=CALL_NEUTRALS
    )
    call_values.append(values)
    call_metas.append(meta)
    for parameter, description in target.value_colormaps:
        category, greys = call_categories(target, calls, calls_it, parameter)
        values, meta, _ = categorical_colormap(parameter, description, category, neutrals=greys)
        call_values.append(values)
        call_metas.append(meta)
    for parameter, description in target.given_colormaps:
        category = pd.Series(
            [
                GIVEN_LABELS[usage.how_often_given(project, parameter)] if direct else NO_CALL
                for project, direct in zip(calls, calls_it, strict=True)
            ]
        )
        values, meta, _ = categorical_colormap(
            f"{parameter}_given", description, category, neutrals=CALL_NEUTRALS
        )
        call_values.append(values)
        call_metas.append(meta)
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
            "umap_use": [
                html.escape(use_detail(row, shown, found))
                for row, shown, found in zip(df.itertuples(), use_values, purposes, strict=True)
            ],
            "use_dot": [use_colors[v] for v in use_values],
            "data": [
                html.escape(data_detail(row, shown))
                for row, shown in zip(df.itertuples(), data_values, strict=True)
            ],
            "data_dot": [data_colors[v] for v in data_values],
            "in_code": [
                code_detail(row, category, c, f, sentence)
                for row, category, c, f, sentence in zip(
                    df.itertuples(), code_values, calls, features, sentences, strict=True
                )
            ],
            "code_dot": [code_colors[v] for v in code_values],
            "listed_in": [
                declaration_detail(row, category)
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
                        # What the code calls, by its dotted name, so that searching for a
                        # class or function finds the projects that use it.
                        " ".join([*row.called, *row.elsewhere, *row.through]),
                        # Every feature and every argument value, which the colormaps
                        # show one of: "metric=cosine" finds each project that sets it.
                        " ".join([*used, *usage.argument_tokens(c)]),
                        " ".join([*row.import_forms, *row.signals]),
                        # What stage 10 read in the code: the purposes, the data, and the
                        # sentence, which names the models and methods beside the call.
                        " ".join([*found, sentence])
                        + (f" {row.input_data} {row.input_form} {row.input_detail}" if ok else ""),
                        # The requirement as pip would write it: "umap-learn>=0.5".
                        f"{target.package}{row.version_spec}"
                        if isinstance(row.version_spec, str) and row.version_spec[:1] in SPEC_STARTS
                        else "",
                    ]
                )
                for row, text, domain, kind, used, c, found, sentence, ok in zip(
                    df.itertuples(),
                    summary,
                    domain_label,
                    type_label,
                    features,
                    calls,
                    purposes,
                    sentences,
                    described,
                    strict=True,
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
            use_values,
            data_values,
            code_values,
            *call_values,
            listed_values,
            version_values,
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
            use_meta,
            data_meta,
            code_meta,
            *call_metas,
            listed_meta,
            version_meta,
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
            # GitHub's dependency graph also counts lock files and dumped environments.
            # Stage 06 left out the projects listed only for those, and the count of
            # all it lists says so.
            f"{len(df):,} projects that use or declare {target.package}, "
            f"of the {n_listed:,} GitHub lists."
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
        data_files = sorted(output.parent.glob(f"{DATA_PREFIX}_*"))
        assert data_files, "no data files were written beside the page"
        text = version_data_urls(text, data_files)
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
        "listed": n_listed,
        "crawled": f"{crawled.max():%Y-%m}",
    }
    write_bytes_safely(target.map_meta_json, (json.dumps(meta, indent=2) + "\n").encode())

    print(f"Wrote {output.relative_to(ROOT)} ({output.stat().st_size / 1e3:.0f} KB) and:")
    for path in data_files:
        print(f"  {path.name} ({path.stat().st_size / 1e3:.0f} KB)")
    total = output.stat().st_size + sum(path.stat().st_size for path in data_files)
    print(
        f"Total {total / 1e6:.2f} MB. Serve {output.parent.relative_to(ROOT)}/ over HTTP to view."
    )


if __name__ == "__main__":
    main()
