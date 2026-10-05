"""Build docs/index.html, the page that lists the published maps.

It covers every target in config.TARGETS whose map has been rendered, reading the
facts the render stage left in docs/<target>/meta.json. Run it after rendering.

Usage:
    uv run python pipeline/build_index.py
"""

import html
import json
from datetime import datetime

from config import DOCS_DIR, PUBLIC_BASE_URL, REPO_URL, ROOT, TARGETS
from storage import write_bytes_safely

TITLE = "Dependents Maps"
INTRO = (
    "Each map lays out the open-source projects that depend on one package, placed by "
    "what their READMEs say. Projects that sit close together do similar things."
)

# The preview image is the page: it is the one thing here that shows what a map is, so
# it runs the full width of the column and everything around it stays quiet. The type
# is IBM Plex Sans, the same face the maps use.
PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
{head_tags}
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&display=swap">
<style>
  :root {{ --ink: #1f2328; --muted: #6b7280; --rule: #e5e7eb; }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0; padding: 56px 24px 72px;
    font-family: "IBM Plex Sans", system-ui, sans-serif; font-size: 17px; line-height: 1.5;
    color: var(--ink); background: #fff;
  }}
  main {{ max-width: 960px; margin: 0 auto; }}
  h1 {{ margin: 0 0 12px; font-size: 34px; font-weight: 600; line-height: 1.15;
       letter-spacing: -0.015em; }}
  .intro {{ max-width: 62ch; margin: 0 0 44px; color: var(--muted); }}
  article {{ margin: 0 0 56px; }}
  article a.preview {{ display: block; border: 1px solid var(--rule); border-radius: 10px;
                      overflow: hidden; }}
  article img {{ display: block; width: 100%; height: auto; aspect-ratio: 1200 / 630; }}
  h2 {{ margin: 18px 0 4px; font-size: 22px; font-weight: 600; line-height: 1.25; }}
  h2 a {{ color: inherit; text-decoration: none; }}
  h2 a:hover {{ text-decoration: underline; text-underline-offset: 3px; }}
  article p {{ max-width: 62ch; margin: 0; color: var(--muted); }}
  footer {{ max-width: 62ch; padding-top: 20px; border-top: 1px solid var(--rule);
           font-size: 15px; color: var(--muted); }}
  footer a {{ color: var(--ink); text-underline-offset: 3px; }}
  a:focus-visible {{ outline: 2px solid var(--ink); outline-offset: 3px; border-radius: 4px; }}
  @media (max-width: 600px) {{
    body {{ padding: 32px 16px 48px; font-size: 16px; }}
    h1 {{ font-size: 28px; }}
    h2 {{ font-size: 20px; }}
  }}
</style>
</head>
<body>
<main>
  <h1>{title}</h1>
  <p class="intro">{intro}</p>
{entries}
  <footer>
    Built by <a href="https://stevenfazzio.com/">Steven Fazzio</a>.
    The code and the method are <a href="{repo_url}">on GitHub</a>.
  </footer>
</main>
</body>
</html>
"""

ENTRY = """  <article>
    <a class="preview" href="{slug}/" aria-hidden="true" tabindex="-1">
      <img src="{slug}/social-preview.png" width="1200" height="630" alt="">
    </a>
    <h2><a href="{slug}/">{title}</a></h2>
    <p>{description}</p>
  </article>"""


def entry(slug: str, meta: dict) -> str:
    crawled = datetime.strptime(meta["crawled"], "%Y-%m")
    description = (
        f"{meta['documents']:,} projects that use or declare {meta['package']}, of the "
        f"{meta['listed']:,} that GitHub's dependency graph listed in {crawled:%B %Y}."
    )
    return ENTRY.format(
        slug=slug, title=html.escape(meta["title"]), description=html.escape(description)
    )


def head_tags(first_slug: str) -> str:
    og = {
        "og:title": TITLE,
        "og:description": INTRO,
        "og:type": "website",
        "og:url": PUBLIC_BASE_URL,
        "og:image": f"{PUBLIC_BASE_URL}{first_slug}/social-preview.png",
    }
    tags = [
        f'<meta property="{key}" content="{html.escape(value, quote=True)}">'
        for key, value in og.items()
    ]
    tags.append(f'<meta name="description" content="{html.escape(INTRO, quote=True)}">')
    tags.append('<meta name="twitter:card" content="summary_large_image">')
    return "\n".join(tags)


def main() -> None:
    published = {
        slug: json.loads(target.map_meta_json.read_text())
        for slug, target in TARGETS.items()
        if target.map_meta_json.exists()
    }
    if not published:
        raise SystemExit("No rendered maps found under docs/. Run pipeline/12_render.py first.")

    page = PAGE.format(
        title=TITLE,
        intro=html.escape(INTRO),
        head_tags=head_tags(next(iter(published))),
        entries="\n".join(entry(slug, meta) for slug, meta in published.items()),
        repo_url=REPO_URL,
    )
    write_bytes_safely(DOCS_DIR / "index.html", page.encode())
    # GitHub Pages runs Jekyll over the site unless told not to; these are plain files.
    (DOCS_DIR / ".nojekyll").touch()
    print(f"Wrote {(DOCS_DIR / 'index.html').relative_to(ROOT)} listing {', '.join(published)}")


if __name__ == "__main__":
    main()
