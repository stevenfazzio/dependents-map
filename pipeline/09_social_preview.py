"""Screenshot the rendered map for its link card: docs/<target>/social-preview.png.

The card is the interactive map itself rather than a separate static plot, so a shared
link previews what the reader will get. The map is WebGL, so this drives the installed
Chrome through Playwright; no bundled browser is downloaded.

Usage:
    uv run python pipeline/09_social_preview.py umap-learn
    uv run python pipeline/09_social_preview.py umap-learn --zoom 2 --out /tmp/try.png
"""

import argparse
import functools
import http.server
import io
import threading
from pathlib import Path

from config import ROOT, get_target
from PIL import Image
from playwright.sync_api import sync_playwright
from storage import write_bytes_safely

CARD_SIZE = (1200, 630)
SCALE = 2  # capture at 2x and downsample, so labels and points stay crisp
SETTLE_MS = 6000

# The controls do nothing in a still image, and link previews print the page title
# under the card, so the whole frame can go to the map.
HIDE_PANELS = """
for (const id of ['search-container', 'colormap-selector-container', 'title-container']) {
  const el = document.getElementById(id);
  if (el) (el.closest('.container-box') || el).style.display = 'none';
}
"""


class QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass


def capture(directory: Path, zoom: int, pan_x: int, pan_y: int) -> bytes:
    """Serve the map's directory, open it in Chrome, frame it, and return a PNG."""
    handler = functools.partial(QuietHandler, directory=str(directory))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    width, height = CARD_SIZE
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(channel="chrome")
            page = browser.new_page(
                viewport={"width": width, "height": height}, device_scale_factor=SCALE
            )
            page.goto(f"http://127.0.0.1:{server.server_port}/", wait_until="load", timeout=120_000)
            page.wait_for_load_state("networkidle", timeout=45_000)
            page.wait_for_timeout(SETTLE_MS)
            page.mouse.move(width / 2, height / 2)
            for _ in range(zoom):
                page.mouse.wheel(0, -50)
                page.wait_for_timeout(150)
            if pan_x or pan_y:
                page.mouse.down()
                page.mouse.move(width / 2 + pan_x, height / 2 + pan_y, steps=8)
                page.mouse.up()
            page.mouse.move(0, 0)  # park the cursor off the points: no hovercard
            page.evaluate(HIDE_PANELS)
            # Labels fade in and out as their collision boxes settle after a move.
            page.wait_for_timeout(4000)
            shot = page.screenshot()
            browser.close()
    finally:
        server.shutdown()
    return shot


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("target", help="target slug from config.TARGETS")
    # The defaults are the framing chosen by eye for the umap-learn map: two ticks in is
    # the first zoom at which all six top-level names show at full strength in a frame
    # this size, and the nudge centres the map.
    parser.add_argument("--zoom", type=int, default=2, help="wheel ticks in from the default view")
    parser.add_argument("--pan-x", type=int, default=30, help="drag the map right by N px")
    parser.add_argument("--pan-y", type=int, default=0, help="drag the map down by N px")
    parser.add_argument("--out", type=Path, help="write somewhere else, to compare framings")
    args = parser.parse_args()
    target = get_target(args.target)
    out = args.out or target.social_preview_png

    shot = capture(target.map_html.parent, args.zoom, args.pan_x, args.pan_y)
    with Image.open(io.BytesIO(shot)) as image:
        card = image.convert("RGB").resize(CARD_SIZE, Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    card.save(buffer, format="PNG", optimize=True)
    write_bytes_safely(out, buffer.getvalue())
    shown = out.relative_to(ROOT) if out.is_relative_to(ROOT) else out
    print(f"Wrote {shown} {card.size} ({out.stat().st_size / 1e3:.0f} KB)")


if __name__ == "__main__":
    main()
