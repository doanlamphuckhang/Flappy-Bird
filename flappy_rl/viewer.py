"""Build a self-contained HTML viewer with an embedded replay and weights."""
import html
import json
from pathlib import Path

VIEWER_TEMPLATE = Path(__file__).resolve().parent / "assets" / "flappy_3d.html"
DATA_PLACEHOLDER = "/*__FLAPPY_DATA__*/"


# Escaping these keeps embedded JSON from closing the <script> or opening an HTML comment.
_JS_ESCAPES = str.maketrans({"<": "\\u003c", ">": "\\u003e", "&": "\\u0026"})


def _as_js(obj):
    return json.dumps(obj, separators=(",", ":")).translate(_JS_ESCAPES)


def build_3d_viewer(replay=None, weights=None, template=VIEWER_TEMPLATE, out_path="flappy_3d_embedded.html"):
    """Write the viewer page with `window.FLAPPY_REPLAY` / `window.FLAPPY_WEIGHTS` injected."""
    page = Path(template).read_text(encoding="utf-8")
    if DATA_PLACEHOLDER not in page:
        raise ValueError(f"{template} has no {DATA_PLACEHOLDER} placeholder")
    script = ""
    if weights is not None:
        script += f"window.FLAPPY_WEIGHTS={_as_js(weights)};"
    if replay is not None:
        script += f"window.FLAPPY_REPLAY={_as_js(replay)};"
    out_path = Path(out_path)
    out_path.write_text(page.replace(DATA_PLACEHOLDER, script, 1), encoding="utf-8")
    print(f"Viewer: {out_path} ({out_path.stat().st_size / 1024:.0f} KB)")
    return out_path


def show_3d(path="flappy_3d_embedded.html", height=640):
    """Display the viewer inline in a notebook (Three.js is loaded from a CDN)."""
    from IPython.display import HTML

    srcdoc = html.escape(Path(path).read_text(encoding="utf-8"), quote=True)
    return HTML(f'<iframe srcdoc="{srcdoc}" allow="fullscreen" allowfullscreen '
                f'style="width:100%;height:{height}px;border:0;border-radius:14px;background:#0b1020"></iframe>')
