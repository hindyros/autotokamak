# provenance: Human/Claude-authored platform code (engineered, not agent-generated)
"""Flatten a generated trace_to_html report dir into ONE self-contained HTML file.

The report written by ``tools/trace_to_html.py`` is a *directory*: an
``index.html`` that links to per-run pages (``<run_id>.html``), each of which
embeds PNGs from ``physics/``, ``eval/`` and ``meta/`` subdirs via relative
paths. Emailing only ``index.html`` therefore breaks every link and image on
the recipient's machine.

This script inlines the whole thing into a single file that opens anywhere with
no external dependencies:

  * every ``<run_id>.html`` body is folded in as an in-page ``#run-<id>`` section
  * cross-page links (``<id>.html`` / ``index.html``) become in-page anchors
  * every PNG is embedded as a ``data:image/png;base64,...`` URI

Usage:
    python tools/bundle_report.py                       # <experiments>/_report -> report_standalone.html
    python tools/bundle_report.py --report DIR --out FILE

Zero dependencies beyond the stdlib.
"""

from __future__ import annotations

import argparse
import base64
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

BODY_RE = re.compile(r"<body[^>]*>(.*)</body>", re.IGNORECASE | re.DOTALL)
HEAD_RE = re.compile(r"<head[^>]*>(.*)</head>", re.IGNORECASE | re.DOTALL)
# href/src pointing at a PNG under physics/ eval/ meta/ (optionally wrapped in <a>)
IMG_REF_RE = re.compile(r'(href|src)="((?:physics|eval|meta)/[^"]+\.png)"')
# links to sibling report pages
PAGE_LINK_RE = re.compile(r'href="([^"/]+)\.html"')


def _body(html_text: str) -> str:
    m = BODY_RE.search(html_text)
    return m.group(1) if m else html_text


def _head(html_text: str) -> str:
    m = HEAD_RE.search(html_text)
    return m.group(1) if m else ""


def _inline_images(html_text: str, report_dir: Path) -> tuple[str, int]:
    """Replace href/src=".../x.png" with a base64 data URI. Missing files left as-is."""
    count = 0

    def repl(m: re.Match) -> str:
        nonlocal count
        attr, rel = m.group(1), m.group(2)
        f = report_dir / rel
        if not f.is_file():
            return m.group(0)
        data = base64.b64encode(f.read_bytes()).decode("ascii")
        count += 1
        return f'{attr}="data:image/png;base64,{data}"'

    return IMG_REF_RE.sub(repl, html_text), count


def _rewrite_page_links(html_text: str) -> str:
    def repl(m: re.Match) -> str:
        name = m.group(1)
        if name == "index":
            return 'href="#top"'
        return f'href="#run-{name}"'

    return PAGE_LINK_RE.sub(repl, html_text)


def bundle(report_dir: Path, out_file: Path) -> None:
    index_path = report_dir / "index.html"
    if not index_path.is_file():
        raise SystemExit(f"no index.html in {report_dir}")

    index_html = index_path.read_text()
    head = _head(index_html)  # reuse the index stylesheet for the whole document
    index_body = _body(index_html)

    # Discover run pages from the links the index actually renders, in order.
    run_ids = []
    for m in PAGE_LINK_RE.finditer(index_body):
        name = m.group(1)
        if name != "index" and name not in run_ids and (report_dir / f"{name}.html").is_file():
            run_ids.append(name)

    total_imgs = 0
    sections = []
    for rid in run_ids:
        page = (report_dir / f"{rid}.html").read_text()
        body, n = _inline_images(_body(page), report_dir)
        total_imgs += n
        sections.append(f'<section id="run-{rid}" class="run-section">\n{body}\n</section>')

    index_body, n = _inline_images(index_body, report_dir)
    total_imgs += n

    banner = (
        '<p class="muted" style="margin:0 0 16px">'
        "Self-contained report — every run page and figure is embedded in this single file. "
        "Click a run below to jump to its detail; use your browser's Back button to return."
        "</p>"
    )

    doc = (
        "<!doctype html><html><head><meta charset='utf-8'>"
        + head
        + "<style>.run-section{border-top:3px solid #e1e4e8;margin-top:48px;padding-top:8px}</style>"
        + "</head><body><a id='top'></a>"
        + banner
        + index_body
        + "\n".join(sections)
        + "</body></html>"
    )
    doc = _rewrite_page_links(doc)

    out_file.write_text(doc)
    size_mb = out_file.stat().st_size / 1_048_576
    print(f"bundled {len(run_ids)} run pages + {total_imgs} images -> {out_file} ({size_mb:.1f} MB)")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--report", default=str(REPO_ROOT / "experiments" / "_report"),
                   help="Report dir produced by trace_to_html.py")
    p.add_argument("--out", default=None,
                   help="Output file (default: <report>/report_standalone.html)")
    args = p.parse_args()

    report_dir = Path(args.report)
    out_file = Path(args.out) if args.out else report_dir / "report_standalone.html"
    bundle(report_dir, out_file)


if __name__ == "__main__":
    main()
