"""Inline dashboard/data.json into template.html to produce a single
self-contained dashboard/index.html (no runtime fetch, per Artifact rules).
Run export_data.py first (or let this script call it) to refresh data.json.

Also mirrors the same file to docs/index.html at the repo root -- unlike
dashboard/index.html (a build artifact, gitignored), docs/ IS committed:
it's what GitHub Pages serves as the project's public website (Settings ->
Pages -> Deploy from a branch -> this branch -> /docs). Keeping both in
sync from one build means "rebuild and push" is the entire publish step
for the public site, including feedback-driven updates.
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).parent
DOCS_DIR = HERE.parent / "docs"


def build() -> Path:
    data_path = HERE / "data.json"
    if not data_path.exists():
        from dashboard.export_data import export
        export(data_path)

    data = json.loads(data_path.read_text())
    template = (HERE / "template.html").read_text()
    # Re-serialize compactly and escape "</script>" so it can't close the tag early.
    payload = json.dumps(data).replace("</script>", "<\\/script>")
    output = template.replace("__DASHBOARD_DATA__", payload)

    out_path = HERE / "index.html"
    out_path.write_text(output)
    print(f"Wrote {out_path} ({out_path.stat().st_size} bytes)")

    DOCS_DIR.mkdir(exist_ok=True)
    docs_path = DOCS_DIR / "index.html"
    docs_path.write_text(output)
    print(f"Wrote {docs_path} ({docs_path.stat().st_size} bytes)")

    return out_path


if __name__ == "__main__":
    build()
