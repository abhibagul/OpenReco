"""OpenReco desktop UI — a local web application over the engine.

`openreco ui [project]` starts a zero-dependency (stdlib) local server that serves a single-page
app: a layer tree (the DAG's stages + their artifacts), auto-generated parameter panels (from each
stage's params_schema), a run panel with live progress (Server-Sent Events from the engine's
on_event hook), and a 3D viewport (three.js). Editing in the UI edits the manifest; the engine's
content-addressed cache gives reproducibility + cheap re-runs. Wrap in a webview later for a true
desktop window.
"""

from pathlib import Path

WEB_DIR = Path(__file__).resolve().parent / "web"
