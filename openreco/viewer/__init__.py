"""Static web viewer (three.js). The `template/` dir holds the shareable site scaffold that
the export stage fills in with the project's artifacts."""

from pathlib import Path

TEMPLATE_DIR = Path(__file__).resolve().parent / "template"
