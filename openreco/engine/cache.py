"""Content-addressed cache — the heart of reproducibility, checkpointing, and resume.

A stage's cache key is:

    sha256( stage_type + impl_version + canonical(params) + [sorted upstream keys] )

Because upstream keys are themselves content addresses, any change to params anywhere
upstream propagates downstream automatically. A completed stage writes a `.done` marker
plus `result.json`. On the next run, a present marker = cache hit = skip (this is both
checkpointing and resume). Bumping a stage's `version` or changing any param yields a new
key, so old outputs are never silently reused — and old caches remain on disk for `diff`.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openreco.engine.context import StageResult

DONE_MARKER = ".done"
RESULT_FILE = "result.json"
KEYINFO_FILE = "keyinfo.json"


def canonical_json(obj: Any) -> str:
    """Stable JSON for hashing: sorted keys, no insignificant whitespace."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def compute_key(stage_type: str, version: str, params: dict[str, Any], input_keys: list[str]) -> str:
    payload = {
        "type": stage_type,
        "version": version,
        "params": params,
        "inputs": sorted(input_keys),
    }
    digest = hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()
    return digest


@dataclass
class CacheEntry:
    key: str
    dir: Path

    @property
    def done(self) -> bool:
        return (self.dir / DONE_MARKER).exists()

    def load_result(self) -> StageResult:
        data = json.loads((self.dir / RESULT_FILE).read_text(encoding="utf-8"))
        return StageResult.from_dict(data)


class Cache:
    """Filesystem cache rooted at <project>/.openreco/cache/<key>/."""

    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def entry(self, key: str) -> CacheEntry:
        return CacheEntry(key=key, dir=self.root / key)

    def open_for_write(self, key: str) -> Path:
        """Create (and clean any partial) cache dir for a stage about to run."""
        d = self.root / key
        if d.exists():
            # remove a previous incomplete attempt (no .done marker)
            if not (d / DONE_MARKER).exists():
                _rmtree(d)
        d.mkdir(parents=True, exist_ok=True)
        return d

    def finalize(self, key: str, result: StageResult, keyinfo: dict[str, Any]) -> None:
        d = self.root / key
        (d / RESULT_FILE).write_text(
            json.dumps(result.to_dict(), indent=2, sort_keys=True), encoding="utf-8"
        )
        (d / KEYINFO_FILE).write_text(
            json.dumps(keyinfo, indent=2, sort_keys=True), encoding="utf-8"
        )
        (d / DONE_MARKER).write_text("ok", encoding="utf-8")


def _rmtree(path: Path) -> None:
    import shutil

    shutil.rmtree(path, ignore_errors=True)
