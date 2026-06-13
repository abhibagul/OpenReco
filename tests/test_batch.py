"""Batch processing — discovery + multi-project run (dummy stages, CI-safe)."""

from __future__ import annotations

import textwrap

from openreco.batch import discover_projects, run_batch

_PROJ = """
[project]
name = "{name}"
[[stage]]
id = "gen"
type = "dummy_generate"
params = {{ n = {n} }}
[[stage]]
id = "sum"
type = "dummy_sum"
inputs = ["gen"]
"""


def _make(root, name, n):
    d = root / name
    d.mkdir()
    (d / "project.toml").write_text(textwrap.dedent(_PROJ.format(name=name, n=n)), encoding="utf-8")
    return d


def test_discover_projects(tmp_path):
    _make(tmp_path, "a", 3)
    _make(tmp_path, "b", 4)
    found = discover_projects(tmp_path)
    assert len(found) == 2
    # a directory that itself holds a project.toml resolves to just itself
    assert discover_projects(tmp_path / "a") == [tmp_path / "a"]


def test_run_batch_sequential(tmp_path):
    _make(tmp_path, "a", 3)
    _make(tmp_path, "b", 4)
    results = run_batch(discover_projects(tmp_path))
    assert len(results) == 2
    assert all(r["ok"] for r in results)
    assert {r["project"] for r in results} == {"a", "b"}
    assert all(r["stages"] == 2 for r in results)


def test_run_batch_reports_failure(tmp_path):
    # a project whose dummy_sum has no input -> stage fails -> project not ok
    d = tmp_path / "bad"
    d.mkdir()
    (d / "project.toml").write_text(
        '[project]\nname="bad"\n[[stage]]\nid="s"\ntype="dummy_sum"\n', encoding="utf-8")
    results = run_batch([d])
    assert len(results) == 1 and results[0]["ok"] is False
