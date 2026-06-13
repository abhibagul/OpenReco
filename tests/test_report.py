"""Processing-report rendering — exercises summary cards, QA grouping, and the repro block
without running a real pipeline (constructs a RunOutcome directly)."""

from __future__ import annotations

from openreco.engine.context import Issue, Severity
from openreco.engine.report import write_report
from openreco.engine.runner import RunOutcome, StageRun, StageStatus


def _outcome(run_dir):
    stages = [
        StageRun(
            id="sfm", type="sfm", key="abc123def456ghi7", status=StageStatus.EXECUTED,
            seconds=12.5,
            metrics={"reg_images": 5, "input_images": 8, "mean_reproj_error": 0.63, "points3D": 8541},
            issues=[Issue(Severity.WARNING, "3/8 images not registered", hint="add overlap")],
            params={"matcher": "exhaustive", "max_image_size": 1600},
        ),
        StageRun(
            id="georef", type="georef", key="def456ghi789jkl0", status=StageStatus.EXECUTED,
            seconds=0.5, metrics={"crs": "EPSG:32613", "rms_residual_m": 2.74, "method": "gps"},
            issues=[], params={"method": "auto"},
        ),
    ]
    return RunOutcome(project="t", started="2026-06-13T00:00:00Z", finished="2026-06-13T00:00:13Z",
                      ok=True, stages=stages, run_dir=run_dir)


def test_report_has_cards_qa_and_repro(tmp_path):
    out = _outcome(tmp_path)
    p = tmp_path / "report.html"
    write_report(out, p)
    h = p.read_text(encoding="utf-8")
    # summary cards
    assert "images registered" in h and "5 / 8" in h
    assert "mean reprojection error" in h and "0.63 px" in h
    assert "EPSG:32613" in h and "GPS alignment RMS" in h and "2.74 m" in h
    # QA grouped by severity
    assert "QA issues" in h and "warning" in h.lower()
    assert "3/8 images not registered" in h and "add overlap" in h
    # reproducibility block surfaces resolved params + cache keys
    assert "Reproducibility" in h and "resolved parameters" in h
    assert "matcher=exhaustive" in h and "abc123def456ghi7"[:16] in h


def test_report_handles_no_metrics(tmp_path):
    out = RunOutcome(project="empty", started="t", finished="t", ok=True,
                     stages=[StageRun(id="x", type="dummy_sum", key="k" * 16,
                                      status=StageStatus.EXECUTED)],
                     run_dir=tmp_path)
    p = tmp_path / "r.html"
    write_report(out, p)
    h = p.read_text(encoding="utf-8")
    assert "OpenReco" in h and "QA issues" in h and "none" in h
