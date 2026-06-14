"""Processing report — a self-contained HTML summary for auditability and reproducibility.

Surfaces the headline quality numbers (registration, reprojection error, GPS residuals, CRS,
point/mesh/raster sizes), QA issues grouped by severity, a per-stage table, and a
reproducibility block (tool versions + the exact resolved parameters and content-address keys
for every stage). Everything here is derived from the run record — no recomputation.
"""

from __future__ import annotations

import html
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from openreco.engine.runner import RunOutcome

_STATUS_COLOR = {
    "executed": "#2563eb", "cached": "#16a34a", "failed": "#dc2626",
    "skipped": "#a16207", "cancelled": "#6b7280",
}
_SEV_COLOR = {"error": "#dc2626", "warning": "#d97706", "info": "#2563eb"}


def _e(x: Any) -> str:
    return html.escape(str(x))


def _summary_cards(stages: list[dict]) -> str:
    """Pull a few headline metrics from known stage *types* into highlight cards (works regardless
    of how layers are named in the UI)."""
    by_type: dict[str, dict] = {}
    for s in stages:                              # first stage of each type wins
        by_type.setdefault(s["type"], s)
    cards: list[tuple[str, str]] = []

    def metric(stype: str, key: str):
        return by_type.get(stype, {}).get("metrics", {}).get(key)

    if (reg := metric("sfm", "reg_images")) is not None:
        cards.append(("images registered", f"{reg} / {metric('sfm', 'input_images')}"))
    if (err := metric("sfm", "mean_reproj_error")) is not None:
        cards.append(("mean reprojection error", f"{err} px"))
    if (pts := metric("sfm", "points3D")) is not None:
        cards.append(("sparse points", f"{pts:,}"))
    if (crs := metric("georef", "crs")) is not None:
        cards.append(("coordinate system", str(crs)))
    if (rms := metric("georef", "rms_residual_m")) is not None:
        label = "GCP control RMS" if metric("georef", "method") == "gcp" else "GPS alignment RMS"
        cards.append((label, f"{rms} m"))
    if (chk := metric("georef", "check_rms_m")) is not None:
        cards.append(("GCP check RMS", f"{chk} m ({metric('georef', 'num_check')} pts)"))
    if (np_ := metric("mvs", "num_points")) is not None:
        mode = metric("mvs", "mode") or ""
        cards.append(("dense points", f"{np_:,}" + (f" ({mode})" if mode else "")))
    if (faces := metric("mesh", "faces")) is not None:
        cards.append(("mesh faces", f"{faces:,}"))
    if (cov := metric("texture", "atlas_coverage")) is not None:
        cards.append(("texture coverage", f"{int(cov * 100)}%"))
    if (w := metric("dsm", "width")) is not None:
        cards.append(("DSM size", f"{w}×{metric('dsm', 'height')} px"))
    if metric("classify", "ground_pct") is not None:
        cards.append(("ground / building / veg",
                      f"{metric('classify', 'ground')} / {metric('classify', 'building')} / "
                      f"{metric('classify', 'vegetation')}"))
    if (mo := metric("coverage", "max_overlap")) is not None:
        cards.append(("max image overlap", f"{mo}×"))

    if not cards:
        return ""
    items = "".join(
        f"<div class=card><div class=cval>{_e(v)}</div><div class=clabel>{_e(k)}</div></div>"
        for k, v in cards
    )
    return f"<div class=cards>{items}</div>"


def _issues_section(stages: list[dict]) -> str:
    buckets: dict[str, list[str]] = {"error": [], "warning": [], "info": []}
    for s in stages:
        for i in s["issues"]:
            sev = i["severity"]
            hint = f" <span class=hint>— {_e(i['hint'])}</span>" if i.get("hint") else ""
            buckets.setdefault(sev, []).append(
                f"<li><code>{_e(s['id'])}</code>: {_e(i['message'])}{hint}</li>"
            )
    blocks = []
    for sev in ("error", "warning", "info"):
        if buckets.get(sev):
            color = _SEV_COLOR[sev]
            blocks.append(
                f"<h3 style='color:{color}'>{sev}s ({len(buckets[sev])})</h3>"
                f"<ul>{''.join(buckets[sev])}</ul>"
            )
    return "<h2>QA issues</h2>" + ("".join(blocks) if blocks else "<p class=muted>none</p>")


def _stage_rows(stages: list[dict]) -> str:
    rows = []
    for s in stages:
        color = _STATUS_COLOR.get(s["status"], "#374151")
        metrics = ", ".join(f"{_e(k)}={_e(v)}" for k, v in s["metrics"].items())
        rows.append(
            f"<tr><td><code>{_e(s['id'])}</code></td><td>{_e(s['type'])}</td>"
            f"<td><span style='color:{color};font-weight:600'>{_e(s['status'])}</span></td>"
            f"<td>{s['seconds']:.3f}s</td>"
            f"<td>{metrics or '<span class=muted>—</span>'}</td></tr>"
        )
    return "".join(rows)


def _repro_block(d: dict) -> str:
    plat = d["platform"]
    rows = []
    for s in d["stages"]:
        params = ", ".join(f"{_e(k)}={_e(v)}" for k, v in s.get("params", {}).items())
        rows.append(
            f"<tr><td><code>{_e(s['id'])}</code></td>"
            f"<td><code class=muted>{_e(s['key'][:16])}</code></td>"
            f"<td>{params or '<span class=muted>—</span>'}</td></tr>"
        )
    return (
        "<h2>Reproducibility</h2>"
        f"<p class=meta>openreco {_e(d['openreco_version'])} · python {_e(plat['python'])} · "
        f"{_e(plat['system'])}/{_e(plat['machine'])}</p>"
        "<table><thead><tr><th>stage</th><th>cache key</th><th>resolved parameters</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
        "<p class=muted>Each key = hash(stage type + version + resolved params + upstream keys). "
        "Re-running the same manifest reuses cached stages; changing any parameter recomputes only "
        "the affected sub-graph.</p>"
    )


def write_report(outcome: "RunOutcome", path: Path) -> None:
    d = outcome.to_dict()
    badge = ("<span style='color:#16a34a'>OK</span>" if d["ok"]
             else "<span style='color:#dc2626'>FAILED</span>")
    total_time = sum(s["seconds"] for s in d["stages"])
    body = f"""<!doctype html>
<html lang=en><head><meta charset=utf-8>
<title>OpenReco report — {_e(d['project'])}</title>
<style>
 body {{ font: 14px/1.55 system-ui, sans-serif; margin: 2rem; color: #111; max-width: 1000px; }}
 h1 {{ font-size: 1.5rem; margin-bottom: .2rem; }}
 h2 {{ font-size: 1.1rem; margin-top: 1.8rem; border-bottom: 1px solid #e5e7eb; padding-bottom: .3rem; }}
 h3 {{ font-size: .95rem; margin: .8rem 0 .3rem; }}
 table {{ border-collapse: collapse; width: 100%; margin-top: .6rem; }}
 th, td {{ text-align: left; padding: .45rem .7rem; border-bottom: 1px solid #eee; vertical-align: top; }}
 th {{ background: #f9fafb; }}
 code {{ background: #f3f4f6; padding: 0 .25rem; border-radius: 3px; }}
 .muted {{ color: #9ca3af; }} .hint {{ color: #6b7280; }} .meta {{ color: #4b5563; }}
 .cards {{ display: flex; flex-wrap: wrap; gap: .8rem; margin-top: 1rem; }}
 .card {{ background: #f8fafc; border: 1px solid #e5e7eb; border-radius: 8px; padding: .7rem 1rem; min-width: 140px; }}
 .cval {{ font-size: 1.25rem; font-weight: 700; }} .clabel {{ color: #6b7280; font-size: .8rem; }}
 ul {{ margin: .2rem 0; }}
</style></head><body>
<h1>OpenReco — {_e(d['project'])} {badge}</h1>
<p class=meta>started {_e(d['started'])} · {total_time:.1f}s total · {len(d['stages'])} stages</p>
{_summary_cards(d['stages'])}
{_issues_section(d['stages'])}
<h2>Stages</h2>
<table><thead><tr><th>stage</th><th>type</th><th>status</th><th>time</th><th>metrics</th></tr></thead>
<tbody>{_stage_rows(d['stages'])}</tbody></table>
{_repro_block(d)}
</body></html>"""
    path.write_text(body, encoding="utf-8")
