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
    "executed": "#1d4ed8", "cached": "#16a34a", "failed": "#dc2626",
    "skipped": "#a16207", "cancelled": "#6b7280",
}
_SEV_COLOR = {"error": "#dc2626", "warning": "#b7791f", "info": "#1d4ed8"}

# OpenReco brand logo (faceted-hexagon mark) — inlined so the report is self-contained/offline.
_LOGO_SVG = (
    '<svg viewBox="0 0 100 100" width="40" height="40" xmlns="http://www.w3.org/2000/svg">'
    '<polygon points="15.4,32 50,12 50,52" fill="#7FB0FF"/>'
    '<polygon points="50,12 84.6,32 50,52" fill="#3B82F6"/>'
    '<polygon points="84.6,32 84.6,72 50,52" fill="#1D4ED8"/>'
    '<polygon points="50,12 84.6,32 84.6,72 50,92 15.4,72 15.4,32" fill="none" stroke="#1D4ED8" '
    'stroke-width="3" stroke-linejoin="round"/>'
    '<circle cx="50" cy="52" r="3.4" fill="#3B82F6"/></svg>'
)


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
    ok = d["ok"]
    badge = (f"<span class='badge {'ok' if ok else 'fail'}'>{'OK' if ok else 'FAILED'}</span>")
    total_time = sum(s["seconds"] for s in d["stages"])
    body = f"""<!doctype html>
<html lang=en><head><meta charset=utf-8>
<title>OpenReco report — {_e(d['project'])}</title>
<link rel=preconnect href="https://fonts.gstatic.com" crossorigin>
<link rel=stylesheet href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=JetBrains+Mono:wght@400;500&display=swap">
<style>
 :root {{ --ink:#142233; --muted:#5b6b82; --line:#e2e9f4; --accent:#1d4ed8; --paper:#f5f7fb; }}
 * {{ box-sizing:border-box; }}
 body {{ font: 14px/1.6 "Space Grotesk","Segoe UI",system-ui,sans-serif; margin:0; color:var(--ink); background:var(--paper); }}
 .wrap {{ max-width: 1000px; margin: 0 auto; padding: 28px 30px 64px; }}
 .brandbar {{ display:flex; align-items:center; gap:12px; border-bottom:2px solid var(--accent); padding-bottom:14px; }}
 .brandbar .ttl {{ font-weight:700; font-size:22px; letter-spacing:.3px; }} .brandbar .ttl .lr {{ color:var(--accent); }}
 .badge {{ margin-left:auto; font-weight:700; padding:3px 14px; border-radius:20px; font-size:13px; }}
 .badge.ok {{ color:#16a34a; background:#16a34a1c; }} .badge.fail {{ color:#dc2626; background:#dc26261c; }}
 h1 {{ font-size:19px; margin:16px 0 .1rem; }}
 h2 {{ font-size:12px; text-transform:uppercase; letter-spacing:.05em; color:var(--muted); margin-top:30px;
       border-bottom:1px solid var(--line); padding-bottom:6px; }}
 h3 {{ font-size:.95rem; margin:.8rem 0 .3rem; }}
 table {{ border-collapse: collapse; width: 100%; margin-top: .6rem; }}
 th, td {{ text-align:left; padding:.5rem .7rem; border-bottom:1px solid var(--line); vertical-align:top; }}
 th {{ color:var(--muted); font-weight:600; text-transform:uppercase; font-size:11px; letter-spacing:.03em; }}
 code {{ font-family:"JetBrains Mono",ui-monospace,monospace; background:#eef2fb; border:1px solid var(--line);
         padding:0 5px; border-radius:4px; font-size:12.5px; }}
 .muted {{ color:#9aa7ba; }} .hint {{ color:var(--muted); }} .meta {{ color:var(--muted); }}
 .cards {{ display:flex; flex-wrap:wrap; gap:10px; margin-top:14px; }}
 .card {{ background:#fff; border:1px solid var(--line); border-radius:10px; padding:11px 14px; min-width:150px;
          box-shadow:0 1px 3px rgba(20,45,90,.05); }}
 .cval {{ font-size:21px; font-weight:700; }} .clabel {{ color:var(--muted); font-size:11px;
          text-transform:uppercase; letter-spacing:.03em; }}
 ul {{ margin:.2rem 0; }}
</style></head><body><div class=wrap>
<div class=brandbar>{_LOGO_SVG}<span class=ttl>Open<span class=lr>Reco</span></span>{badge}</div>
<h1>{_e(d['project'])}</h1>
<p class=meta>started {_e(d['started'])} · {total_time:.1f}s total · {len(d['stages'])} stages</p>
{_summary_cards(d['stages'])}
{_issues_section(d['stages'])}
<h2>Stages</h2>
<table><thead><tr><th>stage</th><th>type</th><th>status</th><th>time</th><th>metrics</th></tr></thead>
<tbody>{_stage_rows(d['stages'])}</tbody></table>
{_repro_block(d)}
</div></body></html>"""
    path.write_text(body, encoding="utf-8")
