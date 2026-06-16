"""Processing report — a self-contained HTML summary for auditability and reproducibility.

Surfaces the headline quality numbers (registration, reprojection error, GPS residuals, CRS,
point/mesh/raster sizes), QA issues grouped by severity, a per-stage table, and a
reproducibility block (tool versions + the exact resolved parameters and content-address keys
for every stage). Everything here is derived from the run record — no recomputation.
"""

from __future__ import annotations

import html
import json
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


def _artifact(stages, stype, key):
    for s in stages:
        if s.get("type") == stype:
            p = s.get("artifacts", {}).get(key)
            if p and Path(p).is_file():
                return p
    return None


def _hero_img(stages) -> str:
    """Embed the orthomosaic (or DEM) as a base64 figure — keeps the report self-contained."""
    import base64
    path = _artifact(stages, "ortho", "ortho") or _artifact(stages, "dsm", "dsm")
    if not path:
        return ""
    try:
        from openreco.io.raster import raster_to_png
        b64 = base64.b64encode(raster_to_png(Path(path), max_dim=1400)).decode()
    except Exception:  # noqa: BLE001
        return ""
    return (f"<div class=hero><img src='data:image/png;base64,{b64}' alt='preview'>"
            "<div class=cap>Orthomosaic / elevation preview.</div></div>")


def _cameras_section(stages) -> str:
    p = _artifact(stages, "ingest", "images")
    if not p:
        return ""
    try:
        data = json.loads(Path(p).read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return ""
    groups: dict = {}
    for im in data.get("images", []):
        key = ((im.get("model") or im.get("make") or "Unknown camera").strip(),
               f"{im.get('width', '?')} × {im.get('height', '?')}", im.get("focal_mm"))
        groups[key] = groups.get(key, 0) + 1
    if not groups:
        return ""
    rows = "".join(f"<tr><td>{_e(m)}</td><td>{_e(r)}</td><td>{_e(f'{fl} mm' if fl else '—')}</td>"
                   f"<td>{n}</td></tr>" for (m, r, fl), n in groups.items())
    return ("<h2>Survey data</h2><table><thead><tr><th>camera</th><th>resolution</th>"
            f"<th>focal length</th><th>images</th></tr></thead><tbody>{rows}</tbody></table>")


def _system_section(d) -> str:
    plat = d.get("platform", {})
    rows = [("software", f"OpenReco {d.get('openreco_version', '')}"),
            ("python", plat.get("python", "")),
            ("OS", f"{plat.get('system', '')} {plat.get('machine', '')}".strip())]
    try:
        from openreco import compute
        c = compute.describe()
        rows.append(("GPU", c.get("gpu_name") or ("NVIDIA GPU" if c.get("nvidia_gpu") else "none")))
        rows.append(("CPU cores", str(c.get("cpu_count", ""))))
        rows.append(("dense backend", str(c.get("auto_dense_backend", ""))))
    except Exception:  # noqa: BLE001
        pass
    body = "".join(f"<tr><td>{_e(k)}</td><td>{_e(v)}</td></tr>" for k, v in rows)
    return f"<h2>System</h2><table><tbody>{body}</tbody></table>"


def _stages_table(stages) -> str:
    return ("<table><thead><tr><th>stage</th><th>type</th><th>status</th><th>time</th><th>metrics</th>"
            f"</tr></thead><tbody>{_stage_rows(stages)}</tbody></table>")


_TEMPLATE = Path(__file__).with_name("report_template.html")


def report_html(d: dict) -> str:
    """Render the run record `d` into the HTML report by filling {{tokens}} in report_template.html.
    Edit that template to customize the report's look — this only substitutes values."""
    ok = d["ok"]
    total_time = sum(s["seconds"] for s in d["stages"])
    tokens = {
        "project": _e(d["project"]),
        "logo": _LOGO_SVG,
        "badge": f"<span class='badge {'ok' if ok else 'fail'}'>{'OK' if ok else 'FAILED'}</span>",
        "meta": f"started {_e(d['started'])} · {total_time:.1f}s total · {len(d['stages'])} stages",
        "hero": _hero_img(d["stages"]),
        "summary_cards": _summary_cards(d["stages"]),
        "cameras": _cameras_section(d["stages"]),
        "issues": _issues_section(d["stages"]),
        "stages": _stages_table(d["stages"]),
        "repro": _repro_block(d),
        "system": _system_section(d),
    }
    out = _TEMPLATE.read_text(encoding="utf-8")
    for k, v in tokens.items():
        out = out.replace("{{" + k + "}}", v)
    return out


def write_report(outcome: "RunOutcome", path: Path) -> None:
    path.write_text(report_html(outcome.to_dict()), encoding="utf-8")
