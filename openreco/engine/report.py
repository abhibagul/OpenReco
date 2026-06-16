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
    '<svg viewBox="0 0 100 100" width="46" height="46" xmlns="http://www.w3.org/2000/svg">'
    '<polygon points="15.4,32 50,12 50,52" fill="#7FB0FF"/>'
    '<polygon points="50,12 84.6,32 50,52" fill="#3B82F6"/>'
    '<polygon points="84.6,32 84.6,72 50,52" fill="#1D4ED8"/>'
    '<line x1="50" y1="52" x2="84.6" y2="72" stroke="#9DB8E6" stroke-width="2" stroke-linecap="round"/>'
    '<line x1="50" y1="52" x2="50" y2="92" stroke="#9DB8E6" stroke-width="2" stroke-linecap="round"/>'
    '<line x1="50" y1="52" x2="15.4" y2="72" stroke="#9DB8E6" stroke-width="2" stroke-linecap="round"/>'
    '<line x1="50" y1="52" x2="15.4" y2="32" stroke="#9DB8E6" stroke-width="2" stroke-linecap="round"/>'
    '<polygon points="50,12 84.6,32 84.6,72 50,92 15.4,72 15.4,32" fill="none" stroke="#1D4ED8" '
    'stroke-width="3" stroke-linejoin="round"/>'
    '<circle cx="50" cy="12" r="3.4" fill="#1D4ED8"/><circle cx="84.6" cy="32" r="3.4" fill="#1D4ED8"/>'
    '<circle cx="84.6" cy="72" r="3.4" fill="#1D4ED8"/><circle cx="50" cy="92" r="3.4" fill="#1D4ED8"/>'
    '<circle cx="15.4" cy="72" r="3.4" fill="#1D4ED8"/><circle cx="15.4" cy="32" r="3.4" fill="#1D4ED8"/>'
    '<circle cx="50" cy="52" r="3.4" fill="#3B82F6"/></svg>'
)


def _e(x: Any) -> str:
    return html.escape(str(x))


def _by_type(stages):
    out: dict[str, dict] = {}
    for s in stages:
        out.setdefault(s["type"], s)
    return out


def _stage_id(stages, stype, default=None):
    for s in stages:
        if s.get("type") == stype:
            return s["id"]
    return default or stype


def _img_tag(path, max_dim, colormap=None) -> str:
    """A base64 <img> of a GeoTIFF artifact (self-contained), or '' if unavailable."""
    import base64
    if not path:
        return ""
    try:
        from openreco.io.raster import raster_to_png
        b64 = base64.b64encode(raster_to_png(Path(path), max_dim=max_dim, colormap=colormap)).decode()
        return f"<img src='data:image/png;base64,{b64}' alt='preview'>"
    except Exception:  # noqa: BLE001
        return ""


def _cards(stages) -> str:
    by = _by_type(stages)

    def m(t, k):
        return by.get(t, {}).get("metrics", {}).get(k)

    cells: list[tuple[str, str, bool]] = []
    if (reg := m("sfm", "reg_images")) is not None:
        cells.append((f"{reg} <span class=u>/ {m('sfm', 'input_images')}</span>", "Images registered", False))
    if (err := m("sfm", "mean_reproj_error")) is not None:
        cells.append((f"{err} <span class=u>px</span>", "Mean reproj error", False))
    if (pts := m("sfm", "points3D")) is not None:
        cells.append((f"{pts:,}", "Sparse points", False))
    if (npn := m("mvs", "num_points")) is not None:
        cells.append((f"{npn:,}", "Dense points", False))
    if (crs := m("georef", "crs")) is not None:
        cells.append((_e(crs), "Coordinate system", False))
    if (rms := m("georef", "rms_residual_m")) is not None:
        cells.append((f"{rms} <span class=u>m</span>", "GCP / GPS control RMS", True))
    if (faces := m("mesh", "faces")) is not None:
        cells.append((f"{faces:,}", "Mesh faces", False))
    if (cov := m("texture", "atlas_coverage")) is not None:
        cells.append((f"{int(cov * 100)}<span class=u>%</span>", "Texture coverage", False))
    if (w := m("dsm", "width")) is not None:
        cells.append((f"{w}×{m('dsm', 'height')} <span class=u>px</span>", "DSM size", False))
    return "".join(f"<div class='metric{' hl' if hl else ''}'><div class=v>{v}</div>"
                   f"<div class='l lab'>{_e(lbl)}</div></div>" for v, lbl, hl in cells)


def _survey(stages) -> str:
    p = _artifact(stages, "ingest", "images")
    if not p:
        return "<tr><td colspan=5 class=muted>No camera metadata.</td></tr>"
    try:
        data = json.loads(Path(p).read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return "<tr><td colspan=5 class=muted>No camera metadata.</td></tr>"
    groups: dict = {}
    for im in data.get("images", []):
        key = ((im.get("model") or "Unknown").strip(), f"{im.get('width', '?')} × {im.get('height', '?')}",
               im.get("focal_mm"), (im.get("make") or "").strip())
        groups[key] = groups.get(key, 0) + 1
    return "".join(f"<tr><td><b>{_e(model)}</b></td><td class=mono>{_e(res)}</td>"
                   f"<td class=mono>{_e(f'{fl} mm' if fl else '—')}</td><td class=mono>{n}</td>"
                   f"<td class=mono>{_e(make or '—')}</td></tr>"
                   for (model, res, fl, make), n in groups.items())


_SEV_PILL = {"warning": ("warn", "WARN"), "info": ("info", "INFO"), "error": ("failed", "ERR")}


def _qa(stages):
    rows, n = [], {"warning": 0, "info": 0, "error": 0}
    for s in stages:
        for i in s["issues"]:
            sev = i["severity"]
            n[sev] = n.get(sev, 0) + 1
            cls, lbl = _SEV_PILL.get(sev, ("info", sev.upper()))
            msg = _e(i["message"]) + (f" — {_e(i['hint'])}" if i.get("hint") else "")
            rows.append(f"<div class=qarow><span class='pill {cls}'>{lbl}</span><span class=msg>{msg}</span></div>")
    parts = []
    if n.get("error"):
        parts.append(f"{n['error']} errors")
    if n.get("warning"):
        parts.append(f"{n['warning']} warnings")
    if n.get("info"):
        parts.append(f"{n['info']} notes")
    suffix = ("· " + " · ".join(parts)) if parts else ""
    return suffix, ("".join(rows) if rows else "<p class=muted>No issues flagged.</p>")


def _measure_result(mz) -> str:
    r, t = mz.get("result") or {}, mz.get("type")
    if t == "dist":
        return f"{r.get('length_m', '?')} <span class=unit>m</span>"
    if t == "area":
        return f"{r.get('area_m2', '?')} <span class=unit>m²</span>"
    if t == "vol":
        return (f"net {r.get('net_m3', '?')} <span class=unit>m³</span> · "
                f"cut {r.get('cut_m3', '?')} / fill {r.get('fill_m3', '?')}")
    if t == "prof":
        return (f"{r.get('length_m', '?')} <span class=unit>m</span> · "
                f"Δ {r.get('relief_m', '?')} <span class=unit>m</span> · {r.get('slope_pct', '?')}%")
    return "annotation" if t == "note" else "—"


def _measures(measurements) -> str:
    if not measurements:
        return "<tr><td colspan=3 class=muted>No measurements.</td></tr>"
    return "".join(f"<tr><td><b>{_e(mz.get('name', '?'))}</b></td>"
                   f"<td><span class='pill type'>{_e(mz.get('type', '?'))}</span></td>"
                   f"<td class=mono>{_measure_result(mz)}</td></tr>" for mz in measurements)


def _stage_rows_p(stages):
    nexec = sum(1 for s in stages if s["status"] == "executed")
    ncached = sum(1 for s in stages if s["status"] == "cached")
    suffix = f"· {len(stages)} · {nexec} executed · {ncached} cached"
    rows = []
    for s in stages:
        cls = s["status"] if s["status"] in ("executed", "cached", "failed") else "cached"
        metrics = ", ".join(f"{_e(k)}={_e(v)}" for k, v in s["metrics"].items())
        rows.append(f"<tr><td><b>{_e(s['id'])}</b></td><td><span class='pill type'>{_e(s['type'])}</span></td>"
                    f"<td><span class='pill {cls}'>{_e(s['status'])}</span></td>"
                    f"<td class=mono>{s['seconds']:.2f}s</td>"
                    f"<td class=mono>{metrics or '—'}</td></tr>")
    return suffix, "".join(rows)


def _repro_rows(stages) -> str:
    rows = []
    for s in stages:
        params = ", ".join(f"{_e(k)}={_e(v)}" for k, v in s.get("params", {}).items())
        rows.append(f"<tr><td><b>{_e(s['id'])}</b></td><td class=key>{_e((s.get('key') or '')[:18])}</td>"
                    f"<td class=mono>{params or '—'}</td></tr>")
    return "".join(rows)


def _artifact(stages, stype, key):
    for s in stages:
        if s.get("type") == stype:
            p = s.get("artifacts", {}).get(key)
            if p and Path(p).is_file():
                return p
    return None


_TEMPLATE = Path(__file__).with_name("report_template.html")


def report_html(d: dict, measurements=None) -> str:
    """Fill the multi-page report_template.html tokens from a run record (+ optional measurements).
    Edit report_template.html to restyle the report; this only substitutes values."""
    ok = d["ok"]
    stages = d["stages"]
    total = sum(s["seconds"] for s in stages)
    by = _by_type(stages)
    crs = by.get("georef", {}).get("metrics", {}).get("crs") or "LOCAL"
    chunks = {s.get("chunk") for s in stages if s.get("chunk")}
    ortho_p = _artifact(stages, "ortho", "ortho")
    dsm_p = _artifact(stages, "dsm", "dsm")
    dsm_id = _stage_id(stages, "dsm", "dsm")
    dm = by.get("dsm", {}).get("metrics", {})
    dw, dh = dm.get("width", "?"), dm.get("height", "?")
    qa_suffix, qa_rows = _qa(stages)
    st_suffix, st_rows = _stage_rows_p(stages)
    hero_id = _stage_id(stages, "ortho") if ortho_p else dsm_id
    hero_cap = (f"ORTHOMOSAIC · {hero_id} · point-cloud based" if ortho_p
                else f"DIGITAL ELEVATION MODEL · {hero_id}")
    tokens = {
        "project": _e(d["project"]), "project_uc": _e(str(d["project"]).upper()),
        "logo": _LOGO_SVG,
        "date": _e((d.get("started") or "")[:10]), "crs": _e(crs),
        "runtime": f"{total:.1f} s", "stages_n": str(len(stages)),
        "status": "OK" if ok else "FAILED", "status_cls": "ok" if ok else "fail",
        "started": _e(d.get("started") or ""),
        "hero": _img_tag(ortho_p or dsm_p, 1600), "hero_cap": _e(hero_cap), "hero_epsg": _e(crs),
        "sum_meta": (f"started {_e(d.get('started') or '')}<br>{total:.1f} s total · "
                     f"{len(stages)} stages · {len(chunks) or 1} chunk(s)"),
        "cards": _cards(stages), "survey": _survey(stages),
        "dem": _img_tag(dsm_p, 1400, colormap="turbo"),
        "dem_cap": _e(f"{dsm_id} · {crs if crs != 'LOCAL' else 'local frame'} · {dw}×{dh} px"),
        "qa_suffix": qa_suffix, "qa_rows": qa_rows,
        "measures": _measures(measurements),
        "stages_suffix": st_suffix, "stage_rows": st_rows,
        "repro_rows": _repro_rows(stages),
    }
    out = _TEMPLATE.read_text(encoding="utf-8")
    for k, v in tokens.items():
        out = out.replace("{{" + k + "}}", v)
    return out


def write_report(outcome: "RunOutcome", path: Path, measurements=None) -> None:
    path.write_text(report_html(outcome.to_dict(), measurements), encoding="utf-8")
