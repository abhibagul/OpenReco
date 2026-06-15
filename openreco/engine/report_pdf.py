"""Processing report as a PDF — rendered with Pillow (no extra dependencies).

Builds a paginated A4 PDF from a run record (the same dict written to runs/latest.json): headline
summary cards, QA issues, a per-stage table, GCP accuracy (control/check + per-GCP residuals when a
georef.json is available), and a reproducibility block. Returns PDF bytes.
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

OK = (38, 166, 91)
ERR = (220, 38, 38)
WARN = (217, 119, 6)
BLUE = (37, 99, 235)
INK = (17, 17, 17)
GREY = (110, 116, 128)
LINE = (224, 226, 232)


def _font(size: int, bold: bool = False):
    from PIL import ImageFont
    names = (["segoeuib.ttf", "arialbd.ttf", "DejaVuSans-Bold.ttf"] if bold
             else ["segoeui.ttf", "arial.ttf", "DejaVuSans.ttf"])
    for d in ("C:/Windows/Fonts/", "/usr/share/fonts/truetype/dejavu/", "/Library/Fonts/", ""):
        for n in names:
            try:
                return ImageFont.truetype(d + n, size)
            except OSError:
                continue
    try:
        return ImageFont.load_default(size)        # Pillow >= 10: scalable default
    except TypeError:
        return ImageFont.load_default()


class _Doc:
    W, H, M = 1240, 1754, 72

    def __init__(self):
        from PIL import Image, ImageDraw
        self._Image, self._Draw = Image, ImageDraw
        self.pages = []
        self._new()
        self.f_title, self.f_h, self.f_b = _font(34, True), _font(21, True), _font(16)
        self.f_bb, self.f_s, self.f_card = _font(16, True), _font(13), _font(24, True)

    def _new(self):
        img = self._Image.new("RGB", (self.W, self.H), "white")
        self.pages.append(img)
        self.d = self._Draw.Draw(img)
        self.y = self.M

    def need(self, h):
        if self.y + h > self.H - self.M:
            self._new()

    def text(self, s, font, color=INK, dx=0, lh=None):
        lh = lh or (font.size + 8 if hasattr(font, "size") else 22)
        self.need(lh)
        self.d.text((self.M + dx, self.y), str(s), font=font, fill=color)
        self.y += lh

    def gap(self, h=10):
        self.y += h

    def rule(self):
        self.need(12)
        self.d.line([(self.M, self.y), (self.W - self.M, self.y)], fill=LINE, width=1)
        self.y += 12


def _cards(doc: _Doc, cards: list[tuple[str, str]]):
    if not cards:
        return
    cols, cw, ch, gap = 3, 0, 78, 14
    cw = (doc.W - 2 * doc.M - (cols - 1) * gap) / cols
    for r in range(0, len(cards), cols):
        doc.need(ch + gap)
        row = cards[r:r + cols]
        for i, (label, val) in enumerate(row):
            x = doc.M + i * (cw + gap)
            doc.d.rounded_rectangle([x, doc.y, x + cw, doc.y + ch], radius=10, fill=(248, 250, 252), outline=LINE)
            doc.d.text((x + 14, doc.y + 12), str(val), font=doc.f_card, fill=INK)
            doc.d.text((x + 14, doc.y + 48), str(label), font=doc.f_s, fill=GREY)
        doc.y += ch + gap


def _by_type(stages):
    out = {}
    for s in stages:
        out.setdefault(s["type"], s)
    return out


def _summary(stages) -> list[tuple[str, str]]:
    t = _by_type(stages)
    def m(st, k):
        return t.get(st, {}).get("metrics", {}).get(k)
    cards = []
    if (v := m("sfm", "reg_images")) is not None:
        cards.append(("images registered", f"{v} / {m('sfm', 'input_images')}"))
    if (v := m("sfm", "mean_reproj_error")) is not None:
        cards.append(("mean reproj error", f"{v} px"))
    if (v := m("sfm", "points3D")) is not None:
        cards.append(("sparse points", f"{v:,}"))
    if (v := m("georef", "crs")) is not None:
        cards.append(("coordinate system", str(v)))
    if (v := m("georef", "rms_residual_m")) is not None:
        cards.append(("GCP/GPS control RMS", f"{v} m"))
    if (v := m("georef", "check_rms_m")) is not None:
        cards.append(("GCP check RMS", f"{v} m"))
    if (v := m("mvs", "num_points")) is not None:
        cards.append(("dense points", f"{v:,}"))
    if (v := m("mesh", "faces")) is not None:
        cards.append(("mesh faces", f"{v:,}"))
    if (v := m("texture", "atlas_coverage")) is not None:
        cards.append(("texture coverage", f"{int(v * 100)}%"))
    if (v := m("dsm", "width")) is not None:
        cards.append(("DSM size", f"{v}×{m('dsm', 'height')} px"))
    return cards


def _table(doc: _Doc, headers, rows, widths):
    xs, x = [], doc.M
    for w in widths:
        xs.append(x)
        x += w * (doc.W - 2 * doc.M)
    doc.need(30)
    for i, hd in enumerate(headers):
        doc.d.text((xs[i], doc.y), hd, font=doc.f_bb, fill=GREY)
    doc.y += 24
    doc.d.line([(doc.M, doc.y), (doc.W - doc.M, doc.y)], fill=LINE, width=1)
    doc.y += 8
    for row in rows:
        doc.need(24)
        for i, cell in enumerate(row):
            col = cell[1] if isinstance(cell, tuple) else INK
            txt = cell[0] if isinstance(cell, tuple) else cell
            maxw = (widths[i] * (doc.W - 2 * doc.M)) - 12
            txt = _truncate(doc, str(txt), doc.f_s, maxw)
            doc.d.text((xs[i], doc.y), txt, font=doc.f_s, fill=col)
        doc.y += 22


def _truncate(doc, s, font, maxw):
    if doc.d.textlength(s, font=font) <= maxw:
        return s
    while s and doc.d.textlength(s + "…", font=font) > maxw:
        s = s[:-1]
    return s + "…"


def _gcp_section(doc: _Doc, stages):
    g = next((s for s in stages if s["type"] == "georef" and s.get("metrics", {}).get("method") == "gcp"), None)
    if not g:
        return
    info = {}
    gp = g.get("artifacts", {}).get("georef")
    if gp and Path(gp).is_file():
        try:
            info = json.loads(Path(gp).read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            info = {}
    doc.gap(8)
    doc.text("GCP accuracy", doc.f_h)
    met = g["metrics"]
    doc.text(f"control RMSE {met.get('control_rms_m', met.get('rms_residual_m'))} m "
             f"({met.get('num_control', '?')} pts)   ·   check RMSE {met.get('check_rms_m', '—')} m "
             f"({met.get('num_check', 0)} pts)", doc.f_b, GREY)
    if info.get("gcps"):
        doc.gap(4)
        rows = [[g["name"], g["type"], (f"{g['error_m']}", ERR if g["error_m"] > 0.5 else INK),
                 f"{g['dx']}", f"{g['dy']}", f"{g['dz']}"] for g in info["gcps"]]
        _table(doc, ["GCP", "type", "err (m)", "dx", "dy", "dz"], rows, [0.28, 0.16, 0.16, 0.13, 0.13, 0.14])


def _measure_value(m: dict) -> str:
    r = m.get("result") or {}
    t = m.get("type")
    if t == "dist":
        return f"{r.get('length_m', '?')} m"
    if t == "area":
        per = r.get("perimeter_m")
        return f"{r.get('area_m2', '?')} m²" + (f"  (perim {per} m)" if per is not None else "")
    if t == "vol":
        return f"net {r.get('net_m3', '?')} m³  ·  cut {r.get('cut_m3', '?')} / fill {r.get('fill_m3', '?')}"
    if t == "prof":
        return f"{r.get('length_m', '?')} m  ·  Δ {r.get('relief_m', '?')} m  ·  {r.get('slope_pct', '?')}%"
    if t == "note":
        return "annotation"
    return "—"


_MEASURE_LABEL = {"dist": "distance", "area": "area", "vol": "volume", "prof": "profile",
                  "note": "note"}


def _measurements_section(doc: _Doc, measurements):
    if not measurements:
        return
    doc.gap(12)
    doc.text("Measurements", doc.f_h)
    rows = [[m.get("name", "?"), _MEASURE_LABEL.get(m.get("type"), m.get("type", "?")),
             _measure_value(m)] for m in measurements]
    _table(doc, ["name", "type", "value"], rows, [0.26, 0.16, 0.58])


def write_report_pdf(data: dict[str, Any] | None, measurements=None) -> bytes:
    """Render a run record (latest.json dict) to PDF bytes. None -> a 'no report yet' page.
    `measurements` (the persisted list) are appended as their own section when present."""
    doc = _Doc()
    if not data:
        doc.text("OpenReco — Processing report", doc.f_title)
        doc.gap(6)
        doc.text("No processing report yet. Run the pipeline (Run), then open it again.", doc.f_b, GREY)
        _measurements_section(doc, measurements)
        return _save(doc)

    badge = ("OK", OK) if data.get("ok") else ("FAILED", ERR)
    doc.text(f"OpenReco — {data.get('project', 'project')}", doc.f_title)
    doc.d.text((doc.W - doc.M - doc.d.textlength(badge[0], font=doc.f_h), doc.M + 6), badge[0],
               font=doc.f_h, fill=badge[1])
    total = sum(s.get("seconds", 0) for s in data["stages"])
    plat = data.get("platform", {})
    doc.text(f"started {data.get('started', '')}   ·   {total:.1f}s total   ·   {len(data['stages'])} stages",
             doc.f_s, GREY)
    doc.text(f"openreco {data.get('openreco_version', '')}   ·   python {plat.get('python', '')} "
             f"· {plat.get('system', '')}/{plat.get('machine', '')}", doc.f_s, GREY)
    doc.gap(10)
    _cards(doc, _summary(data["stages"]))

    # QA issues
    buckets = {"error": [], "warning": [], "info": []}
    for s in data["stages"]:
        for it in s.get("issues", []):
            buckets.setdefault(it["severity"], []).append((s["id"], it["message"], it.get("hint")))
    if any(buckets.values()):
        doc.gap(10)
        doc.text("QA issues", doc.f_h)
        for sev, col in (("error", ERR), ("warning", WARN), ("info", BLUE)):
            for sid, msg, hint in buckets.get(sev, []):
                doc.text(f"[{sev}] {sid}: {msg}" + (f"  — {hint}" if hint else ""), doc.f_s, col)

    # GCP accuracy
    _gcp_section(doc, data["stages"])

    # measurements (volumes / areas / distances / profiles)
    _measurements_section(doc, measurements)

    # stages
    doc.gap(12)
    doc.text("Stages", doc.f_h)
    rows = []
    for s in data["stages"]:
        sc = {"executed": BLUE, "cached": OK, "failed": ERR, "skipped": WARN, "cancelled": GREY}.get(s["status"], INK)
        met = ", ".join(f"{k}={v}" for k, v in s.get("metrics", {}).items())
        rows.append([s["id"], s["type"], (s["status"], sc), f"{s.get('seconds', 0):.2f}s", met or "—"])
    _table(doc, ["stage", "type", "status", "time", "metrics"], rows, [0.18, 0.16, 0.12, 0.1, 0.44])

    # reproducibility
    doc.gap(12)
    doc.text("Reproducibility", doc.f_h)
    rows = [[s["id"], (s.get("key", "") or "")[:16], ", ".join(f"{k}={v}" for k, v in s.get("params", {}).items()) or "—"]
            for s in data["stages"]]
    _table(doc, ["stage", "cache key", "resolved parameters"], rows, [0.18, 0.2, 0.62])
    return _save(doc)


def _save(doc: _Doc) -> bytes:
    buf = io.BytesIO()
    doc.pages[0].save(buf, "PDF", save_all=True, append_images=doc.pages[1:])
    return buf.getvalue()
