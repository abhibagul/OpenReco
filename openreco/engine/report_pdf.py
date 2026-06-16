"""Processing report as a PDF — rendered with Pillow (no extra dependencies).

Mirrors the multi-page HTML report (report_template.html): a dark cover page, then light pages for
Project summary, Survey data + colorized DEM, Quality assurance + Measurements, Stages, and
Reproducibility — with the Blueprint palette, metric cards, and status/type pills. Returns PDF bytes.
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

# OpenReco "Blueprint" brand palette (RGB)
OKG = (22, 163, 74)
ERRC = (220, 38, 38)
WARNC = (138, 109, 31)
BLUE = (29, 78, 216)        # cobalt — primary accent
AZURE = (59, 130, 246)
SKY = (127, 176, 255)
INK = (20, 34, 51)
NAVY = (11, 26, 43)         # cover / dark surfaces
MUTED = (91, 107, 130)
SOFT = (159, 178, 201)
LINE = (226, 233, 244)
LINE2 = (238, 242, 251)
PAPER = (245, 247, 251)
WHITE = (255, 255, 255)
HL = (234, 240, 254)        # highlighted card
PILL = (238, 242, 247)
INFOBG = (230, 238, 251)
WARNBG = (246, 234, 210)
ERRBG = (253, 231, 234)
_TURBO = [(0, (48, 18, 59)), (0.25, (65, 105, 225)), (0.5, (27, 207, 212)),
          (0.75, (250, 186, 57)), (1, (165, 30, 20))]


def _font(size: int, bold: bool = False, mono: bool = False):
    from PIL import ImageFont
    if mono:
        names = ["consola.ttf", "DejaVuSansMono.ttf", "cour.ttf"]
    else:
        names = (["segoeuib.ttf", "arialbd.ttf", "DejaVuSans-Bold.ttf"] if bold
                 else ["segoeui.ttf", "arial.ttf", "DejaVuSans.ttf"])
    for d in ("C:/Windows/Fonts/", "/usr/share/fonts/truetype/dejavu/", "/Library/Fonts/", ""):
        for n in names:
            try:
                return ImageFont.truetype(d + n, size)
            except OSError:
                continue
    try:
        return ImageFont.load_default(size)
    except TypeError:
        return ImageFont.load_default()


def _logo(d, ox: float, oy: float, s: float) -> None:
    """Draw the faceted-hexagon brand mark (s = size px)."""
    k = s / 100.0

    def P(pts):
        return [(ox + px * k, oy + py * k) for px, py in pts]

    d.polygon(P([(15.4, 32), (50, 12), (50, 52)]), fill=SKY)
    d.polygon(P([(50, 12), (84.6, 32), (50, 52)]), fill=AZURE)
    d.polygon(P([(84.6, 32), (84.6, 72), (50, 52)]), fill=BLUE)
    d.polygon(P([(50, 12), (84.6, 32), (84.6, 72), (50, 92), (15.4, 72), (15.4, 32)]),
              outline=BLUE, width=max(1, int(round(3 * k))))
    r, cx, cy = 3.4 * k, ox + 50 * k, oy + 52 * k
    d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=AZURE)


class _Doc:
    W, H, M = 1240, 1754, 84

    def __init__(self):
        from PIL import Image, ImageDraw
        self._Image, self._Draw = Image, ImageDraw
        self.pages = []
        self.f_xl = _font(52, True)
        self.f_h1 = _font(30, True)
        self.f_h = _font(19, True)
        self.f_b = _font(15)
        self.f_s = _font(13)
        self.f_card = _font(26, True)
        self.lab = _font(11, mono=True)
        self.mono = _font(12, mono=True)
        self.mono_b = _font(13, True, mono=True)
        self._new()

    def _new(self):
        img = self._Image.new("RGB", (self.W, self.H), "white")
        self.pages.append(img)
        self.d = self._Draw.Draw(img)
        self.y = self.M

    def fill_page(self, color):
        self.d.rectangle([0, 0, self.W, self.H], fill=color)

    def grid(self, color, step=38):
        for x in range(0, self.W, step):
            self.d.line([(x, 0), (x, self.H)], fill=color, width=1)
        for y in range(0, self.H, step):
            self.d.line([(0, y), (self.W, y)], fill=color, width=1)

    def text(self, s, font, color=INK, x=None, y=None):
        self.d.text((self.M if x is None else x, self.y if y is None else y), str(s), font=font, fill=color)

    def rtext(self, s, font, color, ry, rx=None):       # right-aligned at rx (default right margin)
        rx = self.W - self.M if rx is None else rx
        self.d.text((rx - self.d.textlength(str(s), font=font), ry), str(s), font=font, fill=color)

    def pill(self, x, y, text, bg, fg, font=None):
        font = font or self.lab
        tw = self.d.textlength(str(text), font=font)
        self.d.rounded_rectangle([x, y, x + tw + 16, y + 22], radius=6, fill=bg)
        self.d.text((x + 8, y + 4), str(text), font=font, fill=fg)
        return tw + 16

    def heading(self, title, suffix=""):
        self.y += 26
        self.d.ellipse([self.M, self.y + 4, self.M + 10, self.y + 14], fill=BLUE)
        self.d.text((self.M + 20, self.y), title, font=self.f_h, fill=INK)
        if suffix:
            tw = self.d.textlength(title, font=self.f_h)
            self.d.text((self.M + 32 + tw, self.y + 5), suffix, font=self.lab, fill=SOFT)
        self.y += 34

    def image(self, pil, max_h, caption=None, legend=False):
        cw = self.W - 2 * self.M
        w, h = pil.size
        scale = min(cw / w, max_h / h)
        nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
        try:
            rim = pil.resize((nw, nh), self._Image.LANCZOS)
        except Exception:  # noqa: BLE001
            rim = pil.resize((nw, nh))
        x = int(self.M + (cw - nw) / 2)
        self.pages[-1].paste(rim, (x, int(self.y)))
        if caption:
            self.d.rectangle([x, self.y + nh - 26, x + nw, self.y + nh], fill=NAVY)
            self.d.text((x + 10, self.y + nh - 22), caption, font=self.lab, fill=(207, 224, 255))
        self.y += nh + 8
        if legend:
            self._legend(self.W - self.M - 170, self.y)
            self.y += 18

    def _legend(self, x, y):
        for i in range(120):
            t = i / 119
            col = _ramp_rgb(t)
            self.d.line([(x + 32 + i, y), (x + 32 + i, y + 9)], fill=col)
        self.d.text((x, y - 1), "LOW", font=self.lab, fill=MUTED)
        self.d.text((x + 158, y - 1), "HIGH", font=self.lab, fill=MUTED)


def _ramp_rgb(t):
    ts = [s[0] for s in _TURBO]
    return tuple(int(_interp(t, ts, [s[1][c] for s in _TURBO])) for c in range(3))


def _interp(t, xs, ys):
    for i in range(1, len(xs)):
        if t <= xs[i]:
            f = (t - xs[i - 1]) / ((xs[i] - xs[i - 1]) or 1)
            return ys[i - 1] + (ys[i] - ys[i - 1]) * f
    return ys[-1]


# ---- data helpers ----------------------------------------------------------
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


def _artifact(stages, stype, key):
    for s in stages:
        if s.get("type") == stype:
            p = s.get("artifacts", {}).get(key)
            if p and Path(p).is_file():
                return p
    return None


def _open_raster(path, max_dim=1500, colormap=None):
    if not path:
        return None
    try:
        from PIL import Image

        from openreco.io.raster import raster_to_png
        return Image.open(io.BytesIO(raster_to_png(Path(path), max_dim=max_dim,
                                                    colormap=colormap))).convert("RGB")
    except Exception:  # noqa: BLE001
        return None


def _cameras(stages):
    p = _artifact(stages, "ingest", "images")
    if not p:
        return []
    try:
        data = json.loads(Path(p).read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return []
    groups: dict = {}
    for im in data.get("images", []):
        key = ((im.get("model") or "Unknown").strip(), f"{im.get('width', '?')} x {im.get('height', '?')}",
               im.get("focal_mm"), (im.get("make") or "").strip())
        groups[key] = groups.get(key, 0) + 1
    return [[model, res, (f"{fl} mm" if fl else "—"), str(n), make or "—"]
            for (model, res, fl, make), n in groups.items()]


def _measure_result(mz):
    r, t = mz.get("result") or {}, mz.get("type")
    if t == "dist":
        return f"{r.get('length_m', '?')} m"
    if t == "area":
        return f"{r.get('area_m2', '?')} m²"
    if t == "vol":
        return f"net {r.get('net_m3', '?')} m³ · cut {r.get('cut_m3', '?')} / fill {r.get('fill_m3', '?')}"
    if t == "prof":
        return f"{r.get('length_m', '?')} m · Δ {r.get('relief_m', '?')} m · {r.get('slope_pct', '?')}%"
    return "annotation" if t == "note" else "—"


def _trunc(doc, s, font, maxw):
    s = str(s)
    if doc.d.textlength(s, font=font) <= maxw:
        return s
    while s and doc.d.textlength(s + "…", font=font) > maxw:
        s = s[:-1]
    return s + "…"


def _cards(stages):
    by = _by_type(stages)

    def m(t, k):
        return by.get(t, {}).get("metrics", {}).get(k)

    cells = []
    if (v := m("sfm", "reg_images")) is not None:
        cells.append((f"{v}", f"/ {m('sfm', 'input_images')}", "Images registered", False))
    if (v := m("sfm", "mean_reproj_error")) is not None:
        cells.append((f"{v}", "px", "Mean reproj error", False))
    if (v := m("sfm", "points3D")) is not None:
        cells.append((f"{v:,}", "", "Sparse points", False))
    if (v := m("mvs", "num_points")) is not None:
        cells.append((f"{v:,}", "", "Dense points", False))
    if (v := m("georef", "crs")) is not None:
        cells.append((f"{v}", "", "Coordinate system", False))
    if (v := m("georef", "rms_residual_m")) is not None:
        cells.append((f"{v}", "m", "GCP / GPS control RMS", True))
    if (v := m("mesh", "faces")) is not None:
        cells.append((f"{v:,}", "", "Mesh faces", False))
    if (v := m("dsm", "width")) is not None:
        cells.append((f"{v}×{m('dsm', 'height')}", "px", "DSM size", False))
    return cells


# ---- page builders ---------------------------------------------------------
def _light_header(doc, ctx):
    _logo(doc.d, doc.M, doc.M - 4, 26)
    doc.d.text((doc.M + 34, doc.M), "open", font=doc.f_h, fill=INK)
    ow = doc.d.textlength("open", font=doc.f_h)
    doc.d.text((doc.M + 34 + ow, doc.M), "reco", font=doc.f_h, fill=BLUE)
    if ctx:
        doc.rtext(ctx, doc.lab, SOFT, doc.M + 6)
    doc.d.line([(doc.M, doc.M + 34), (doc.W - doc.M, doc.M + 34)], fill=LINE, width=1)
    doc.y = doc.M + 52


def _footer(doc, pageno):
    y = doc.H - 56
    doc.d.text((doc.M, y), "OPENRECO · PROCESSING REPORT", font=doc.lab, fill=SOFT)
    doc.rtext(pageno, doc.lab, SOFT, y)


def _row_cols(doc, x0, widths):
    xs, x = [], x0
    for w in widths:
        xs.append(x)
        x += w
    return xs


def _table(doc, headers, rows, widths, pill_cols=None):
    """Simple table; pill_cols maps column index -> (bg,fg) to render that cell as a pill."""
    pill_cols = pill_cols or {}
    cw = doc.W - 2 * doc.M
    xs = _row_cols(doc, doc.M, [w * cw for w in widths])
    for i, h in enumerate(headers):
        doc.d.text((xs[i], doc.y), h.upper(), font=doc.lab, fill=MUTED)
    doc.y += 22
    doc.d.line([(doc.M, doc.y), (doc.W - doc.M, doc.y)], fill=LINE, width=1)
    doc.y += 9
    for row in rows:
        for i, cell in enumerate(row):
            maxw = widths[i] * cw - 12
            if i in pill_cols and cell:
                bg, fg = pill_cols[i](cell) if callable(pill_cols[i]) else pill_cols[i]
                doc.pill(xs[i], doc.y - 2, str(cell), bg, fg)
            else:
                font = doc.mono if i else doc.f_s
                doc.d.text((xs[i], doc.y), _trunc(doc, cell, font, maxw), font=font,
                           fill=INK if i == 0 else MUTED)
        doc.y += 26
        doc.d.line([(doc.M, doc.y - 6), (doc.W - doc.M, doc.y - 6)], fill=LINE2, width=1)


_STATUS_PILL = {"executed": (INFOBG, BLUE), "cached": (PILL, MUTED), "failed": (ERRBG, ERRC),
                "skipped": (WARNBG, WARNC)}


def write_report_pdf(data: dict[str, Any] | None, measurements=None) -> bytes:
    doc = _Doc()
    if not data:
        _light_header(doc, "PROCESSING REPORT")
        doc.text("No processing report yet.", doc.f_h1)
        doc.y += 44
        doc.text("Run the pipeline (Run), then open the report again.", doc.f_b, MUTED)
        return _save(doc)

    stages = data["stages"]
    by = _by_type(stages)
    ok = data.get("ok")
    total = sum(s.get("seconds", 0) for s in stages)
    crs = by.get("georef", {}).get("metrics", {}).get("crs") or "LOCAL"
    proj = str(data.get("project", "project"))
    started = data.get("started", "")
    ortho = _open_raster(_artifact(stages, "ortho", "ortho"), 1600)
    dsm = _open_raster(_artifact(stages, "dsm", "dsm"), 1400, colormap="turbo")

    # ===== page 1 · cover =====
    doc.fill_page(NAVY)
    doc.grid((20, 38, 64), 38)
    _logo(doc.d, doc.M, doc.M - 4, 28)
    doc.d.text((doc.M + 36, doc.M), "open", font=doc.f_h, fill=WHITE)
    ow = doc.d.textlength("open", font=doc.f_h)
    doc.d.text((doc.M + 36 + ow, doc.M), "reco", font=doc.f_h, fill=SKY)
    doc.rtext("PROCESSING REPORT", doc.lab, SKY, doc.M + 4)
    doc.d.text((doc.M, doc.M + 230), "PHOTOGRAMMETRY · RECONSTRUCTION", font=doc.lab, fill=SKY)
    doc.d.text((doc.M, doc.M + 250), _trunc(doc, proj, doc.f_xl, doc.W - 2 * doc.M), font=doc.f_xl, fill=WHITE)
    mx, my = doc.M, doc.M + 330
    for k, v, c in [("DATE", started[:10], WHITE), ("CRS", crs, WHITE), ("RUNTIME", f"{total:.1f} s", WHITE),
                    ("STAGES", str(len(stages)), WHITE),
                    ("STATUS", "OK" if ok else "FAILED", (134, 239, 172) if ok else (252, 165, 165))]:
        doc.d.text((mx, my), k, font=doc.lab, fill=SKY)
        doc.d.text((mx, my + 18), str(v), font=doc.mono_b, fill=c)
        mx += max(doc.d.textlength(k, font=doc.lab), doc.d.textlength(str(v), font=doc.mono_b)) + 34
    if ortho is not None or dsm is not None:
        doc.y = my + 70
        cap = f"ORTHOMOSAIC · {_stage_id(stages, 'ortho', 'ortho')}" if ortho is not None else \
              f"DIGITAL ELEVATION MODEL · {_stage_id(stages, 'dsm', 'dsm')}"
        doc.image(ortho if ortho is not None else dsm, 760, caption=cap)
    yb = doc.H - 56
    doc.d.text((doc.M, yb), _trunc(doc, f"{proj} · started {started}", doc.lab, 560), font=doc.lab, fill=SOFT)
    doc.rtext("created using openreco", doc.lab, SOFT, yb)

    # ===== page 2 · project summary =====
    doc._new()
    _light_header(doc, None)
    doc.pill(doc.W - doc.M - 70, doc.M - 2, "OK" if ok else "FAILED",
             (224, 247, 233) if ok else ERRBG, OKG if ok else ERRC, doc.lab)
    doc.text("PROJECT SUMMARY", doc.lab, MUTED)
    doc.y += 18
    doc.text(proj, doc.f_h1, INK)
    doc.rtext(f"{total:.1f} s total · {len(stages)} stages", doc.lab, MUTED, doc.y + 4)
    doc.y += 44
    cells = _cards(stages)
    cw = doc.W - 2 * doc.M
    cellw, cellh = cw / 4, 84
    for i, (val, unit, label, hl) in enumerate(cells):
        cx = doc.M + (i % 4) * cellw
        cy = doc.y + (i // 4) * cellh
        doc.d.rectangle([cx, cy, cx + cellw, cy + cellh], fill=HL if hl else WHITE, outline=LINE)
        doc.d.text((cx + 14, cy + 16), val, font=doc.f_card, fill=INK)
        if unit:
            doc.d.text((cx + 16 + doc.d.textlength(val, font=doc.f_card), cy + 30), unit, font=doc.mono, fill=MUTED)
        doc.d.text((cx + 14, cy + 54), label.upper(), font=doc.lab, fill=MUTED)
    doc.y += cellh * ((len(cells) + 3) // 4) + 6

    cams = _cameras(stages)
    if cams:
        doc.heading("Survey data")
        _table(doc, ["camera", "resolution", "focal length", "images", "sensor"], cams,
               [0.34, 0.20, 0.18, 0.12, 0.16])
    if dsm is not None:
        doc.heading("Digital Elevation Model")
        dw, dh = by.get("dsm", {}).get("metrics", {}).get("width", "?"), by.get("dsm", {}).get("metrics", {}).get("height", "?")
        doc.image(dsm, 520, caption=f"{_stage_id(stages, 'dsm', 'dsm')} · {dw}×{dh} px", legend=True)
    _footer(doc, "02 / 05")

    # ===== page 3 · QA & measurements =====
    doc._new()
    _light_header(doc, f"{proj.upper()} · QA & MEASUREMENTS")
    nwarn = sum(1 for s in stages for i in s["issues"] if i["severity"] == "warning")
    ninfo = sum(1 for s in stages for i in s["issues"] if i["severity"] == "info")
    doc.heading("Quality assurance", f"· {nwarn} warnings · {ninfo} notes" if (nwarn or ninfo) else "")
    any_issue = False
    for s in stages:
        for it in s["issues"]:
            any_issue = True
            sev = it["severity"]
            bg, fg, lbl = ((WARNBG, WARNC, "WARN") if sev == "warning"
                           else (ERRBG, ERRC, "ERR") if sev == "error" else (INFOBG, BLUE, "INFO"))
            doc.pill(doc.M, doc.y, lbl, bg, fg)
            msg = it["message"] + (f" — {it['hint']}" if it.get("hint") else "")
            doc.d.text((doc.M + 88, doc.y + 2), _trunc(doc, msg, doc.f_s, doc.W - 2 * doc.M - 96), font=doc.f_s, fill=INK)
            doc.y += 30
    if not any_issue:
        doc.text("No issues flagged.", doc.f_s, MUTED)
        doc.y += 24
    doc.heading("Measurements")
    if measurements:
        rows = [[mz.get("name", "?"), mz.get("type", "?"), _measure_result(mz)] for mz in measurements]
        _table(doc, ["name", "type", "result"], rows, [0.3, 0.16, 0.54], pill_cols={1: (PILL, (59, 91, 143))})
    else:
        doc.text("No measurements.", doc.f_s, MUTED)
    _footer(doc, "03 / 05")

    # ===== page 4 · stages =====
    doc._new()
    _light_header(doc, f"{proj.upper()} · PIPELINE")
    nex = sum(1 for s in stages if s["status"] == "executed")
    nca = sum(1 for s in stages if s["status"] == "cached")
    doc.heading("Stages", f"· {len(stages)} · {nex} executed · {nca} cached")
    rows = []
    for s in stages:
        met = ", ".join(f"{k}={v}" for k, v in s["metrics"].items())
        rows.append([s["id"], s["type"], s["status"], f"{s['seconds']:.2f}s", met or "—"])
    _table(doc, ["stage", "type", "status", "time", "metrics"], rows, [0.17, 0.15, 0.13, 0.1, 0.45],
           pill_cols={1: (PILL, (59, 91, 143)), 2: lambda v: _STATUS_PILL.get(v, (PILL, MUTED))})
    _footer(doc, "04 / 05")

    # ===== page 5 · reproducibility =====
    doc._new()
    _light_header(doc, f"{proj.upper()} · PIPELINE")
    doc.heading("Reproducibility", "· cache keys & resolved parameters")
    doc.text("Every stage is content-addressed by a cache key from its inputs + resolved parameters;",
             doc.f_s, MUTED)
    doc.y += 20
    doc.text("re-running with identical keys reuses cached results — a byte-for-byte reproducible pipeline.",
             doc.f_s, MUTED)
    doc.y += 26
    rows = [[s["id"], (s.get("key") or "")[:18], ", ".join(f"{k}={v}" for k, v in s.get("params", {}).items()) or "—"]
            for s in stages]
    _table(doc, ["stage", "cache key", "resolved parameters"], rows, [0.18, 0.22, 0.6])
    _footer(doc, "05 / 05")
    return _save(doc)


def _save(doc: _Doc) -> bytes:
    buf = io.BytesIO()
    doc.pages[0].save(buf, "PDF", save_all=True, append_images=doc.pages[1:])
    return buf.getvalue()
