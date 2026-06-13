"""Run report — a self-contained HTML summary for auditability and reproducibility.

Phase 0 reports stage status, timing, the content-address keys, parameters, and any QA
issues. Phase 1+ enriches this with reprojection error, GCP residuals, and coverage maps.
"""

from __future__ import annotations

import html
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from openreco.engine.runner import RunOutcome

_STATUS_COLOR = {
    "executed": "#2563eb",
    "cached": "#16a34a",
    "failed": "#dc2626",
    "skipped": "#a16207",
    "cancelled": "#6b7280",
}


def write_report(outcome: "RunOutcome", path: Path) -> None:
    d = outcome.to_dict()
    rows = []
    for s in d["stages"]:
        color = _STATUS_COLOR.get(s["status"], "#374151")
        issues = "".join(
            f"<li><b>{html.escape(i['severity'])}</b>: {html.escape(i['message'])}"
            + (f" <i>({html.escape(i['hint'])})</i>" if i.get("hint") else "")
            + "</li>"
            for i in s["issues"]
        )
        issues_html = f"<ul>{issues}</ul>" if issues else "<span class=muted>none</span>"
        metrics = ", ".join(f"{html.escape(k)}={html.escape(str(v))}" for k, v in s["metrics"].items())
        rows.append(
            f"<tr>"
            f"<td><code>{html.escape(s['id'])}</code></td>"
            f"<td>{html.escape(s['type'])}</td>"
            f"<td><span style='color:{color};font-weight:600'>{html.escape(s['status'])}</span></td>"
            f"<td>{s['seconds']:.3f}s</td>"
            f"<td><code class=muted>{html.escape(s['key'][:16])}</code></td>"
            f"<td>{html.escape(metrics) or '<span class=muted>—</span>'}</td>"
            f"<td>{issues_html}</td>"
            f"</tr>"
        )

    status_badge = (
        "<span style='color:#16a34a'>OK</span>" if d["ok"] else "<span style='color:#dc2626'>FAILED</span>"
    )
    plat = d["platform"]
    body = f"""<!doctype html>
<html lang=en><head><meta charset=utf-8>
<title>OpenReco run — {html.escape(d['project'])}</title>
<style>
 body {{ font: 14px/1.5 system-ui, sans-serif; margin: 2rem; color: #111; }}
 h1 {{ font-size: 1.4rem; }}
 table {{ border-collapse: collapse; width: 100%; margin-top: 1rem; }}
 th, td {{ text-align: left; padding: .5rem .7rem; border-bottom: 1px solid #e5e7eb; vertical-align: top; }}
 th {{ background: #f9fafb; }}
 code {{ background: #f3f4f6; padding: 0 .25rem; border-radius: 3px; }}
 .muted {{ color: #9ca3af; }}
 .meta {{ color: #4b5563; }}
</style></head><body>
<h1>OpenReco run — {html.escape(d['project'])} {status_badge}</h1>
<p class=meta>
  openreco {html.escape(d['openreco_version'])} ·
  python {html.escape(plat['python'])} · {html.escape(plat['system'])}/{html.escape(plat['machine'])}<br>
  started {html.escape(d['started'])} · finished {html.escape(d['finished'])}
</p>
<table>
 <thead><tr><th>stage</th><th>type</th><th>status</th><th>time</th><th>cache key</th><th>metrics</th><th>QA issues</th></tr></thead>
 <tbody>{''.join(rows)}</tbody>
</table>
<p class=muted>Reproducibility: every stage's cache key is a hash of its type, version, parameters,
and upstream keys. Re-running with the same manifest reuses cached stages (status=cached).</p>
</body></html>"""
    path.write_text(body, encoding="utf-8")
