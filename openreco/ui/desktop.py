"""Desktop launcher — run the UI server and present it in a native window (pywebview) when
available, else fall back to the system browser. Keeps the UI a true local desktop app while the
server stays a plain stdlib web server.
"""

from __future__ import annotations

import threading
import webbrowser

from openreco.api import Project
from openreco.ui.server import serve


def _have_webview() -> bool:
    try:
        import webview  # noqa: F401
        return True
    except Exception:  # noqa: BLE001
        return False


def resolve_mode(mode: str) -> str:
    """auto -> 'window' if pywebview is available else 'browser'; explicit modes are honored,
    but 'window' downgrades to 'browser' when pywebview is missing."""
    if mode == "window":
        return "window" if _have_webview() else "browser"
    if mode == "browser":
        return "browser"
    return "window" if _have_webview() else "browser"


def launch(project: Project, host: str = "127.0.0.1", port: int = 8000,
           mode: str = "auto", open_browser: bool = True) -> None:
    httpd = serve(project, host, port)
    url = f"http://{host}:{port}/"
    title = f"OpenReco — {project.manifest.name}"
    if resolve_mode(mode) == "window":
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        print(f"OpenReco desktop window -> {url}")
        import webview
        webview.create_window(title, url, width=1320, height=860)
        try:
            webview.start()                    # blocks until the window is closed
        finally:
            httpd.shutdown()
            httpd.server_close()
        return
    # browser / headless fallback
    print(f"OpenReco UI -> {url}  (Ctrl+C to stop)")
    if open_browser:
        try:
            webbrowser.open(url)
        except Exception:  # noqa: BLE001
            pass
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping")
    finally:
        httpd.server_close()
