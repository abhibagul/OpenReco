"""License gate — fail CI if any installed dependency is not permissively licensed.

Enforces the project's hard constraint (docs/03-architecture.md): BSD/MIT/Apache only.
Copyleft (GPL/LGPL/AGPL/MPL) and non-commercial licenses are rejected. This is what keeps
AGPL components like OpenMVS — and non-commercial ones like SuperGlue — out of the build.

Phase 0 has no third-party runtime deps, so this passes trivially. It becomes load-bearing
in Phase 1 when pycolmap/gdal/open3d/etc. are installed. Run: python scripts/check_licenses.py
"""

from __future__ import annotations

import re
import sys
from importlib import metadata

# Regexes that mark an acceptable permissive license. Word boundaries avoid matching license
# tokens that appear as substrings of unrelated words (e.g. 'mpl' inside "Implementation").
ALLOW = (
    r"\bmit\b",
    r"\bbsd\b",
    r"\bapache\b",
    r"\bisc\b",
    r"python software foundation",
    r"\bpsf\b",
    r"\bzlib\b",
    r"unlicense",
    r"\bhpnd\b",
)
# Regexes that are hard rejections regardless of anything else. The canonical phrases below
# catch OSI classifier strings (e.g. "GNU General Public License v3 (GPLv3+)") that the bare
# SPDX tokens would miss.
DENY = (
    r"affero",
    r"\bagpl",
    r"\blgpl",
    r"\bgpl",
    r"\bmpl\b",
    r"mozilla public",
    r"gnu (general|lesser|affero) public",
    r"common development and distribution",
    r"\bcddl\b",
    r"non.?commercial",
    r"cc.?by.?nc",
    r"proprietary",
)

# Known-good packages whose metadata is ambiguous/empty but are verified permissive.
KNOWN_PERMISSIVE = {
    "pillow": "MIT-CMU / HPND (permissive)",
}

# Reviewed exceptions: packages whose license is technically copyleft but accepted for this
# project, with a recorded rationale. Each entry is auditable in the report.
#   certifi (MPL-2.0): a CA-certificate DATA bundle pulled transitively by pyproj for PROJ
#   grid downloads. MPL-2.0 is file-level copyleft; we ship it unmodified and link nothing
#   into it, so it imposes no obligation on OpenReco's own code. Accepted.
EXCEPTIONS = {
    "certifi": "MPL-2.0 CA-cert data bundle (via pyproj); shipped unmodified — accepted",
}


def _any(patterns: tuple[str, ...], text: str) -> str | None:
    for pat in patterns:
        if re.search(pat, text):
            return pat
    return None


def classify(
    name: str, license_text: str, classifiers: list[str], license_expr: str = ""
) -> tuple[str, str]:
    if name.lower() in EXCEPTIONS:
        return "ALLOW", f"reviewed exception: {EXCEPTIONS[name.lower()]}"
    # Most authoritative: the PEP 639 SPDX `License-Expression` (e.g. "MIT", "Apache-2.0").
    if license_expr:
        blob = license_expr.lower()
        if (hit := _any(DENY, blob)) and not _any(ALLOW, blob):
            return "DENY", f"license-expression matched /{hit}/"
        if (hit := _any(ALLOW, blob)):
            return "ALLOW", f"license-expression /{license_expr}/"
    # Next: the structured `License :: OSI Approved :: ...` trove classifiers; the free-text
    # `License` field often bundles vendored-license text (e.g. numpy mentions "affero"),
    # which produces false positives. Only fall back to free text when no classifiers exist.
    lic_classifiers = [c.lower() for c in classifiers if c.lower().startswith("license")]
    if lic_classifiers:
        blob = " ".join(lic_classifiers)
        # A permissive classifier wins even alongside a copyleft one (dual "MIT OR MPL"
        # means you may use the permissive option) — e.g. tqdm is MPL-2.0 OR MIT.
        if (hit := _any(ALLOW, blob)):
            return "ALLOW", f"classifier matched /{hit}/"
        if (hit := _any(DENY, blob)):
            return "DENY", f"classifier matched /{hit}/"

    blob = license_text.lower()
    if (hit := _any(DENY, blob)):
        return "DENY", f"license text matched /{hit}/"
    if (hit := _any(ALLOW, blob)):
        return "ALLOW", f"license text matched /{hit}/"
    if name.lower() in KNOWN_PERMISSIVE:
        return "ALLOW", KNOWN_PERMISSIVE[name.lower()]
    return "UNKNOWN", "no recognized license string"


def _base_name(req: str) -> str:
    return re.split(r"[\s<>=!~;\[(]", req, maxsplit=1)[0].strip().lower()


def _wanted_extra(req: str, extras: set[str]) -> bool:
    """Decide whether an extras-gated requirement is in scope. A bare runtime requirement
    (no `extra ==` marker) is always in scope; an extras-gated one only if we opted into
    that extra."""
    marker = req.split(";", 1)[1] if ";" in req else ""
    if "extra ==" not in marker:
        return True
    return any(f'extra == "{e}"' in marker or f"extra == '{e}'" in marker for e in extras)


def dependency_closure(roots: list[str], extras: set[str] | None = None) -> set[str]:
    """Installed distributions reachable from `roots` via runtime requirements (plus the
    named `extras`). This scopes the gate to what OpenReco actually ships, not the whole
    environment (dev tools and unrelated global packages don't taint a product)."""
    extras = extras or set()
    installed = {d.metadata["Name"].lower(): d for d in metadata.distributions() if d.metadata["Name"]}
    root_set = {r.lower() for r in roots}
    seen: set[str] = set()
    queue = list(root_set)
    while queue:
        name = queue.pop()
        if name in seen or name not in installed:
            continue
        seen.add(name)
        # extras only apply when resolving the root package's own optional dependencies
        active_extras = extras if name in root_set else set()
        for req in metadata.requires(installed[name].metadata["Name"]) or []:
            if not _wanted_extra(req, active_extras):
                continue
            base = _base_name(req)
            if base and base not in seen:
                queue.append(base)
    return seen


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Permissive-only license gate")
    ap.add_argument("--all", action="store_true", help="audit the whole environment, not just openreco's closure")
    ap.add_argument("--root", default="openreco", help="root package whose dependency closure is checked")
    ap.add_argument("--extras", default="", help="comma-separated extras to include (e.g. 'slice')")
    args = ap.parse_args(argv)

    if args.all:
        dists = list(metadata.distributions())
        scope = "entire environment"
    else:
        extras = {e.strip() for e in args.extras.split(",") if e.strip()}
        names = dependency_closure([args.root], extras)
        dists = [d for d in metadata.distributions() if (d.metadata["Name"] or "").lower() in names]
        extra_note = f" + extras [{','.join(sorted(extras))}]" if extras else ""
        scope = f"{args.root} dependency closure{extra_note} ({len(names)} package(s))"

    problems: list[str] = []
    unknown: list[str] = []
    for dist in dists:
        name = dist.metadata["Name"] or "?"
        lic = dist.metadata.get("License", "") or ""
        lic_expr = dist.metadata.get("License-Expression", "") or ""
        classifiers = dist.metadata.get_all("Classifier") or []
        verdict, why = classify(name, lic, classifiers, lic_expr)
        if verdict == "DENY":
            problems.append(f"  DENY    {name:30s} {why}")
        elif verdict == "UNKNOWN":
            unknown.append(f"  UNKNOWN {name:30s} {why}")

    print(f"Scope: {scope}")

    if problems:
        print("Non-permissive dependencies detected:")
        print("\n".join(sorted(problems)))
        print("\nThis violates the permissive-only constraint (docs/03-architecture.md).")
        return 1

    if unknown:
        print("Dependencies with unrecognized licenses (review manually):")
        print("\n".join(sorted(unknown)))
        # Non-fatal: print for review but don't block. Flip to `return 1` to enforce strictly.

    print("License check passed: no copyleft/non-commercial dependencies detected.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
