"""Stage implementations.

Importing this package registers all built-in stages with the engine registry. Phase 0
ships only the `dummy_*` stages used to prove the engine. Phase 1 adds the real
photogrammetry stages (ingest, sfm, georef, mvs, mesh, dsm, ortho, export).
"""

from openreco.stages import classify  # noqa: F401
from openreco.stages import contours  # noqa: F401
from openreco.stages import coverage  # noqa: F401
from openreco.stages import dsm  # noqa: F401
from openreco.stages import dtm  # noqa: F401
from openreco.stages import dummy  # noqa: F401 — import-for-side-effect (registration)
from openreco.stages import export  # noqa: F401
from openreco.stages import fuse  # noqa: F401
from openreco.stages import georef  # noqa: F401
from openreco.stages import indices  # noqa: F401
from openreco.stages import ingest  # noqa: F401
from openreco.stages import mesh  # noqa: F401
from openreco.stages import mvs  # noqa: F401
from openreco.stages import panorama  # noqa: F401
from openreco.stages import refine  # noqa: F401
from openreco.stages import sfm  # noqa: F401
from openreco.stages import splat  # noqa: F401
from openreco.stages import texture  # noqa: F401

__all__ = ["classify", "contours", "coverage", "dsm", "dtm", "dummy", "export", "fuse", "georef",
           "indices", "ingest", "mesh", "mvs", "panorama", "refine", "sfm", "splat", "texture"]
