"""PyInstaller entry point — the frozen binary is just the OpenReco CLI.

`openreco.exe` (or the mac/linux equivalent) behaves exactly like the `openreco` console script:
`openreco.exe doctor`, `openreco.exe init …`, `openreco.exe ui …`, `openreco.exe run …`.
"""

import multiprocessing
import sys

from openreco.cli import main

if __name__ == "__main__":
    multiprocessing.freeze_support()      # safe no-op on the main process; needed if children spawn
    sys.exit(main())
