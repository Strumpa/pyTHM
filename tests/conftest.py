"""
Shared pytest configuration for pyTHM tests.

All data paths are anchored to ``Path(__file__).parent`` so that tests work
regardless of the working directory from which ``pytest`` is invoked
(e.g. ``pyTHM/`` or ``pyTHM/tests/``).
"""

from pathlib import Path

# ── directory anchors ─────────────────────────────────────────────────────
TESTS_DIR = Path(__file__).resolve().parent          # .../pyTHM/tests
DATA_DIR = (TESTS_DIR / ".." / "data").resolve()     # .../pyTHM/data
OUTPUTS_DIR = (TESTS_DIR / "outputs").resolve()      # .../pyTHM/tests/outputs

# ----- yaml data for starterDD related imports :

GE14_DATA = str(DATA_DIR / "GE14")
PSBT_DATA = str(DATA_DIR / "PSBT") 