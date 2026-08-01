"""Make the repository root importable so `import fl` works without installation.

Its mere presence at the root also fixes pytest's rootdir detection; the explicit
sys.path insertion additionally covers being invoked from another directory.
"""

import sys
from pathlib import Path

# torch and tensorflow coexist in one process only if torch is imported FIRST:
# with TF first, torch's std::random_device breaks ("random_device could not be
# read") and the process aborts at the first torch RNG use (verified on
# torch 2.0.1+cpu / tf 2.14.1). Test collection imports TF via fl.models, so
# torch must be pulled in here, before any test module. Guarded: environments
# without torch (none currently) still run the non-torch suites.
try:  # noqa: SIM105
    import torch  # noqa: F401
except ImportError:  # pragma: no cover
    pass

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
