"""Make the repository root importable so `import fl` works without installation.

Its mere presence at the root also fixes pytest's rootdir detection; the explicit
sys.path insertion additionally covers being invoked from another directory.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
