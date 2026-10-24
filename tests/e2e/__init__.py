# tests/e2e package
import sys
from pathlib import Path
root = str(Path(__file__).resolve().parent.parent.parent)
if root not in sys.path:
    sys.path.insert(0, root)
