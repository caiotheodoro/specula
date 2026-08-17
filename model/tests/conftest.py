"""Put forge on sys.path so model tests can import specula_forge."""

import sys
from pathlib import Path

_forge_src = Path(__file__).resolve().parents[2] / "forge" / "src"
if str(_forge_src) not in sys.path:
    sys.path.insert(0, str(_forge_src))
