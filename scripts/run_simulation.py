"""Script entrypoint to run the civilization simulation."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from civilization import run


if __name__ == "__main__":
    run()

