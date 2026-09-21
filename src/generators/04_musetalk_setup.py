"""Run tomorrow on a fast connection: resume downloads and install the isolated venv."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.generators.preparation.musetalk_setup import setup

if __name__ == '__main__':
    try:
        setup()
    except (OSError, ValueError, RuntimeError) as exc:
        raise SystemExit(f'{exc}\nSee src/generators/README.md')
