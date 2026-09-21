"""Run locally after 04/05: exactly five two-second MuseTalk 1.5 previews."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.generators.preparation.musetalk_smoke import generate

if __name__ == '__main__':
    try:
        generate()
    except (OSError, ValueError, RuntimeError) as exc:
        raise SystemExit(f'{exc}\nSee src/generators/README.md')
