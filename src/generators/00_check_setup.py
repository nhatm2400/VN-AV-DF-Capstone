"""Run to check generator source, Python dependencies and weights; no generation/download."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.generators.preparation import settings
from src.generators.preparation.wav2lip import check_setup

if __name__ == '__main__':
    try:
        check_setup(settings)
        print('Setup and generator checkpoint forward passed. Video quality still needs a real smoke run.')
    except (OSError, ValueError, RuntimeError) as exc:
        raise SystemExit(f'{exc}\nSee src/generators/README.md')
