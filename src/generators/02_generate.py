"""Run after setup and plan: generate the small batch, controls, metadata and logs."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.generators.preparation import settings
from src.generators.preparation.wav2lip import generate

if __name__ == '__main__':
    try:
        generate(settings)
    except (OSError, ValueError, RuntimeError) as exc:
        raise SystemExit(f'{exc}\nSee src/generators/README.md')
