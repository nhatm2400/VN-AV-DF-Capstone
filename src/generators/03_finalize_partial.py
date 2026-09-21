"""After deliberate Ctrl+C: validate complete real/fake files and publish partial metadata."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.generators.preparation import settings
from src.generators.preparation.wav2lip import finalize_existing

if __name__ == '__main__':
    try:
        finalize_existing(settings)
    except (OSError, ValueError, RuntimeError) as exc:
        raise SystemExit(f'{exc}\nSee src/generators/README.md')
