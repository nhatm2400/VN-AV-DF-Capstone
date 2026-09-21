"""Run to select 10 train clips and audio donors; writes metadata only."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.generators.preparation import settings
from src.generators.preparation.wav2lip import prepare

if __name__ == '__main__':
    try:
        prepare(settings)
    except (OSError, ValueError, RuntimeError) as exc:
        raise SystemExit(str(exc))
