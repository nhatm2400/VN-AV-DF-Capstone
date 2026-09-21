"""Offline check: five inputs, weight hashes, imports and the CUDA/MMCV runtime."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.generators.preparation.musetalk_smoke import check_setup

if __name__ == '__main__':
    try:
        check_setup()
    except (OSError, ValueError, RuntimeError) as exc:
        raise SystemExit(f'{exc}\nSee src/generators/README.md')
