"""Press Run: 10 clips x 2s, no 4K, both generators and comparison panels."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.generators.preparation.quality_preview import main

RUN_ID = 'quality_10x2s_002'

if __name__ == '__main__':
    main(RUN_ID)
