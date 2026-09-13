"""STEP 05: AFTER review and speaker identification. Run to build real_splits.csv."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data.preparation._run import run
from src.data.preparation.settings import MANIFESTS

RATIOS = '0.7,0.15,0.15'  # Source/host groups remain intact; actual ratios may differ.

if __name__ == '__main__':
    run('src.data.preparation.build_splits', ['--input', MANIFESTS / 'reviewed_clips.csv',
                                '--out', MANIFESTS / 'real_splits.csv', '--ratios', RATIOS])
