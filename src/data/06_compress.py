"""STEP 06: AFTER generation and master manifest preparation (generator is not implemented)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data.preparation._run import run
from src.data.preparation.settings import ROOT, DATASET_VERSION, MANIFESTS

CRFS = ''  # Fill the levels agreed for the experiment; no final-test defaults.
COMPRESSION_RUN = 'compression_001'

if __name__ == '__main__':
    if not CRFS and '--help' not in sys.argv:
        raise SystemExit('Set CRFS here and prepare masters.csv with real/fake provenance before running.')
    run('src.data.preparation.compress', ['--input_csv', MANIFESTS / 'masters.csv',
                             '--out_dir', ROOT / 'data' / 'compressed' / DATASET_VERSION / COMPRESSION_RUN,
                             '--crfs', CRFS])
