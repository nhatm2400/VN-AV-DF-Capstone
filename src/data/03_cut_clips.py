"""STEP 03: Run after a fully successful download batch. Settings below control cutting."""
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data.preparation.settings import ROOT, RAW, REAL, CUT_RUN, SOURCE_GROUP

NUM_WORKERS = 1
USE_GPU = False


def main():
    if '--help' in sys.argv:
        print(__doc__)
        print('Edit NUM_WORKERS / USE_GPU here and dataset paths in preparation/settings.py.')
        return
    from src.data.preparation.cut_clips import CutConfig, run_batch
    from src.data.preparation.manifest_io import read_rows
    os.chdir(ROOT)
    input_csv = RAW / 'download_results.csv'
    rows = read_rows(input_csv)
    if any(row.get('status') != 'downloaded' for row in rows):
        raise ValueError('Download batch has failed rows. Resolve them before cutting; none will be silently skipped.')
    config = CutConfig(tier=SOURCE_GROUP, dataset_dir=str(RAW), input_csv=str(input_csv),
                       output_root=str(REAL), run_id=CUT_RUN, expected_input_count=len(rows),
                       num_workers=NUM_WORKERS, use_hwaccel_decode=USE_GPU, use_nvenc=USE_GPU)
    run_batch(config)


if __name__ == '__main__':
    main()
