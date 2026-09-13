"""STEP 02: Run to download selected videos with audio. No license check required.
Use --dry_run to validate the list without downloading.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data.preparation._run import run
from src.data.preparation.settings import SOURCES, RAW

if __name__ == '__main__':
    run('src.data.preparation.download', ['--videos', SOURCES / 'selected_videos.csv', '--out_dir', RAW])
