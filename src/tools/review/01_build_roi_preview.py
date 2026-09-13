"""REVIEW 01 (optional): Run to make mouth previews with audio; full video review also works."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from src.data.preparation._run import run
from src.data.preparation.settings import MANIFESTS, PREVIEWS

LIMIT = 0  # 0 means all clips; use a small number when checking cost.

if __name__ == '__main__':
    run('src.tools.review.build_roi_preview', ['--csv', MANIFESTS / 'clips.csv',
        '--out_dir', PREVIEWS, '--limit', LIMIT, '--skip_existing'])
