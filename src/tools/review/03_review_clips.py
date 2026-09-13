"""REVIEW 03: Set REVIEWER in settings.py, then Run to open the local review interface."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from src.data.preparation._run import run
from src.data.preparation.settings import REVIEWS, REVIEWER, PREVIEWS

if __name__ == '__main__':
    run('src.tools.review.clip_review', ['--csv', REVIEWS / 'assignments' / f'assignment_{REVIEWER}.csv',
        '--out', REVIEWS / 'results' / f'review_{REVIEWER}.csv', '--reviewer', REVIEWER,
        '--roi_dir', PREVIEWS])
