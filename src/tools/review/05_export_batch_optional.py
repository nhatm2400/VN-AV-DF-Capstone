"""REVIEW 05 (optional): Package your assigned clips for review on another machine."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from src.data.preparation._run import run
from src.data.preparation.settings import REVIEWS, REVIEWER

if __name__ == '__main__':
    run('src.tools.review.export_review_batch', ['--csv', REVIEWS / 'assignments' / f'assignment_{REVIEWER}.csv',
        '--out_dir', REVIEWS / 'exports' / 'clips' / REVIEWER])
