"""REVIEW 03: Set REVIEWER in settings.py, then Run to open the local review interface."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from src.data.preparation._run import run
from src.data.preparation.settings import REVIEWS, REVIEWER, PREVIEWS

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--reviewer', default=REVIEWER)
    reviewer = parser.parse_known_args()[0].reviewer
    package = REVIEWS / 'exports' / 'clips' / reviewer
    run('src.tools.review.clip_review', ['--csv', package / f'assignment_{reviewer}.csv',
        '--out', package / f'review_{reviewer}.csv', '--reviewer', reviewer,
        '--roi_dir', package / 'roi' if (package / 'roi').is_dir() else PREVIEWS])
