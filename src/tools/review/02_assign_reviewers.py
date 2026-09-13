"""REVIEW 02: Set reviewer names in src/data/preparation/settings.py, then Run to split the work."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from src.data.preparation._run import run
from src.data.preparation.settings import MANIFESTS, REVIEWS, REVIEWERS

if __name__ == '__main__':
    run('src.tools.review.build_review_assignments', ['--manifest', MANIFESTS / 'clips.csv',
        '--reviewers', *REVIEWERS, '--no_shared_calibration', '--out_dir', REVIEWS / 'assignments'])
