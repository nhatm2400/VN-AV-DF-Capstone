"""Edit these settings, then open a numbered script and select Run Python File."""
from pathlib import Path

DATASET_VERSION = 'dataset_v1'
DOWNLOAD_RUN = 'download_001'  # Rerun to resume; change when selecting a different source list.
CUT_RUN = 'cut_001'
SOURCE_GROUP = 'podcast'  # Fallback when a video has no tier; not a license label.
REVIEWERS = ['member_1', 'member_2', 'member_3']
REVIEWER = 'member_1'  # Set your own name before opening review.

ROOT = Path(__file__).resolve().parents[3]
SOURCES = ROOT / 'data' / 'sources' / DATASET_VERSION
RAW = ROOT / 'data' / 'raw' / DATASET_VERSION / DOWNLOAD_RUN
REAL = ROOT / 'data' / 'real' / DATASET_VERSION
MANIFESTS = ROOT / 'data' / 'manifests' / DATASET_VERSION
REVIEWS = MANIFESTS / 'reviews'
PREVIEWS = ROOT / 'cache' / 'previews' / DATASET_VERSION
