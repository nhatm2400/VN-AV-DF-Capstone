"""STEP 05: Run to split reviewed real clips using the explicitly selected protocol."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data.preparation._run import run
from src.data.preparation.settings import MANIFESTS

RATIOS = '0.7,0.15,0.15'  # Source groups remain intact; actual ratios may differ.
PROTOCOL = 'source_disjoint_single_speaker'  # Preliminary results on the same speaker.
SINGLE_SPEAKER_ID = 'spk_001'  # User confirmed all current videos show the same speaker.

if __name__ == '__main__':
    speaker_args = ['--single-speaker-id', SINGLE_SPEAKER_ID] if PROTOCOL == 'source_disjoint_single_speaker' else []
    run('src.data.preparation.build_splits', ['--input', MANIFESTS / 'reviewed_clips.csv',
                                '--out', MANIFESTS / 'real_splits.csv', '--ratios', RATIOS,
                                '--protocol', PROTOCOL, *speaker_args])
