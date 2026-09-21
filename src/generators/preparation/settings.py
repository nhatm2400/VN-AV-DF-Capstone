"""Small development batch; edit before Run, use a new RUN_ID for each attempt."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
DATASET = 'dataset_v1'
RUN_ID = 'wav2lip_gpu_batch_004'  # One-clip GPU smoke passed; new output for the next batch.
MANIFEST = ROOT / 'data/manifests' / DATASET / 'real_splits.csv'
PLAN_DIR = ROOT / 'data/manifests' / DATASET / 'generators' / RUN_ID
MEDIA_DIR = ROOT / 'data/generated' / DATASET / RUN_ID
UPSTREAM = ROOT / 'external/Wav2Lip'
CHECKPOINT = ROOT / 'weights/wav2lip/wav2lipGAN.pth'
GENERATOR_PYTHON = Path(sys.executable)  # Can point to a separate environment.
COUNT = 50
NUM_FRAMES = 100  # Four seconds at the normalized 25 fps.
DEVICE = 'cuda'
