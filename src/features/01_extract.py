"""Run this file after editing SETTINGS below, or use --help for CLI overrides."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# Run Python File: paths are relative to the REPO, not the terminal directory.
CHECKPOINT = ROOT / 'weights/avhubert/base_vox_iter5.pt'
UPSTREAM = ROOT / 'external/av_hubert'
PREDICTOR = ROOT / 'weights/face_landmarks/shape_predictor_68_face_landmarks.dat'
MEAN_FACE = ROOT / 'weights/face_landmarks/20words_mean_face.npy'
INPUT_MANIFEST = ROOT / 'data/manifests/dataset_v1/features/media_pairs.csv'
INPUT_VIDEO = None  # Set to a raw clip path for one unlabelled smoke extraction.
OUTPUT = ROOT / 'cache/features/dataset_v1/extract_001'  # Must be a new directory.
START_FRAME = 0
NUM_FRAMES = 125  # 5 seconds at 25 fps; choose a window within the clip.
DEVICE = 'cpu'  # Change to 'cuda' only in a compatible GPU environment.


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=CHECKPOINT)
    parser.add_argument('--upstream', type=Path, default=UPSTREAM)
    parser.add_argument('--predictor', type=Path, default=PREDICTOR)
    parser.add_argument('--mean-face', type=Path, default=MEAN_FACE)
    source = parser.add_mutually_exclusive_group()
    source.add_argument('--manifest', type=Path)
    source.add_argument('--video', type=Path)
    parser.add_argument('--out', type=Path, default=OUTPUT)
    parser.add_argument('--start-frame', type=int, default=START_FRAME)
    parser.add_argument('--num-frames', type=int, default=NUM_FRAMES)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default=DEVICE)
    parser.add_argument('--check', action='store_true', help='Check dependencies/assets without loading weights')
    args = parser.parse_args()
    from src.features.extraction import check_assets, extract
    for key in ('checkpoint', 'upstream', 'predictor', 'mean_face', 'out', 'manifest', 'video'):
        value = getattr(args, key)
        if value is not None and not value.is_absolute():
            setattr(args, key, ROOT / value)
    try:
        check_assets(args)
        if args.check:
            print('Assets/dependencies present. Checkpoint inference has not been tested by --check.')
            return
        if args.video is None and args.manifest is None:
            args.video = Path(INPUT_VIDEO) if INPUT_VIDEO else None
            args.manifest = None if args.video else INPUT_MANIFEST
            if args.video and not args.video.is_absolute():
                args.video = ROOT / args.video
        extract(args)
    except (OSError, ValueError, RuntimeError, ImportError) as exc:
        parser.exit(1, f'{exc}\nSee src/features/README.md\n')


if __name__ == '__main__':
    main()
