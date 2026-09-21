"""Raw media pairs -> immutable feature cache -> existing paired training loader."""
import csv
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import shutil

import torch

from src.features.avhubert import load_checkpoint, sha256
from src.features.media import MediaProcessor, run_media_command


def check_assets(args):
    missing = [str(path) for path in (args.checkpoint, args.predictor, args.mean_face,
               args.upstream / 'avhubert/resnet.py',
               args.upstream / 'avhubert/preparation/align_mouth.py') if not path.is_file()]
    missing += [f'Python package: {name}' for name in
                ('dlib', 'python_speech_features', 'skimage', 'tqdm')
                if importlib.util.find_spec(name) is None]
    missing += [f'Executable on PATH: {name}' for name in ('ffmpeg', 'ffprobe')
                if shutil.which(name) is None]
    if missing:
        raise FileNotFoundError('Missing requirements:\n- ' + '\n- '.join(missing))
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA is unavailable in this Python environment; select cpu')


def read_pairs(manifest):
    manifest = Path(manifest).resolve()
    with manifest.open(encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle)
        required = {'sample_id', 'label', 'split', 'media_path', 'paired_media_path',
                    'view_id', 'paired_view_id', 'start_frame', 'num_frames'}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError('Use configs/templates/media_pairs.csv; review CSV is not a feature manifest')
        rows = list(reader)
    if not rows:
        raise ValueError('Empty media manifest')
    seen = set()
    for row in rows:
        if not row['sample_id'] or row['sample_id'] in seen:
            raise ValueError('sample_id must identify a unique labelled clip AND window')
        seen.add(row['sample_id'])
        if row['label'] not in {'0', '1'} or row['split'] not in {'train', 'val', 'test'}:
            raise ValueError('Expected label 0/1 and split train/val/test; do not invent labels/splits')
        row['label'] = int(row['label'])
        row['start_frame'], row['num_frames'] = int(row['start_frame']), int(row['num_frames'])
        if row['start_frame'] < 0 or not 1 <= row['num_frames'] <= 250:
            raise ValueError('Use nonnegative start_frame and num_frames between 1 and 250')
        if not row['view_id'] or not row['paired_view_id'] or row['view_id'] == row['paired_view_id']:
            raise ValueError('Two different compression view IDs are required')
        for key in ('media_path', 'paired_media_path'):
            row[key] = (manifest.parent / row[key]).resolve()
            if not row[key].is_file():
                raise FileNotFoundError(row[key])
        if row['media_path'] == row['paired_media_path']:
            raise ValueError('Compression pair cannot point to the same media file')
    return rows


def extract(args):
    if args.out.exists():
        raise FileExistsError(f'Output already exists; choose a new run directory: {args.out}')
    if args.video:
        if not args.video.is_file():
            raise FileNotFoundError(args.video)
        rows = [dict(sample_id=args.video.stem, label=None, split='unassigned',
                     media_path=args.video, view_id='smoke',
                     start_frame=args.start_frame, num_frames=args.num_frames)]
        manifest_hash = None
    else:
        rows = read_pairs(args.manifest)
        manifest_hash = sha256(args.manifest)
    adapter, config, backbone_info = load_checkpoint(args.checkpoint, args.upstream, args.device)
    processor = MediaProcessor(args.upstream, args.predictor, args.mean_face, config)
    versions = {name: importlib.metadata.version(name) for name in
                ('torch', 'numpy', 'opencv-python', 'scipy', 'python_speech_features', 'scikit-image', 'dlib')}
    provenance = dict(backbone=backbone_info, media=processor.provenance, versions=versions,
                      device=args.device,
                      ffmpeg=run_media_command(['ffmpeg', '-version']).decode().splitlines()[0],
                      implementation={name: sha256(Path(__file__).parent / name)
                                      for name in ('avhubert.py', 'media.py', 'extraction.py')})
    extractor_id = hashlib.sha256(json.dumps(provenance, sort_keys=True).encode()).hexdigest()
    args.out.mkdir(parents=True, exist_ok=False)
    report = dict(schema='avhubert_feature_run_v1', extractor_id=extractor_id,
                  provenance=provenance, manifest_sha256=manifest_hash, samples=[])
    output_rows = []
    # A failed run has no pairs.csv. Do not silently skip a failed class/view.
    for index, row in enumerate(rows):
        print(f'[{index + 1}/{len(rows)}] {row["sample_id"]}', flush=True)
        output_row = {key: row[key] for key in ('sample_id', 'label', 'split')}
        view_specs = [('media_path', 'view_id', 'feature_path')]
        if not args.video:
            view_specs.append(('paired_media_path', 'paired_view_id', 'paired_feature_path'))
        for media_key, view_key, feature_key in view_specs:
            source_hash = sha256(row[media_key])
            preview = args.out / f'{index:06d}_{feature_key}_mouth.mp4' if getattr(args, 'previews', True) else None
            audio, video, quality = processor(row[media_key], row['start_frame'], row['num_frames'],
                                              preview_path=preview)
            a, v = adapter(audio.to(args.device), video.to(args.device))
            if (a.shape[1] != row['num_frames'] or not torch.isfinite(a).all()
                    or not torch.isfinite(v).all()):
                raise ValueError('Extractor returned invalid features')
            cache = dict(sample_id=row['sample_id'], label=row['label'], split=row['split'],
                         view_id=row[view_key], extractor_id=extractor_id,
                         timeline_id=f'25fps:{row["start_frame"]}:{row["num_frames"]}',
                         source_sha256=source_hash, audio=a[0].cpu(), visual=v[0].cpu())
            filename = f'{index:06d}_{feature_key}.pt'
            torch.save(cache, args.out / filename)
            output_row[feature_key] = filename
            report['samples'].append(dict(sample_id=row['sample_id'], view_id=row[view_key],
                                          source_path=str(row[media_key]), source_sha256=source_hash,
                                          feature_path=filename, quality=quality))
        output_rows.append(output_row)
    (args.out / 'run.json').write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    if not args.video:
        with (args.out / 'pairs.csv').open('w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
            writer.writeheader()
            writer.writerows(output_rows)
    print(f'Completed: {args.out}', flush=True)
