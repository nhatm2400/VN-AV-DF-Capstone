"""Plan and run small Wav2Lip development batches from a locked real split."""
from collections import defaultdict
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

from src.data.preparation.build_splits import inherit_variant_split, verify_splits
from src.data.preparation.manifest_io import read_rows, safe_id, unique_rows, write_rows


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def select_plan(rows, count, frames, run_id):
    safe_id(run_id)
    if not 1 <= count <= 100 or not 25 <= frames <= 250:
        raise ValueError('Use 1..100 clips and 25..250 frames for this development runner')
    verify_splits(rows)
    sources = defaultdict(list)
    for row in sorted(rows, key=lambda r: r['clip_id']):
        if row.get('label') != '0' or row.get('decision') != 'keep':
            raise ValueError('Input must contain only reviewed real clips')
        duration = float(row.get('duration', 'nan'))
        if row['split'] == 'train' and math.isfinite(duration) and duration >= frames / 25 + .2:
            sources[row['source_video']].append(row)
    # Round robin across sources, deterministic regardless of CSV row ordering.
    pool = [group[i] for i in range(max(map(len, sources.values()), default=0))
            for _, group in sorted(sources.items()) if i < len(group)]
    if len(pool) < count:
        raise ValueError(f'Only {len(pool)} eligible train clips; requested {count}')
    selected = pool[:count]
    plan = []
    for i, source in enumerate(selected):
        candidates = pool[i+1:] + pool[:i]
        donor = next((r for r in candidates if r['source_video'] != source['source_video']
                      and r['speaker_id'] == source['speaker_id']), None)
        if donor is None:
            raise ValueError('Need another train video of the same speaker for replacement audio')
        plan.append(dict(clip_id=f'{source["clip_id"]}__{run_id}__fake',
                         source_clip=source['clip_id'], audio_source_clip_id=donor['clip_id'],
                         start_frame=0, audio_start_frame=0, num_frames=frames,
                         generator='wav2lip', label=1, split='train'))
    return plan


def validate_plan(plan, rows):
    verify_splits(rows)
    index = unique_rows(rows)
    unique_rows(plan)
    for item in plan:
        if item.get('generator') != 'wav2lip' or str(item.get('label')) != '1':
            raise ValueError('Expected a Wav2Lip fake plan')
        inherited = inherit_variant_split(item, rows)
        if inherited['split'] != 'train':
            raise ValueError('This preliminary runner only uses train; leave val/test untouched')
        source, donor = index[item['source_clip']], index[item['audio_source_clip_id']]
        if source['speaker_id'] != donor['speaker_id'] or source['source_video'] == donor['source_video']:
            raise ValueError('Use same-speaker replacement audio from a different train video')
        frames = int(item['num_frames'])
        if not 25 <= frames <= 250:
            raise ValueError('Expected 25..250 frames')
        for row, field in ((source, 'start_frame'), (donor, 'audio_start_frame')):
            start = int(item[field])
            duration = float(row.get('duration', 'nan'))
            if (row.get('label') != '0' or row.get('decision') != 'keep' or start < 0
                    or not math.isfinite(duration) or (start + frames)/25 > duration):
                raise ValueError(f'Invalid reviewed source/window: {row["clip_id"]}')
    return index


def media_path(row, manifest):
    path = Path(row['file_path'])
    path = path if path.is_absolute() else Path(manifest).parent / path
    if not path.is_file():
        raise FileNotFoundError(path)
    return path.resolve()


def prepare(settings):
    if settings.PLAN_DIR.exists():
        raise FileExistsError(f'Plan exists; use a new RUN_ID: {settings.PLAN_DIR}')
    rows = read_rows(settings.MANIFEST)
    plan = select_plan(rows, settings.COUNT, settings.NUM_FRAMES, settings.RUN_ID)
    index = validate_plan(plan, rows)
    for item in plan:
        for key in ('source_clip', 'audio_source_clip_id'):
            media_path(index[item[key]], settings.MANIFEST)
    write_rows(settings.PLAN_DIR / 'plan.csv', plan)
    config = dict(schema='wav2lip_plan_v1', run_id=settings.RUN_ID,
                  manifest_sha256=digest(settings.MANIFEST),
                  plan_sha256=digest(settings.PLAN_DIR / 'plan.csv'),
                  count=len(plan), num_frames=settings.NUM_FRAMES, split='train',
                  audio_policy='same_speaker_other_train_video', selection='source_round_robin_sorted')
    (settings.PLAN_DIR / 'plan.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    print(f'Prepared {len(plan)} train-only requests: {settings.PLAN_DIR / "plan.csv"}')


def worker_command(settings, check=False):
    command = [str(settings.GENERATOR_PYTHON), str(Path(__file__).with_name('worker.py').resolve()),
               '--upstream', str(settings.UPSTREAM.resolve()),
               '--checkpoint', str(settings.CHECKPOINT.resolve()), '--device', settings.DEVICE]
    return command + (['--check'] if check else [])


def worker_environment(settings):
    env = dict(os.environ, PYTHONUTF8='1', TORCH_FORCE_WEIGHTS_ONLY_LOAD='1',
               NUMBA_CACHE_DIR=str(Path(__file__).resolve().parents[3] / 'cache/numba_wav2lip'))
    if settings.DEVICE == 'cpu':
        env['CUDA_VISIBLE_DEVICES'] = ''
    return env


def check_setup(settings):
    sfd = settings.UPSTREAM / 'face_detection/detection/sfd/s3fd.pth'
    missing = [str(p) for p in [settings.GENERATOR_PYTHON, settings.UPSTREAM / 'inference.py',
                                settings.CHECKPOINT, sfd] if not p.is_file()]
    missing += [name for name in ('ffmpeg', 'ffprobe') if shutil.which(name) is None]
    if missing:
        raise FileNotFoundError('Missing setup:\n- ' + '\n- '.join(missing))
    try:
        result = subprocess.run(worker_command(settings, check=True), env=worker_environment(settings),
                                capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=120)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError('Generator setup/checkpoint check exceeded 120 seconds; inspect environment/device') from exc
    if result.returncode:
        raise RuntimeError(f'Generator setup/checkpoint check failed:\n{result.stdout}\n{result.stderr}')
    print(result.stdout.strip())


def check_media(path, frames):
    from src.features.media import run_media_command
    info = json.loads(run_media_command(['ffprobe', '-v', 'error', '-count_frames', '-show_streams',
                                        '-of', 'json', path]))
    video = next((s for s in info['streams'] if s['codec_type'] == 'video'), None)
    audio = next((s for s in info['streams'] if s['codec_type'] == 'audio'), None)
    if not video or not audio:
        raise ValueError(f'Missing video/audio in generator output: {path}')
    if int(video['nb_read_frames']) != frames or abs(float(Fraction(video['avg_frame_rate'])) - 25) > .001:
        raise ValueError(f'Generator output has wrong frame count/FPS: {path}')
    if abs(float(audio['duration']) - frames/25) > .05:
        raise ValueError(f'Generator output has wrong audio duration: {path}')


def export_control(video, audio, destination, frames):
    from src.features.media import run_media_command
    run_media_command(['ffmpeg', '-nostdin', '-v', 'error', '-n', '-i', video, '-i', audio,
                       '-map', '0:v:0', '-map', '1:a:0', '-t', str(frames / 25),
                       '-c:v', 'libx264', '-crf', '18', '-preset', 'medium', '-pix_fmt', 'yuv420p',
                       '-c:a', 'aac', '-b:a', '128k', '-ar', '16000', '-ac', '1', destination])
    check_media(destination, frames)


def run_worker(settings, work, log_path):
    with log_path.open('x', encoding='utf-8') as log:
        result = subprocess.run(worker_command(settings), cwd=work, env=worker_environment(settings),
                                stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f'Wav2Lip exited {result.returncode}; see {log_path}')


def candidate_rows(settings, item, index, fake_path, real_path):
    frames = int(item['num_frames'])
    source, donor = index[item['source_clip']], index[item['audio_source_clip_id']]
    source_path, donor_path = media_path(source, settings.MANIFEST), media_path(donor, settings.MANIFEST)
    real_id = item['clip_id'].removesuffix('__fake') + '__real'
    check_media(fake_path, frames)
    check_media(real_path, frames)
    inherited = inherit_variant_split(item, list(index.values()))
    common = dict(split=inherited['split'], group_id=inherited['group_id'],
                  speaker_id=inherited['speaker_id'], source_video=inherited['source_video'],
                  split_protocol=inherited['split_protocol'], source_clip=source['clip_id'],
                  source_start_frame=int(item['start_frame']), num_frames=frames, duration=frames/25,
                  fps=25, source_sha256=digest(source_path), review_status='pending',
                  run_id=settings.RUN_ID, encoding='libx264_crf18_aac128k_16k_mono')
    real = dict(common, clip_id=real_id, file_path=str(real_path.resolve()), label=0, generator='',
                audio_source_clip_id=source['clip_id'], audio_start_frame=int(item['start_frame']))
    fake = dict(common, clip_id=item['clip_id'], file_path=str(fake_path.resolve()), label=1,
                generator='wav2lip', audio_source_clip_id=donor['clip_id'],
                audio_start_frame=int(item['audio_start_frame']), audio_source_sha256=digest(donor_path))
    return [real, fake]


def generate_one(settings, item, index, media_dir, log_path):
    from src.features.media import decode_window, run_media_command
    frames = int(item['num_frames'])
    source, donor = index[item['source_clip']], index[item['audio_source_clip_id']]
    source_path, donor_path = media_path(source, settings.MANIFEST), media_path(donor, settings.MANIFEST)
    fake_path = media_dir / (item['clip_id'] + '.mp4')
    real_id = item['clip_id'].removesuffix('__fake') + '__real'
    real_path = media_dir / (real_id + '.mp4')
    with tempfile.TemporaryDirectory(prefix='wav2lip_') as temporary:
        work = Path(temporary)
        (work / 'source').mkdir()
        (work / 'donor').mkdir()
        video, _ = decode_window(source_path, int(item['start_frame']), frames, work / 'source')
        decode_window(donor_path, int(item['audio_start_frame']), frames, work / 'donor')
        # Both real and fake use decoded 25 fps frames. No lossy pre-encoding for either.
        run_media_command(['ffmpeg', '-nostdin', '-v', 'error', '-n', '-i', video,
                           '-an', '-c:v', 'ffv1', work / 'face.avi'])
        # Upstream mel windows need right context. Pad inference audio, then trim BOTH
        # outputs to the requested duration; no video looping survives the final trim.
        run_media_command(['ffmpeg', '-nostdin', '-v', 'error', '-n', '-i', work / 'donor/audio.wav',
                           '-af', 'apad=pad_dur=0.2', '-c:a', 'pcm_s16le', work / 'audio.wav'])
        run_worker(settings, work, log_path)
        export_control(work / 'generated.mkv', work / 'donor/audio.wav', fake_path, frames)
        export_control(video, work / 'source/audio.wav', real_path, frames)
    return candidate_rows(settings, item, index, fake_path, real_path)


def load_locked_plan(settings):
    plan_path = settings.PLAN_DIR / 'plan.csv'
    config = json.loads((settings.PLAN_DIR / 'plan.json').read_text(encoding='utf-8'))
    if (digest(settings.MANIFEST) != config['manifest_sha256']
            or digest(plan_path) != config['plan_sha256']):
        raise ValueError('Manifest/plan changed after preparation; prepare a new RUN_ID')
    if settings.RUN_ID != config['run_id']:
        raise ValueError('RUN_ID differs from prepared plan')
    plan, rows = read_rows(plan_path), read_rows(settings.MANIFEST)
    return config, plan, validate_plan(plan, rows)


def finalize_existing(settings):
    """Publish metadata for complete pairs after a deliberate Ctrl+C interruption."""
    config, plan, index = load_locked_plan(settings)
    if not settings.MEDIA_DIR.is_dir():
        raise FileNotFoundError(settings.MEDIA_DIR)
    if (settings.PLAN_DIR / 'candidates.csv').exists() or (settings.PLAN_DIR / 'summary.json').exists():
        raise FileExistsError('This run is already finalized')
    successful, incomplete = [], []
    for item in plan:
        fake_path = settings.MEDIA_DIR / (item['clip_id'] + '.mp4')
        real_path = settings.MEDIA_DIR / (item['clip_id'].removesuffix('__fake') + '__real.mp4')
        existing = (fake_path.exists(), real_path.exists())
        if existing == (False, False):
            continue
        if existing != (True, True):
            incomplete.append(dict(clip_id=item['clip_id'], error='Only one side of real/fake pair exists'))
            continue
        try:
            successful.extend(candidate_rows(settings, item, index, fake_path, real_path))
        except (OSError, ValueError, RuntimeError) as exc:
            incomplete.append(dict(clip_id=item['clip_id'], error=str(exc)))
    if not successful:
        raise ValueError('No complete real/fake pairs to finalize')
    write_rows(settings.PLAN_DIR / 'candidates.csv', successful)
    if incomplete:
        write_rows(settings.PLAN_DIR / 'failures.csv', incomplete)
    completed = len(successful) // 2
    summary = dict(requested=len(plan), successful_pairs=completed, failures=len(incomplete),
                   unattempted=len(plan)-completed-len(incomplete), partial=True,
                   status='partial_awaiting_visual_review')
    (settings.PLAN_DIR / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary))
    print(f'Finalized existing complete pairs: {settings.PLAN_DIR / "candidates.csv"}')


def generate(settings):
    config, plan, index = load_locked_plan(settings)
    if settings.MEDIA_DIR.exists():
        raise FileExistsError(f'Generation run already exists; use a new RUN_ID: {settings.MEDIA_DIR}')
    check_setup(settings)
    for item in plan:
        for key in ('source_clip', 'audio_source_clip_id'):
            media_path(index[item[key]], settings.MANIFEST)
    code_files = sorted(settings.UPSTREAM.rglob('*.py'))
    code_hashes = {str(p.relative_to(settings.UPSTREAM)): digest(p) for p in code_files}
    provenance = dict(config, generator='wav2lip', device=settings.DEVICE,
                      checkpoint_sha256=digest(settings.CHECKPOINT),
                      s3fd_sha256=digest(settings.UPSTREAM / 'face_detection/detection/sfd/s3fd.pth'),
                      upstream_python_sha256=code_hashes,
                      adapter_sha256=digest(Path(__file__)),
                      worker_sha256=digest(Path(__file__).with_name('worker.py')),
                      inference_audio_right_padding_seconds=.2,
                      intermediate='ffv1_pcm16', final_encoding='libx264_crf18_aac128k_16k_mono',
                      pads=[0, 10, 0, 0], smoothing=True, face_batch=1, wav2lip_batch=8)
    settings.MEDIA_DIR.mkdir(parents=True, exist_ok=False)
    (settings.PLAN_DIR / 'logs').mkdir(exist_ok=False)
    (settings.PLAN_DIR / 'generation.json').write_text(json.dumps(provenance, indent=2), encoding='utf-8')
    successful, failures = [], []
    for number, item in enumerate(plan, 1):
        print(f'[{number}/{len(plan)}] {item["clip_id"]}', flush=True)
        try:
            successful.extend(generate_one(settings, item, index, settings.MEDIA_DIR,
                              settings.PLAN_DIR / 'logs' / f'{item["clip_id"]}.log'))
        except (OSError, ValueError, RuntimeError) as exc:
            failures.append(dict(clip_id=item['clip_id'], error=str(exc)))
            print(f'FAILED: {exc}', flush=True)
    if successful:
        write_rows(settings.PLAN_DIR / 'candidates.csv', successful)
    if failures:
        write_rows(settings.PLAN_DIR / 'failures.csv', failures)
    summary = dict(requested=len(plan), successful_pairs=len(successful)//2, failures=len(failures),
                   status='incomplete' if failures else 'awaiting_visual_review')
    (settings.PLAN_DIR / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary))
    if failures:
        raise RuntimeError('Some clips failed; inspect failures.csv/logs before preparing another run')
    print(f'Open generated videos for review: {settings.MEDIA_DIR}')
