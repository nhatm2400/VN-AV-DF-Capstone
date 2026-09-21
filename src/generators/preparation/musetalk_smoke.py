"""Run the prepared five-clip, two-second local MuseTalk 1.5 preview."""
from pathlib import Path
import json
import os
import subprocess
import time
from fractions import Fraction
import shutil
import wave

from src.generators.preparation.musetalk_setup import ROOT, UPSTREAM, PYTHON, patch_mux
from src.generators.preparation.download_parts import file_hash, verified

RUN = ROOT / 'data/generated/dataset_v1/musetalk_smoke_5x2s_001'


def environment():
    return dict(os.environ, PYTHONUTF8='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                HF_HOME=str(ROOT / 'cache/huggingface'), NUMBA_CACHE_DIR=str(ROOT / 'cache/numba_musetalk'),
                OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')


def input_tasks(run=RUN):
    tasks = json.loads((run / 'smoke.yaml').read_text(encoding='utf-8'))
    if len(tasks) != 5:
        raise ValueError('This preview requires exactly five tasks')
    for name, task in tasks.items():
        if name not in {f'task_{i:02}' for i in range(1, 6)} or task['result_name'] != f'{name}_musetalk_v15_fake.mp4':
            raise ValueError(f'Unexpected task/output name: {name}')
        for key in ('video_path', 'audio_path'):
            path = Path(task[key])
            path = path if path.is_absolute() else run / path
            if not path.resolve().is_relative_to((run / 'input').resolve()) or not path.is_file():
                raise ValueError(f'Missing or unexpected preview input: {path}')
            task[key] = path.resolve().as_posix()
        info = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-count_frames',
            '-select_streams', 'v:0', '-show_streams', '-of', 'json', task['video_path']], text=True))['streams'][0]
        if int(info['nb_read_frames']) != 50 or float(Fraction(info['avg_frame_rate'])) != 25:
            raise ValueError(f'Expected exactly 50 frames at 25 fps: {name}')
        with wave.open(task['audio_path'], 'rb') as audio:
            if (audio.getframerate(), audio.getnchannels(), audio.getnframes()) != (16000, 1, 32000):
                raise ValueError(f'Expected two seconds of mono 16 kHz audio: {name}')
    return tasks


def check_setup():
    required = [PYTHON, RUN / 'smoke.yaml', UPSTREAM / 'models/musetalkV15/unet.pth',
                UPSTREAM / 'models/sd-vae/diffusion_pytorch_model.bin',
                UPSTREAM / 'models/whisper/pytorch_model.bin',
                UPSTREAM / 'models/dwpose/dw-ll_ucoco_384.pth',
                UPSTREAM / 'models/face-parse-bisent/79999_iter.pth',
                UPSTREAM / 'models/face-parse-bisent/resnet18-5c106cde.pth',
                UPSTREAM / 'musetalk/utils/face_detection/detection/sfd/s3fd.pth',
                UPSTREAM / 'models/download_provenance.json']
    missing = [str(p) for p in required if not p.is_file()]
    missing += [name for name in ('ffmpeg', 'ffprobe') if shutil.which(name) is None]
    if missing:
        raise FileNotFoundError('Run 04_musetalk_setup.py first. Missing:\n' + '\n'.join(missing))
    tasks = input_tasks()
    provenance = json.loads((UPSTREAM / 'models/download_provenance.json').read_text(encoding='utf-8'))
    for item in provenance['files']:
        if not verified(UPSTREAM / 'models' / item['relative_path'], item['size'], item['expected'], item['algorithm']):
            raise ValueError(f'Weight verification failed: {item["relative_path"]}')
    sfd = UPSTREAM / 'musetalk/utils/face_detection/detection/sfd/s3fd.pth'
    if file_hash(sfd) != provenance['s3fd_sha256']:
        raise ValueError('S3FD hash changed')
    # Real CUDA and compiled MMCV operation, rather than CUDA availability alone.
    code = """
import json, torch, mmcv, mmpose, diffusers, transformers
from mmcv.ops import nms
assert torch.cuda.is_available(), 'CUDA GPU unavailable; preview must run on GPU'
boxes = torch.tensor([[0.,0.,10.,10.],[1.,1.,9.,9.]], device='cuda')
nms(boxes, torch.tensor([.9,.8], device='cuda'), .5)
from musetalk.utils.preprocessing import model, fa
print(json.dumps(dict(torch=torch.__version__, mmcv=mmcv.__version__,
                     gpu=torch.cuda.get_device_name(0), landmark_models='loaded')))
"""
    result = subprocess.run([str(PYTHON), '-c', code], cwd=UPSTREAM, env=environment(),
                            capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=180)
    if result.returncode:
        raise RuntimeError(f'MuseTalk runtime check failed:\n{result.stdout}\n{result.stderr}')
    print(result.stdout.strip())
    print('Setup and five inputs passed. Generator inference has not run yet.')
    return tasks


def make_comparisons(tasks):
    import csv
    with (RUN / 'mapping.csv').open(encoding='utf-8-sig', newline='') as handle:
        mapping = {r['task']: r for r in csv.DictReader(handle)}
    previews = RUN / 'comparison'
    previews.mkdir(exist_ok=True)
    for name, task in tasks.items():
        fake = RUN / 'output/v15' / task['result_name']
        wav2lip = Path(mapping[name]['wav2lip_fake'])
        if not wav2lip.is_absolute():
            wav2lip = ROOT / wav2lip
        if not fake.is_file() or not wav2lip.is_file():
            continue
        # All three panels get the same preview resize. Use originals for pixel-level inspection.
        chains = [f'[{i}:v]fps=25,scale=640:-2,setsar=1,setpts=PTS-STARTPTS[v{i}]' for i in range(3)]
        chains.append('[v0][v1][v2]hstack=inputs=3:shortest=1[v]')
        target = previews / f'{name}__real_wav2lip_musetalk.mp4'
        if target.exists():
            continue
        subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-n', '-i', task['video_path'], '-i', str(wav2lip),
            '-i', str(fake), '-filter_complex', ';'.join(chains), '-map', '[v]', '-map', '2:a:0', '-t', '2',
            '-c:v', 'libx264', '-crf', '18', '-c:a', 'copy', str(target)], check=True)
    print(f'Comparison panels, left to right: REAL | WAV2LIP GAN | MUSETALK: {previews}')


def generate():
    tasks = check_setup()
    output = RUN / 'output'
    if output.exists():
        raise SystemExit(f'Output already exists. Inspect it before starting a new run: {output}')
    output.mkdir()
    patch_mux()
    (RUN / 'runtime.yaml').write_text(json.dumps(tasks, indent=2), encoding='utf-8')
    log_path = RUN / 'generation.log'
    env = environment()
    command = [str(PYTHON), '-u', '-m', 'scripts.inference', '--inference_config', str(RUN / 'runtime.yaml'),
               '--result_dir', str(output), '--unet_model_path', 'models/musetalkV15/unet.pth',
               '--unet_config', 'models/musetalkV15/musetalk.json', '--version', 'v15',
               '--use_float16', '--batch_size', '1', '--saved_coord']
    revision = subprocess.check_output(['git', '-C', str(UPSTREAM), 'rev-parse', 'HEAD'], text=True).strip()
    patch = subprocess.check_output(['git', '-C', str(UPSTREAM), 'diff'], text=True, encoding='utf-8')
    (RUN / 'upstream.patch').write_text(patch, encoding='utf-8')
    provenance = dict(
        upstream_commit=revision, command=command, count=5, frames=50, fps=25, precision='fp16',
        purpose='visual_quality_preview_only', status='running',
        config_sha256=file_hash(RUN / 'runtime.yaml'),
        weights_provenance=json.loads((UPSTREAM / 'models/download_provenance.json').read_text(encoding='utf-8')),
        input_sha256={name: {key: file_hash(task[key]) for key in ('video_path', 'audio_path')}
                      for name, task in tasks.items()})
    (RUN / 'provenance.json').write_text(json.dumps(provenance, indent=2), encoding='utf-8')
    started = time.monotonic()
    print(f'Five clips x two seconds. Outputs appear one by one in {output / "v15"}', flush=True)
    print(f'Log: {log_path}', flush=True)
    with log_path.open('w', encoding='utf-8') as log:
        with subprocess.Popen(command, cwd=UPSTREAM, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, encoding='utf-8', errors='replace') as process:
            for line in process.stdout:
                print(line, end='', flush=True)
                log.write(line)
                log.flush()
            exit_code = process.wait()
    reports = []
    for task, config in tasks.items():
        path = output / 'v15' / config['result_name']
        report = dict(task=task, file_path=str(path), valid=False)
        if path.is_file():
            try:
                probe = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-count_frames',
                    '-show_streams', '-of', 'json', str(path)], text=True))
                video = next(s for s in probe['streams'] if s['codec_type'] == 'video')
                audio = next(s for s in probe['streams'] if s['codec_type'] == 'audio')
                report.update(frames=int(video['nb_read_frames']), duration=float(video['duration']),
                              width=video['width'], height=video['height'], audio_duration=float(audio['duration']))
                report['valid'] = (report['frames'] == 50 and abs(report['duration']-2) < .05
                                   and abs(report['audio_duration']-2) < .05)
            except (subprocess.CalledProcessError, ValueError, KeyError, StopIteration) as exc:
                report['error'] = str(exc)
        reports.append(report)
    valid = sum(r['valid'] for r in reports)
    summary = dict(requested=5, valid_outputs=valid, exit_code=exit_code,
                   elapsed_seconds=round(time.monotonic()-started, 1), outputs=reports,
                   status='awaiting_visual_review' if valid == 5 else 'incomplete')
    (RUN / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    provenance['status'] = summary['status']
    (RUN / 'provenance.json').write_text(json.dumps(provenance, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))
    if valid != 5 or exit_code:
        raise SystemExit(f'Inspect errors in {log_path}')
    make_comparisons(tasks)
