"""Ten paired 2s train previews, inline preprocessing, no inputs above 1080p."""
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

from src.data.preparation.manifest_io import read_rows, write_rows, safe_id
from src.generators.preparation import settings, wav2lip, musetalk_smoke, musetalk_setup

ROOT = settings.ROOT
COUNT = 10
FRAMES = 50


def probe(path):
    return json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
        '-show_entries', 'stream=width,height,nb_frames,avg_frame_rate', '-of', 'json', str(path)],
        text=True))['streams'][0]


def allowed_size(info):
    return max(info['width'], info['height']) <= 1920 and min(info['width'], info['height']) <= 1080


def make_comparison(real, wav, muse, target):
    from src.features.media import run_media_command
    target = Path(target)
    temporary = target.with_suffix('.building.mp4')
    filters = ';'.join(f'[{i}:v]fps=25,scale=640:-2,setsar=1,settb=1/25,setpts=N[v{i}]'
                       for i in range(3))
    filters += ';[v0][v1][v2]hstack=inputs=3:shortest=1[v]'
    run_media_command(['ffmpeg','-v','error','-y','-i',real,'-i',wav,'-i',muse,
        '-filter_complex',filters,'-map','[v]','-map','2:a:0','-t','2','-r','25',
        '-fps_mode','cfr','-frames:v','50','-c:v','libx264','-crf','18','-c:a','copy',temporary])
    wav2lip.check_media(temporary, FRAMES)
    temporary.replace(target)


def prepare(run):
    from src.features.media import decode_window, run_media_command
    rows = read_rows(settings.MANIFEST)
    pool = wav2lip.select_plan(rows, min(100, sum(r['split'] == 'train' and
        float(r['duration']) >= 2.2 for r in rows)), FRAMES, run.name)
    index = wav2lip.validate_plan(pool, rows)
    selected, rejected, sizes = [], [], {}
    for item in pool:
        for key in ('source_clip', 'audio_source_clip_id'):
            cid = item[key]
            if cid not in sizes:
                sizes[cid] = probe(wav2lip.media_path(index[cid], settings.MANIFEST))
        if not all(allowed_size(sizes[item[k]]) for k in ('source_clip', 'audio_source_clip_id')):
            rejected.append(dict(source_clip=item['source_clip'], reason='source_or_donor_above_1080p'))
            continue
        selected.append(item)
        if len(selected) == COUNT:
            break
    if len(selected) != COUNT:
        raise ValueError(f'Only {len(selected)} eligible non-4K pairs among first {len(pool)} candidates')
    run.mkdir(parents=True, exist_ok=False)
    (run / 'logs').mkdir()
    tasks = []
    for i, item in enumerate(selected, 1):
        task = f'task_{i:02}'
        folder = run / 'input' / task
        folder.mkdir(parents=True)
        source = wav2lip.media_path(index[item['source_clip']], settings.MANIFEST)
        donor = wav2lip.media_path(index[item['audio_source_clip_id']], settings.MANIFEST)
        (folder / 'source').mkdir()
        (folder / 'donor').mkdir()
        video, _ = decode_window(source, 0, FRAMES, folder / 'source')
        decode_window(donor, 0, FRAMES, folder / 'donor')
        # One lossless 50-frame file is physically shared by BOTH generators.
        face = folder / 'face.avi'
        run_media_command(['ffmpeg', '-v', 'error', '-n', '-i', video, '-an',
                           '-frames:v', str(FRAMES), '-c:v', 'ffv1', face])
        for name, origin in [('driver.wav', folder/'donor/audio.wav'),
                             ('original.wav', folder/'source/audio.wav')]:
            run_media_command(['ffmpeg', '-v', 'error', '-n', '-i', origin,
                               '-t', '2', '-ac', '1', '-ar', '16000', '-c:a', 'pcm_s16le', folder/name])
        info = probe(face)
        if not allowed_size(info) or int(info['nb_frames']) != FRAMES:
            raise ValueError(f'Unexpected dimensions/frame count: {face}')
        tasks.append(dict(task=task, **item, video_path=str(face),
            audio_path=str(folder/'driver.wav'), original_audio=str(folder/'original.wav'),
            width=info['width'], height=info['height'], source_sha256=wav2lip.digest(source),
            donor_sha256=wav2lip.digest(donor), video_sha256=wav2lip.digest(face),
            audio_sha256=wav2lip.digest(folder/'driver.wav')))
        # Temporary decoder outputs are inside this newly created task only.
        for directory in (folder/'source', folder/'donor'):
            resolved = directory.resolve()
            if not resolved.is_relative_to((run/'input').resolve()):
                raise ValueError('Decoder temporary directory escaped run')
            shutil.rmtree(resolved)
        print(f'Prepared {i}/{COUNT}: {task} {info["width"]}x{info["height"]}', flush=True)
    write_rows(run/'mapping.csv', tasks)
    (run/'plan.json').write_text(json.dumps(dict(tasks=tasks, excluded=rejected,
        manifest_sha256=wav2lip.digest(settings.MANIFEST), count=COUNT, frames=FRAMES,
        fps=25, split='train', purpose='quality_preview_only',
        detection_long_side=960, cudnn_benchmark=False, blend='soft_lower_face_v2_bottom_feather',
        note='One source video and donor waveform shared by both generators; original audio is separate.'),
        indent=2), encoding='utf-8')
    return tasks


def generate(run, tasks):
    from src.features.media import run_media_command
    started = time.monotonic()
    for name in ('wav2lip', 'real', 'musetalk'):
        (run/'output'/name).mkdir(parents=True, exist_ok=True)
    (run/'comparison').mkdir(exist_ok=True)
    configuration = SimpleNamespace(**{k: getattr(settings, k) for k in (
        'UPSTREAM', 'CHECKPOINT', 'DEVICE')}, GENERATOR_PYTHON=Path(sys.executable))
    wav2lip.check_setup(configuration)
    musetalk_setup.patch_mux()
    provenance = dict(status='running', started=time.strftime('%Y-%m-%dT%H:%M:%S'),
        wav2lip_checkpoint_sha256=wav2lip.digest(settings.CHECKPOINT),
        musetalk_weights=json.loads((musetalk_setup.UPSTREAM/'models/download_provenance.json').read_text()),
        plan_sha256=wav2lip.digest(run/'plan.json'),
        code_sha256={p.name: wav2lip.digest(p) for p in Path(__file__).parent.glob('*.py')},
        upstream_revisions={name: subprocess.check_output(['git','-C',str(ROOT/'external'/name),
            'rev-parse','HEAD'],text=True).strip() for name in ('Wav2Lip','MuseTalk')})
    if (run/'provenance.json').exists():
        previous = json.loads((run/'provenance.json').read_text(encoding='utf-8'))
        provenance['previous_attempt'] = previous
    (run/'provenance.json').write_text(json.dumps(provenance, indent=2), encoding='utf-8')
    for i, task in enumerate(tasks, 1):
        fake_path = run/'output/wav2lip'/f'{task["task"]}.mp4'
        real_path = run/'output/real'/f'{task["task"]}.mp4'
        if fake_path.exists() and real_path.exists():
            wav2lip.check_media(fake_path, FRAMES)
            wav2lip.check_media(real_path, FRAMES)
            print(f'Wav2Lip {i}/{COUNT}: reuse verified {task["task"]}', flush=True)
            continue
        if fake_path.exists() or real_path.exists():
            raise ValueError(f'Incomplete real/fake pair; inspect {task["task"]} before retrying')
        print(f'Wav2Lip {i}/{COUNT}: {task["task"]}', flush=True)
        with tempfile.TemporaryDirectory(prefix='quality_wav2lip_') as temp:
            work = Path(temp)
            shutil.copyfile(task['video_path'], work/'face.avi')
            run_media_command(['ffmpeg','-v','error','-n','-i',task['audio_path'],
                               '-af','apad=pad_dur=0.2','-c:a','pcm_s16le',work/'audio.wav'])
            with (run/'logs'/f'{task["task"]}_wav2lip.log').open('w',encoding='utf-8') as log:
                subprocess.run(wav2lip.worker_command(configuration)+['--quality-preview'],
                    cwd=work, env=wav2lip.worker_environment(configuration), stdout=log,
                    stderr=subprocess.STDOUT, check=True)
            wav2lip.export_control(work/'generated.mkv',task['audio_path'],
                                   run/'output/wav2lip'/f'{task["task"]}.mp4',FRAMES)
        wav2lip.export_control(task['video_path'],task['original_audio'],
                               run/'output/real'/f'{task["task"]}.mp4',FRAMES)
    config = {}
    for task in tasks:
        output = run/'output/musetalk/v15'/f'{task["task"]}.mp4'
        if output.exists():
            wav2lip.check_media(output, FRAMES)
        else:
            config[task['task']] = dict(video_path=task['video_path'],audio_path=task['audio_path'],
                                       result_name=task['task']+'.mp4')
    (run/'musetalk.yaml').write_text(json.dumps(config,indent=2),encoding='utf-8')
    command = [str(musetalk_setup.PYTHON), '-u', str(Path(__file__).with_name('musetalk_preview_worker.py')),
        '--inference_config',str(run/'musetalk.yaml'),'--result_dir',str(run/'output/musetalk'),
        '--unet_model_path','models/musetalkV15/unet.pth','--unet_config','models/musetalkV15/musetalk.json',
        '--version','v15','--use_float16','--batch_size','1','--saved_coord']
    print('MuseTalk 10 x 2s; see logs/musetalk.log',flush=True)
    if config:
        with (run/'logs/musetalk.log').open('a',encoding='utf-8') as log:
            log.write(f'\nResuming {len(config)} pending MuseTalk tasks\n')
            log.flush()
            subprocess.run(command,cwd=musetalk_setup.UPSTREAM,env=musetalk_smoke.environment(),
                           stdout=log,stderr=subprocess.STDOUT,check=True)
    for task in tasks:
        name = task['task']+'.mp4'
        real, wav, muse = (run/'output/real'/name,run/'output/wav2lip'/name,
                           run/'output/musetalk/v15'/name)
        wav2lip.check_media(muse, FRAMES)
        comparison = run/'comparison'/name
        if comparison.exists():
            try:
                wav2lip.check_media(comparison, FRAMES)
                continue
            except ValueError:
                pass  # Rebuild only the derived preview, never the generated originals.
        make_comparison(real, wav, muse, comparison)
    provenance.update(status='awaiting_visual_review',elapsed_seconds=round(time.monotonic()-started,1),
                      valid_pairs=COUNT, comparisons=COUNT)
    (run/'provenance.json').write_text(json.dumps(provenance,indent=2),encoding='utf-8')
    print(f'Done: {run / "comparison"}\nLEFT: original video (different speech); MIDDLE: Wav2Lip; RIGHT: MuseTalk. Shared playback audio is the donor.',flush=True)


def main(run_id, prepare_only=False):
    safe_id(run_id)
    run = ROOT/'data/generated/dataset_v1'/run_id
    if run.exists():
        plan = json.loads((run/'plan.json').read_text(encoding='utf-8'))
        if plan['manifest_sha256'] != wav2lip.digest(settings.MANIFEST):
            raise ValueError('Manifest changed since preparation; choose a new RUN_ID')
        tasks = plan['tasks']
        if len(tasks) != COUNT:
            raise ValueError('Incomplete prepared plan')
        for task in tasks:
            for path, sha in (('video_path','video_sha256'),('audio_path','audio_sha256')):
                if wav2lip.digest(task[path]) != task[sha]:
                    raise ValueError(f'Prepared input changed: {task[path]}')
        print('Reusing verified prepared inputs; complete outputs will be checked and skipped.', flush=True)
    else:
        tasks = prepare(run)
    if not prepare_only:
        generate(run,tasks)
