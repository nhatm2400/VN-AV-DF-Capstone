"""One-time, explicitly launched Windows setup for the local MuseTalk preview."""
from pathlib import Path
import json
import os
import shutil
import subprocess
import sys
import venv

from src.generators.preparation.download_parts import download, file_hash

ROOT = Path(__file__).resolve().parents[3]
UPSTREAM = ROOT / 'external/MuseTalk'
REVISION = '0a89dec45a0192b824e3cf4daf96c239440c5ed8'
PYTHON = UPSTREAM / '.venv/Scripts/python.exe'
STATE = ROOT / 'cache/musetalk_setup'
TORCH_WHEEL = ROOT / '.tmp/wheels/torch-2.0.1+cu118-cp310-cp310-win_amd64.whl'
TORCH_URL = 'https://download.pytorch.org/whl/cu118/torch-2.0.1%2Bcu118-cp310-cp310-win_amd64.whl'
# From the official cu118 index, also retained in pip's local index cache.
TORCH_SHA = 'f58d75619bc96e4322343c030b893613701caa2d6db8017155da226c14171335'
WEIGHTS = [
    ('TMElyralab/MuseTalk', '3ef28bc5cff08c90ad8178a25f1b570cd800170f', '',
     ['musetalkV15/musetalk.json', 'musetalkV15/unet.pth']),
    ('stabilityai/sd-vae-ft-mse', '31f26fdeee1355a5c34592e401dd41e45d25a493', 'sd-vae',
     ['config.json', 'diffusion_pytorch_model.bin']),
    ('openai/whisper-tiny', '169d4a4341b33bc18d8881c4b69c2e104e1cc0af', 'whisper',
     ['config.json', 'pytorch_model.bin', 'preprocessor_config.json']),
    ('yzd-v/DWPose', '1a7144101628d69ee7a3768d1ee3a094070dc388', 'dwpose',
     ['dw-ll_ucoco_384.pth']),
    # This mirror is referenced by upstream download_weights.bat. Resolve once and lock it.
    ('ManyOtherFunctions/face-parse-bisent', None, 'face-parse-bisent',
     ['79999_iter.pth', 'resnet18-5c106cde.pth']),
]


def setup_environment():
    env = dict(os.environ, PYTHONUTF8='1', PIP_CACHE_DIR=str(ROOT / 'cache/pip_musetalk'),
               HF_HOME=str(ROOT / 'cache/huggingface'), HF_HUB_DISABLE_SYMLINKS_WARNING='1')
    return env


def run_logged(command):
    STATE.mkdir(parents=True, exist_ok=True)
    with (STATE / 'setup.log').open('a', encoding='utf-8') as log:
        log.write('\n' + subprocess.list2cmdline([str(x) for x in command]) + '\n')
        with subprocess.Popen([str(x) for x in command], cwd=ROOT, env=setup_environment(),
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, encoding='utf-8', errors='replace') as process:
            for line in process.stdout:
                print(line, end='', flush=True)
                log.write(line)
                log.flush()
            code = process.wait()
    if code:
        raise RuntimeError(f'Setup step failed ({code}); see {STATE / "setup.log"}')


def patch_mux():
    path = UPSTREAM / 'scripts/inference.py'
    original = 'cmd_combine_audio = f"ffmpeg -y -v warning -i {audio_path} -i {temp_vid_path} {output_vid_name}"'
    corrected = ('cmd_combine_audio = f"ffmpeg -y -v warning -i {audio_path} -i {temp_vid_path} '
                 '-map 1:v:0 -map 0:a:0 -c:v copy -c:a aac -b:a 128k -shortest {output_vid_name}"')
    text = path.read_text(encoding='utf-8')
    if corrected in text:
        return
    if original not in text:
        raise ValueError('Upstream audio mux changed; inspect before applying the encoding fix')
    path.write_text(text.replace(original, corrected, 1), encoding='utf-8')


def download_weights():
    from huggingface_hub import HfApi, get_hf_file_metadata, hf_hub_url
    models = UPSTREAM / 'models'
    STATE.mkdir(parents=True, exist_ok=True)
    lock_path = STATE / 'weights_lock.json'
    if lock_path.is_file():
        lock = json.loads(lock_path.read_text(encoding='utf-8'))
    else:
        lock = []
        for repo, revision, directory, filenames in WEIGHTS:
            revision = revision or HfApi().model_info(repo).sha
            for filename in filenames:
                url = hf_hub_url(repo, filename, revision=revision)
                meta = get_hf_file_metadata(url)
                lock.append(dict(repo=repo, revision=revision, url=url,
                    relative_path=str(Path(directory) / filename), size=meta.size,
                    expected=meta.etag, algorithm='sha256' if len(meta.etag) == 64 else 'git-sha1'))
        lock_path.write_text(json.dumps(lock, indent=2), encoding='utf-8')
    # Sequential files, four concurrent ranges each: bound bandwidth/RAM and retain interrupted parts.
    for item in lock:
        download(item['url'], models / item['relative_path'], item['size'], item['expected'], item['algorithm'])
    sfd = UPSTREAM / 'musetalk/utils/face_detection/detection/sfd/s3fd.pth'
    old_sfd = ROOT / 'external/Wav2Lip/face_detection/detection/sfd/s3fd.pth'
    if not sfd.is_file():
        if not old_sfd.is_file():
            raise FileNotFoundError(f'Missing local S3FD weight: {old_sfd}')
        shutil.copy2(old_sfd, sfd)
    provenance = dict(files=lock, s3fd_sha256=file_hash(sfd))
    (models / 'download_provenance.json').write_text(json.dumps(provenance, indent=2), encoding='utf-8')


def setup():
    if sys.version_info[:2] != (3, 10) or os.name != 'nt':
        raise RuntimeError('Use the existing vn_av_df Python 3.10 interpreter on Windows for this setup')
    if not (UPSTREAM / 'scripts/inference.py').is_file():
        raise FileNotFoundError(f'MuseTalk clone missing: {UPSTREAM}')
    revision = subprocess.check_output(['git', '-C', str(UPSTREAM), 'rev-parse', 'HEAD'], text=True).strip()
    if revision != REVISION:
        raise ValueError('The prepared setup targets a different upstream commit')
    if not PYTHON.is_file():
        venv.create(UPSTREAM / '.venv', with_pip=True)
    identity = json.loads(subprocess.check_output([str(PYTHON), '-c',
        'import sys,json; print(json.dumps(dict(prefix=sys.prefix,base=sys.base_prefix)))'], text=True))
    if Path(identity['prefix']).resolve() != PYTHON.parents[1].resolve() or identity['prefix'] == identity['base']:
        raise RuntimeError('Refusing to install outside the isolated MuseTalk venv')
    print('[1/4] Download/verify PyTorch (resume retained parts)', flush=True)
    download(TORCH_URL, TORCH_WHEEL, 2619146901, TORCH_SHA)
    print('[2/4] Install inference dependencies in external/MuseTalk/.venv', flush=True)
    run_logged([PYTHON, '-m', 'pip', 'install', '--disable-pip-version-check', 'setuptools==69.5.1', 'wheel'])
    run_logged([PYTHON, '-m', 'pip', 'install', '--disable-pip-version-check', TORCH_WHEEL])
    run_logged([PYTHON, '-m', 'pip', 'install', '--disable-pip-version-check', '--prefer-binary',
                '--no-build-isolation', '-r', ROOT / 'environments/requirements-musetalk.txt'])
    print('[3/4] Download/verify inference weights (resume retained parts)', flush=True)
    run_logged([PYTHON, '-u', '-m', 'src.generators.preparation.musetalk_setup', '--weights'])
    print('[4/4] Check dependency consistency and save versions', flush=True)
    run_logged([PYTHON, '-m', 'pip', 'check'])
    patch_mux()
    frozen = subprocess.check_output([str(PYTHON), '-m', 'pip', 'freeze'], text=True)
    (STATE / 'installed_packages.txt').write_text(frozen, encoding='utf-8')
    print('Setup finished. Next: Run 05_musetalk_check.py. No videos generated yet.', flush=True)


if __name__ == '__main__':
    download_weights() if '--weights' in sys.argv else setup()
