"""Resume the existing 16 MiB download parts; verify the final file before use."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import subprocess
import time

CHUNK = 16 * 1024 * 1024


def file_hash(path, algorithm='sha256'):
    digest = hashlib.new(algorithm)
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def verified(path, size, expected, algorithm='sha256'):
    path = Path(path)
    if not path.is_file() or path.stat().st_size != size:
        return False
    if algorithm == 'git-sha1':
        digest = hashlib.sha1(f'blob {size}\0'.encode())
        with path.open('rb') as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b''):
                digest.update(block)
        return digest.hexdigest() == expected
    return file_hash(path, algorithm) == expected


def assemble(parts, destination, size, expected, algorithm='sha256'):
    destination, parts = Path(destination), Path(parts)
    count = (size + CHUNK - 1) // CHUNK
    temporary = destination.with_name(destination.name + '.partial')
    with temporary.open('wb') as output:
        for i in range(count):
            part = parts / f'{i:05}.part'
            if part.stat().st_size != min(CHUNK, size - i * CHUNK):
                raise ValueError(f'Incomplete download part: {part}')
            with part.open('rb') as source:
                for block in iter(lambda: source.read(1024 * 1024), b''):
                    output.write(block)
    if not verified(temporary, size, expected, algorithm):
        raise ValueError(f'Checksum mismatch: {temporary}. Keep parts for inspection; do not install this file.')
    temporary.replace(destination)
    # Only remove our numbered temporary parts after a verified final file exists.
    for i in range(count):
        (parts / f'{i:05}.part').unlink()


def download(url, destination, size, expected, algorithm='sha256', workers=4):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if verified(destination, size, expected, algorithm):
        print(f'Already verified: {destination.name}', flush=True)
        return
    parts = destination.parent / (destination.name + '.parts')
    parts.mkdir(exist_ok=True)
    identity = dict(url=url, size=size, expected=expected, algorithm=algorithm, chunk_size=CHUNK)
    lock = parts / 'download.json'
    if lock.exists() and json.loads(lock.read_text(encoding='utf-8')) != identity:
        raise ValueError(f'Download target changed; refusing to mix parts: {parts}')
    # Existing parts from the interrupted preview are accepted only after the final hash matches.
    lock.write_text(json.dumps(identity, indent=2), encoding='utf-8')
    count = (size + CHUNK - 1) // CHUNK
    completed = sum(min(CHUNK, size-i*CHUNK) for i in range(count)
                    if (parts / f'{i:05}.part').is_file()
                    and (parts / f'{i:05}.part').stat().st_size == min(CHUNK, size-i*CHUNK))
    print(f'{destination.name}: reuse {completed/1e6:.0f}/{size/1e6:.0f} MB', flush=True)
    started, downloaded = time.monotonic(), 0

    def fetch(i):
        start, end = i * CHUNK, min(size, (i+1)*CHUNK) - 1
        part = parts / f'{i:05}.part'
        if part.exists() and part.stat().st_size == end-start+1:
            return 0
        result = subprocess.run(['curl.exe', '-f', '-sS', '-L', '--connect-timeout', '20',
            '--max-time', '900', '--retry', '3', '--range', f'{start}-{end}', '-o', str(part), url])
        if result.returncode or not part.exists() or part.stat().st_size != end-start+1:
            raise RuntimeError(f'Download incomplete: {part}. Run setup again to resume.')
        return part.stat().st_size

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for future in as_completed([pool.submit(fetch, i) for i in range(count)]):
            added = future.result()
            downloaded += added
            if added:
                speed = downloaded / 1e6 / max(1, time.monotonic()-started)
                print(f'{destination.name}: {(completed+downloaded)/1e6:.0f}/{size/1e6:.0f} MB ({speed:.1f} MB/s)', flush=True)
    assemble(parts, destination, size, expected, algorithm)
    print(f'Checksum passed: {destination.name}', flush=True)
