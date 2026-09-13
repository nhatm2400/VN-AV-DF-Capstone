"""Download selected YouTube videos. License review is an optional separate step.

--dry_run validates manifests without network or media writes.
"""
import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from src.data.preparation.manifest_io import read_rows, unique_rows, write_rows
from src.data.preparation.collect import normalize_sources


def valid_media(path):
    if not path.is_file() or not path.stat().st_size:
        return False
    result = subprocess.run(['ffprobe', '-v', 'error', '-show_entries',
                             'stream=codec_type', '-of', 'json', str(path)], capture_output=True)
    if result.returncode:
        return False
    kinds = {stream.get('codec_type') for stream in json.loads(result.stdout).get('streams', [])}
    return {'video', 'audio'} <= kinds


def save_progress(path, rows):
    """Only the mutable download status is replaced; dataset manifests stay immutable."""
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with tempfile.NamedTemporaryFile(mode='w', newline='', encoding='utf-8',
                                     dir=path.parent, suffix='.partial', delete=False) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        temporary = handle.name
    os.replace(temporary, path)


def approved_sources(videos, rights):
    sources = unique_rows(normalize_sources(videos), 'video_id')
    permissions = unique_rows(rights, 'video_id')
    for vid in sources:
        if not re.fullmatch(r'[A-Za-z0-9_-]{11}', vid):
            raise ValueError('Use canonical YouTube video IDs')
        permission = permissions.get(vid, {})
        if any(permission.get(field, '').strip().lower() != 'yes' for field in ('research_allowed','acquisition_allowed')) or not permission.get('evidence_ref', '').strip():
            raise ValueError(f'Source needs documented research/acquisition permission: {vid}')
    return list(sources.values())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--videos', required=True)
    parser.add_argument('--rights', help='optional: enforce documented permission when supplied')
    parser.add_argument('--out_dir', required=True, help='download directory; rerun to resume the same sources')
    parser.add_argument('--limit', type=int, default=0, help='try at most N incomplete videos; 0 means all')
    parser.add_argument('--dry_run', action='store_true')
    args = parser.parse_args()
    videos = read_rows(args.videos)
    rows = (approved_sources(videos, read_rows(args.rights)) if args.rights
            else normalize_sources(videos))
    if args.dry_run:
        print(f'Validated {len(rows)} selected sources; no network/media writes')
        return
    import yt_dlp
    node = shutil.which('node')
    if not node:
        raise RuntimeError('Install Node.js >= 22 and ensure node is on PATH before downloading')
    output = Path(args.out_dir)
    output.mkdir(parents=True, exist_ok=True)
    snapshot = output / 'download_sources.csv'
    if snapshot.exists():
        if unique_rows(read_rows(snapshot), 'video_id') != unique_rows(rows, 'video_id'):
            raise ValueError('Source list changed. Use a new DOWNLOAD_RUN instead of mixing batches.')
    else:
        write_rows(snapshot, rows)
    progress = output / 'download_results.csv'
    previous = unique_rows(read_rows(progress), 'video_id') if progress.exists() else {}
    if set(previous) - {row['video_id'] for row in rows}:
        raise ValueError('Existing download results belong to different sources; use a new DOWNLOAD_RUN')
    results = [dict(row, filename='', file_path='', status='pending', error='') for row in rows]
    for result in results:
        old = previous.get(result['video_id'], {})
        result.update({key: old.get(key, result[key]) for key in ('filename', 'file_path', 'status', 'error')})
    attempted = 0
    for result in results:
        row = {key: value for key, value in result.items() if key not in ('filename','file_path','status','error')}
        vid = row['video_id']
        path = output / (vid + '.mp4')
        if valid_media(path):
            print(f'[resume] Already complete: {vid}')
            result.update(filename=path.name, file_path=str(path.resolve()), status='downloaded', error='')
            save_progress(progress, results)
            continue
        if args.limit and attempted >= args.limit:
            result.update(filename='', file_path='', status='pending', error='')
            continue
        attempted += 1
        try:
            with yt_dlp.YoutubeDL(dict(format='bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]',
                    outtmpl=str(output / (vid + '.%(ext)s')), merge_output_format='mp4',
                    js_runtimes={'node': {'path': node}}, continuedl=True,
                    noplaylist=True, quiet=False)) as downloader:
                downloader.download(['https://www.youtube.com/watch?v=' + vid])
            if not valid_media(path):
                raise ValueError('Expected complete MP4 with both video and audio')
            result.update(filename=path.name, file_path=str(path.resolve()), status='downloaded', error='')
        except Exception as exc:
            result.update(filename='', file_path='', status='failed', error=str(exc))
        save_progress(progress, results)
    save_progress(progress, results)
    if any(r['status'] == 'failed' for r in results):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
