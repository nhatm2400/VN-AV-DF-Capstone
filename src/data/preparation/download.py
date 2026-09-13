"""Download selected YouTube videos. License review is an optional separate step.

--dry_run validates manifests without network or media writes.
"""
import argparse
import re
from pathlib import Path
from src.data.preparation.manifest_io import read_rows, unique_rows, write_rows
from src.data.preparation.collect import normalize_sources


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
    parser.add_argument('--out_dir', required=True, help='new download run directory')
    parser.add_argument('--dry_run', action='store_true')
    args = parser.parse_args()
    videos = read_rows(args.videos)
    rows = (approved_sources(videos, read_rows(args.rights)) if args.rights
            else normalize_sources(videos))
    if args.dry_run:
        print(f'Validated {len(rows)} selected sources; no network/media writes')
        return
    import yt_dlp
    output = Path(args.out_dir)
    output.mkdir(parents=True, exist_ok=False)
    results = []
    for row in rows:
        vid = row['video_id']
        try:
            with yt_dlp.YoutubeDL(dict(format='bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]',
                    outtmpl=str(output / (vid + '.%(ext)s')), merge_output_format='mp4',
                    noplaylist=True, quiet=False)) as downloader:
                downloader.download(['https://www.youtube.com/watch?v=' + vid])
            path = output / (vid + '.mp4')
            if not path.is_file() or not path.stat().st_size:
                raise ValueError('Expected MP4 output missing')
            results.append(dict(row, filename=path.name, file_path=str(path.resolve()), status='downloaded', error=''))
        except Exception as exc:
            results.append(dict(row, filename='', file_path='', status='failed', error=str(exc)))
    write_rows(output / 'download_results.csv', results)
    if any(r['status'] == 'failed' for r in results):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
