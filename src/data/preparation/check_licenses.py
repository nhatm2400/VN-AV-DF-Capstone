"""Snapshot YouTube license metadata; does NOT approve research/release rights."""
import argparse
import os
from datetime import datetime, timezone
from src.data.preparation.manifest_io import read_rows, unique_rows, write_rows


def snapshot(rows, youtube):
    ids = list(unique_rows(rows, 'video_id'))
    found = {}
    for start in range(0, len(ids), 50):
        response = youtube.videos().list(part='status', id=','.join(ids[start:start+50])).execute()
        for item in response.get('items', []):
            found[item['id']] = item.get('status', {}).get('license', 'unknown')
    checked = datetime.now(timezone.utc).isoformat()
    return [dict(video_id=vid, platform_license=found.get(vid, 'unknown'), checked_at=checked,
                 research_allowed='pending', acquisition_allowed='pending',
                 manipulation_allowed='pending', redistribution_allowed='pending', evidence_ref='') for vid in ids]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--videos', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    from dotenv import load_dotenv
    from googleapiclient.discovery import build
    load_dotenv()
    key = os.environ.get('YOUTUBE_API_KEY')
    if not key:
        raise ValueError('Set YOUTUBE_API_KEY locally; never put it in a manifest')
    youtube = build('youtube', 'v3', developerKey=key)
    write_rows(args.out, snapshot(read_rows(args.videos), youtube))


if __name__ == '__main__':
    main()
