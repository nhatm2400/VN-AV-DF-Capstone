"""STEP 04: Merge cut batches, keeping the source metadata needed for leakage checks."""
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data.preparation.settings import ROOT, SOURCES, REAL, MANIFESTS, SOURCE_GROUP


def main():
    if '--help' in sys.argv:
        print(__doc__)
        print('Reads all accepted_clips.csv under REAL and selected_videos.csv; creates clips.csv.')
        return
    from src.data.preparation.build_manifest import prep_tier
    from src.data.preparation.manifest_io import read_rows, unique_rows, write_rows
    os.chdir(ROOT)
    sources = unique_rows(read_rows(SOURCES / 'selected_videos.csv'), 'video_id')
    rows = prep_tier(SOURCE_GROUP, str(REAL / '**' / 'accepted_clips.csv'), str(REAL))
    unique_rows(rows)
    for row in rows:
        source = sources[row['source_video']]
        for key in ('url', 'title', 'channel', 'program_id', 'episode_id', 'canonical_source_id'):
            row[key] = source.get(key, '')
        row['tier'] = source.get('tier') or SOURCE_GROUP
    write_rows(MANIFESTS / 'clips.csv', rows)
    print(f'{len(rows)} clips -> {MANIFESTS / "clips.csv"}. Next: numbered scripts in src/tools/review/.')


if __name__ == '__main__':
    main()
