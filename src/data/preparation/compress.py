"""Symmetric compression of real/fake masters into a new immutable run directory.

Use explicit CRFs. Encode audio with the same policy from each master at all CRFs.
Record failures rather than silently hiding missing variants; never chain encodes.
"""
import argparse
import hashlib
import json
from pathlib import Path
from src.data.preparation.compression_media import audio_target_samples, video_contract, compress
from src.data.preparation.manifest_io import read_rows, safe_id, unique_rows, write_rows
from src.data.preparation.timeline_contract import build_timeline_contract, validate_timeline_against_media


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--crfs", required=True)
    parser.add_argument("--preset", default="veryfast")
    args = parser.parse_args()
    rows = read_rows(args.input_csv)
    unique_rows(rows)
    crfs = [int(x) for x in args.crfs.split(',')]
    if not crfs or len(set(crfs)) != len(crfs) or any(not 0 <= c <= 51 for c in crfs):
        raise ValueError("CRFs must be unique integers in [0,51]")
    for row in rows:
        if str(row.get('label')) not in ('0','1') or row.get('split') not in ('train','val','test'):
            raise ValueError("Each master needs a binary label and locked split")
        if not row.get('group_id') or not row.get('source_video') or not row.get('speaker_id'):
            raise ValueError("Each master needs speaker/source/group provenance")
        if str(row['label']) == '1' and not row.get('source_clip'):
            raise ValueError("Fake master needs real source_clip")
        if row.get('compression_id') or row.get('crf'):
            raise ValueError("Use master clips, not already compressed variants")
    output = Path(args.out_dir)
    output.mkdir(parents=True, exist_ok=False)
    media = output / 'media'
    media.mkdir()
    config = dict(schema='lip_sync_compression_v1', crfs=crfs, encoder='libx264',
                  preset=args.preset, audio='aac_128k_16khz_mono',
                  input_sha256=hashlib.sha256(Path(args.input_csv).read_bytes()).hexdigest())
    config_id = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:16]
    (output / 'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    successes, failures = [], []
    for row in rows:
        cid = safe_id(row['clip_id'])
        for crf in crfs:
            try:
                source = row['file_path']
                samples = audio_target_samples(source)
                video = video_contract(source)
                if samples is None or video is None:
                    raise ValueError('Missing or invalid audio/video')
                if row.get('timeline_schema_version'):
                    timeline = validate_timeline_against_media(row, samples/16000, video['duration'])
                else:
                    timeline = build_timeline_contract(samples/16000, video['duration'],
                        manipulation_scope='none' if str(row['label']) == '0' else 'global')
                variant_id = f'{cid}_{config_id}_crf{crf}'
                destination = media / (variant_id + '.mp4')
                if not compress(source, str(destination), crf, args.preset, 'libx264', samples, video):
                    raise ValueError('Encode or post-encode media validation failed')
                parent = row.get('source_clip') or cid
                result = dict(row)
                result.update(timeline)
                result.update(clip_id=variant_id, file_path=str(destination.resolve()), parent_clip_id=cid,
                              source_clip=parent, compression_id=config_id, crf=crf)
                successes.append(result)
            except Exception as exc:
                failures.append(dict(clip_id=cid, crf=crf, error=str(exc)))
    if successes:
        write_rows(output / 'variants.csv', successes)
    if failures:
        write_rows(output / 'failures.csv', failures)
    summary = dict(master_clips=len(rows), requested_variants=len(rows)*len(crfs),
                   successful_variants=len(successes), failed_variants=len(failures))
    (output / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(summary)
    if failures:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
