"""Regression checks for the new source/split/media boundary, no downloaded models."""
import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from src.data.preparation.build_splits import split_real_rows, verify_splits, inherit_variant_split
from src.data.preparation.check_licenses import snapshot
from src.data.preparation.download import approved_sources
from src.data.preparation.manifest_io import write_rows, read_rows
from src.data.preparation.timeline_contract import build_timeline_contract, fixed_common_window
from src.tools.review.build_roi_preview import load_cropper

ROOT = Path(__file__).resolve().parents[2]


def real(cid, speaker, source, **extra):
    return dict(clip_id=cid, speaker_id=speaker, source_video=source,
                decision='keep', file_path=cid+'.mp4', **extra)


class ResearchDataContractTest(unittest.TestCase):
    def test_host_episode_and_reupload_links_cannot_cross_splits(self):
        rows = [real('a','host','ep1'), real('b','guest1','ep1'),
                real('c','host','ep2'), real('d','guest2','ep2'),
                real('e','guest3','reupload',canonical_source_id='ep1'),
                real('f','speaker4','ep4'), real('g','speaker5','ep5')]
        output = split_real_rows(rows)
        self.assertEqual(len({r['split'] for r in output[:5]}), 1)
        self.assertEqual({r['split'] for r in output}, {'train','val','test'})
        self.assertEqual(output, split_real_rows(list(reversed(rows))))

    def test_unknown_speaker_and_unreviewed_clips_fail(self):
        for row in (real('a','-1','v'), real('a','speaker','v')):
            if row['speaker_id'] != '-1':
                row['decision'] = 'uncertain'
            with self.assertRaises(ValueError):
                split_real_rows([row], (1,0,0))

    def test_one_connected_host_group_cannot_be_forced_into_three_splits(self):
        with self.assertRaisesRegex(ValueError, 'Too few'):
            split_real_rows([real('a','host','ep1'), real('b','host','ep2')])

    def test_speaker_links_and_split_verification(self):
        rows = [real('a','guest','v1',speaker_ids='host'), real('b','host','v2')]
        output = split_real_rows(rows, (1,0,0))
        output[1]['split'] = 'test'
        with self.assertRaisesRegex(ValueError, 'Leakage'):
            verify_splits(output)

    def test_single_speaker_mode_keeps_sources_reuploads_and_episodes_together(self):
        protocol = 'source_disjoint_single_speaker'
        rows = [real('a', 'host', 'v1'), real('b', 'host', 'v1'),
                real('c', 'host', 'copy', canonical_source_id='v1'),
                real('d', 'host', 'v2', program_id='p', episode_id='e'),
                real('e', 'host', 'v3', program_id='p', episode_id='e'),
                real('f', 'host', 'v4')]
        output = split_real_rows(rows, protocol=protocol)
        self.assertEqual({r['split'] for r in output}, {'train', 'val', 'test'})
        self.assertEqual(len({r['split'] for r in output[:3]}), 1)
        self.assertEqual(output[3]['split'], output[4]['split'])
        self.assertEqual(output, split_real_rows(list(reversed(rows)), protocol=protocol))
        variant = inherit_variant_split(dict(clip_id='fake', source_clip='a'), output)
        self.assertEqual(variant['split_protocol'], protocol)
        output[1]['split'] = next(r['split'] for r in output if r['split'] != output[0]['split'])
        with self.assertRaisesRegex(ValueError, 'Leakage'):
            verify_splits(output)

    def test_single_speaker_mode_rejects_multiple_speakers_and_mixed_protocols(self):
        protocol = 'source_disjoint_single_speaker'
        for extra in ({'speaker_id': 'guest'}, {'speaker_ids': 'guest'}):
            row = real('a', 'host', 'v1')
            row.update(extra)
            with self.assertRaisesRegex(ValueError, 'exactly one'):
                split_real_rows([row, real('b', 'host', 'v2')], (1, 0, 0), protocol=protocol)
        output = split_real_rows([real('a', 'host', 'v1'), real('b', 'host', 'v2')],
                                 (1, 0, 0), protocol=protocol)
        output[0]['split_protocol'] = 'speaker_source_disjoint'
        with self.assertRaisesRegex(ValueError, 'one split protocol'):
            verify_splits(output)

    def test_variants_inherit_split_and_block_cross_split_audio_and_generator(self):
        rows = split_real_rows([real(str(i),f's{i}',f'v{i}') for i in range(6)])
        train = next(r for r in rows if r['split']=='train')
        test = next(r for r in rows if r['split']=='test')
        variant = dict(clip_id='fake1', source_clip=train['clip_id'], generator='seen')
        self.assertEqual(inherit_variant_split(variant, rows)['split'], 'train')
        with self.assertRaisesRegex(ValueError, 'audio'):
            inherit_variant_split(dict(variant, audio_source_clip_id=test['clip_id']), rows)
        with self.assertRaisesRegex(ValueError, 'Held-out'):
            inherit_variant_split(dict(variant, generator='unseen'), rows, ('unseen',))

    def test_cc_snapshot_is_not_automatic_permission_and_missing_video_is_unknown(self):
        youtube = Mock()
        youtube.videos().list().execute.return_value = {'items':[{'id':'abcdefghijk','status':{'license':'creativeCommon'}}]}
        output = snapshot([{'video_id':'abcdefghijk'},{'video_id':'12345678901'}], youtube)
        self.assertEqual(output[1]['platform_license'], 'unknown')
        self.assertEqual(output[0]['research_allowed'], 'pending')
        with self.assertRaisesRegex(ValueError, 'permission'):
            approved_sources([{'video_id':'abcdefghijk'}], output)

    def test_manifest_refuses_overwrite_and_cropper_has_no_model_import_dependency(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'rows.csv'
            write_rows(path, [{'clip_id':'a'}])
            with self.assertRaises(FileExistsError):
                write_rows(path, [{'clip_id':'b'}])
            self.assertEqual(read_rows(path), [{'clip_id':'a'}])
        self.assertTrue(callable(load_cropper().detect_and_crop))

    def test_common_window_stays_inside_both_modalities(self):
        timeline = build_timeline_contract(5, 5, audio_valid=(0.2,4.9), visual_valid=(0.1,4.8))
        start, end = fixed_common_window(timeline, 4, position=1)
        self.assertAlmostEqual(start, 0.8)
        self.assertAlmostEqual(end, 4.8)

    def test_compression_preserves_provenance_audio_and_failure_accounting(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root/'source.mp4'
            subprocess.run(['ffmpeg','-v','error','-y','-f','lavfi','-i','testsrc2=s=96x96:r=25:d=1',
                            '-f','lavfi','-i','sine=frequency=440:sample_rate=16000:duration=1',
                            '-c:v','libx264','-pix_fmt','yuv420p','-c:a','aac','-shortest',str(source)], check=True)
            base = dict(file_path=str(source), label='0', split='train', group_id='g1', speaker_id='s1', source_video='v1')
            # Label=1 only exercises plumbing; this copied test pattern is NOT research fake data.
            rows = [dict(base,clip_id='real'), dict(base,clip_id='fake',label='1',source_clip='real',generator='fixture')]
            manifest = root/'masters.csv'
            write_rows(manifest, rows)
            command = [sys.executable,'-m','src.data.preparation.compress','--input_csv',str(manifest),'--out_dir',str(root/'encoded'),'--crfs','18,23']
            run = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stdout+run.stderr)
            variants = read_rows(root/'encoded/variants.csv')
            self.assertEqual(len(variants), 4)
            self.assertEqual({r['group_id'] for r in variants}, {'g1'})
            audios = [subprocess.check_output(['ffmpeg','-v','error','-i',r['file_path'],'-f','s16le','-acodec','pcm_s16le','-']) for r in variants]
            self.assertTrue(all(audio == audios[0] for audio in audios))
            self.assertNotEqual(subprocess.run(command,cwd=ROOT,capture_output=True).returncode, 0)
            bad = root/'missing.csv'
            write_rows(bad,[dict(base,clip_id='missing',file_path=str(root/'absent.mp4'))])
            command[command.index(str(manifest))] = str(bad)
            command[command.index(str(root/'encoded'))] = str(root/'failed')
            self.assertNotEqual(subprocess.run(command,cwd=ROOT,capture_output=True).returncode, 0)
            self.assertEqual(json.loads((root/'failed/summary.json').read_text())['failed_variants'], 2)


if __name__ == '__main__':
    unittest.main()
