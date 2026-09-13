"""Selected-source input and download behavior; no YouTube requests in tests."""
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from src.data.preparation.collect import collect_sources, normalize_sources, video_id
from src.data.preparation.download import main as download_main
from src.data.preparation.manifest_io import read_rows, write_rows

ROOT = Path(__file__).resolve().parents[2]
VID = 'JEgxmwvo7YY'


class CollectDownloadTests(unittest.TestCase):
    def test_supported_urls_and_id(self):
        for url in [VID, f'https://www.youtube.com/watch?v={VID}&list=abc',
                    f'https://youtu.be/{VID}?t=10', f'https://www.youtube.com/shorts/{VID}',
                    f'https://www.youtube.com/live/{VID}']:
            with self.subTest(url=url):
                self.assertEqual(video_id(url), VID)

    def test_reject_wrong_host_mismatched_id_and_duplicates(self):
        for url in [f'https://example.com/watch?v={VID}',
                    f'https://youtube.com.example.com/watch?v={VID}', 'not-a-valid-video-id']:
            with self.assertRaises(ValueError):
                video_id(url)
        with self.assertRaisesRegex(ValueError, 'disagree'):
            normalize_sources([{'video_id': VID, 'url': 'https://youtu.be/abcdefghijk'}])
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            normalize_sources([{'video_id': VID}, {'url': f'https://youtu.be/{VID}'}])

    def test_playlist_expansion_deduplicates_and_keeps_provenance(self):
        url = 'https://www.youtube.com/playlist?list=PL_example'
        loader = Mock(return_value={'entries': [{'id': VID, 'title': 'Talk', 'channel': 'Channel'}]})
        rows = collect_sources([{'url': url, 'program_id': 'program'},
                                {'url': f'https://youtu.be/{VID}'}], loader)
        loader.assert_called_once_with(url)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['program_id'], 'program')
        self.assertEqual(rows[0]['playlist_url'], url)
        self.assertEqual(rows[0]['channel'], 'Channel')
        with self.assertRaisesRegex(ValueError, 'Conflicting'):
            collect_sources([{'url': url, 'program_id': 'program'},
                             {'video_id': VID, 'program_id': 'different'}], loader)

    def test_watch_link_does_not_expand_playlist_and_empty_playlist_fails(self):
        loader = Mock(return_value={'entries': []})
        rows = collect_sources([{'url': f'https://www.youtube.com/watch?v={VID}&list=abc'}], loader)
        loader.assert_not_called()
        self.assertEqual(rows[0]['video_id'], VID)
        with self.assertRaisesRegex(ValueError, 'no readable entries'):
            collect_sources([{'url': 'https://www.youtube.com/playlist?list=abc'}], loader)

    def test_download_dry_run_needs_no_rights_or_output_directory(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / 'videos.csv'
            write_rows(source, [{'url': f'https://youtu.be/{VID}'}])
            output = root / 'output'
            result = subprocess.run([sys.executable, '-m', 'src.data.preparation.download', '--videos', str(source),
                                     '--out_dir', str(output), '--dry_run'], cwd=ROOT,
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertFalse(output.exists())
            self.assertIn('Validated 1', result.stdout)

    def test_download_without_rights_records_success_and_failure(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / 'videos.csv'
            output = root / 'download'
            write_rows(source, [{'url': f'https://youtu.be/{VID}'}, {'video_id': 'abcdefghijk'}])
            client = Mock()
            def download(urls):
                if urls[0].endswith(VID):
                    (output / f'{VID}.mp4').write_bytes(b'mocked downloader output, not real media')
                else:
                    raise RuntimeError('mock network failure')
            client.download.side_effect = download
            factory = Mock()
            factory.return_value.__enter__ = Mock(return_value=client)
            factory.return_value.__exit__ = Mock(return_value=False)
            with patch('src.data.preparation.download.shutil.which', return_value='node'), patch(
                    'src.data.preparation.download.valid_media', side_effect=lambda path: path.is_file()), patch.dict(sys.modules, {'yt_dlp': Mock(YoutubeDL=factory)}), patch.object(
                    sys, 'argv', ['download', '--videos', str(source), '--out_dir', str(output)]):
                with self.assertRaises(SystemExit) as error:
                    download_main()
            self.assertEqual(error.exception.code, 1)
            results = read_rows(output / 'download_results.csv')
            self.assertEqual([row['status'] for row in results], ['downloaded', 'failed'])
            self.assertEqual(results[1]['error'], 'mock network failure')
            self.assertNotIn('research_allowed', results[0])
            self.assertEqual(factory.call_args.args[0]['js_runtimes'], {'node': {'path': 'node'}})

            client.download.reset_mock()
            client.download.side_effect = lambda urls: (output / 'abcdefghijk.mp4').write_bytes(b'fixture')
            with patch('src.data.preparation.download.shutil.which', return_value='node'), patch(
                    'src.data.preparation.download.valid_media', side_effect=lambda path: path.is_file()), patch.dict(
                    sys.modules, {'yt_dlp': Mock(YoutubeDL=factory)}), patch.object(
                    sys, 'argv', ['download', '--videos', str(source), '--out_dir', str(output)]):
                download_main()
            client.download.assert_called_once_with(['https://www.youtube.com/watch?v=abcdefghijk'])
            self.assertEqual([row['status'] for row in read_rows(output / 'download_results.csv')],
                             ['downloaded', 'downloaded'])

            other = root / 'changed.csv'
            write_rows(other, [{'video_id': '12345678901'}])
            with patch('src.data.preparation.download.shutil.which', return_value='node'), patch.object(
                    sys, 'argv', ['download', '--videos', str(other), '--out_dir', str(output)]):
                with self.assertRaisesRegex(ValueError, 'Source list changed'):
                    download_main()


if __name__ == '__main__':
    unittest.main()
