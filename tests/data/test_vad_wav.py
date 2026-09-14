import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np

from src.data.preparation.cut_clips import read_vad_wav


class VadWavTest(unittest.TestCase):
    def test_pcm16_samples_are_scaled_without_changing_timeline(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'audio.wav'
            samples = np.array([-32768, -16384, 0, 16384, 32767], dtype='<i2')
            with wave.open(str(path), 'wb') as handle:
                handle.setnchannels(1)
                handle.setsampwidth(2)
                handle.setframerate(16000)
                handle.writeframes(samples.tobytes())
            np.testing.assert_array_equal(read_vad_wav(path).numpy(), samples.astype(np.float32) / 32768)

    def test_stereo_wav_is_rejected_instead_of_flattening_channels(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'stereo.wav'
            with wave.open(str(path), 'wb') as handle:
                handle.setnchannels(2)
                handle.setsampwidth(2)
                handle.setframerate(16000)
                handle.writeframes(b'\0' * 16)
            with self.assertRaisesRegex(ValueError, 'mono'):
                read_vad_wav(path)


if __name__ == '__main__':
    unittest.main()
