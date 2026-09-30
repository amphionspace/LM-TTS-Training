import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from qwen3_train.evaluation.audio import load_audio
from qwen3_train.evaluation.metrics import aggregate_content, content_metrics
from qwen3_train.evaluation.quality import DNSMOS
from qwen3_train.evaluation.report import read_pairs


class EvaluationTests(unittest.TestCase):
    def test_corpus_errors_and_chinese_characters(self):
        rows = [
            content_metrics("one two three four", "one two three four"),
            content_metrics("five", "six"),
        ]
        self.assertAlmostEqual(aggregate_content(rows)["wer"], 1 / 5)
        self.assertAlmostEqual(content_metrics("你好，世界！", "你好世界")["cer"], 0)
        self.assertAlmostEqual(content_metrics("你好世界", "你好")["cer"], 0.5)

    def test_audio_resampling_channel_average_and_empty(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "stereo.wav"
            sf.write(
                path,
                np.stack([np.full(8000, 0.5), np.full(8000, -0.5)], axis=1),
                8000,
                subtype="FLOAT",
            )
            audio = load_audio(path)
            self.assertEqual(audio.shape, (16000,))
            np.testing.assert_array_equal(audio, 0)
            sf.write(path, np.empty(0), 16000)
            with self.assertRaisesRegex(ValueError, "Empty"):
                load_audio(path)

    def test_dnsmos_integer_windows_and_short_audio(self):
        class Session:
            def __init__(self):
                self.windows = []

            def run(self, _, inputs):
                self.windows.append(inputs["input_1"].copy())
                return [np.array([[3.0, 3.0, 3.0]])]

        scorer = DNSMOS.__new__(DNSMOS)
        scorer.session = Session()
        result = scorer.score(np.arange(176160, dtype=np.float32))
        self.assertEqual(len(scorer.session.windows), 3)
        for i, window in enumerate(scorer.session.windows):
            self.assertEqual(window.shape, (1, 144160))
            self.assertEqual(window[0, 0], 16000 * i)
        self.assertTrue(1 < result["dnsmos_ovrl"] < 5)
        scorer.session.windows.clear()
        scorer.score(np.ones(1000, dtype=np.float32))
        self.assertGreater(len(scorer.session.windows), 0)
        for audio in [np.empty(0), np.array([np.nan]), np.ones((2, 100))]:
            with self.assertRaises(ValueError):
                scorer.score(audio)

    def test_reference_and_generated_pairing(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "pairs.json"
            row = {
                "id": "x",
                "text": "hello",
                "language": "en",
                "audio": "generated.wav",
                "reference_audio": "ref.wav",
            }
            path.write_text(json.dumps([row]))
            self.assertEqual(read_pairs(path)[0]["audio"], str(path.parent / "generated.wav"))
            path.write_text(json.dumps([row, row]))
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                read_pairs(path)
            row["reference_audio"] = "generated.wav"
            path.write_text(json.dumps([row]))
            with self.assertRaisesRegex(ValueError, "different"):
                read_pairs(path)
