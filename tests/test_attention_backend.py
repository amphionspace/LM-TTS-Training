import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import yaml

from qwen3_train.model import TTSModel, make_config
from qwen3_train.train import main


class AttentionBackendTests(unittest.TestCase):
    def test_disabled_backends_fail_before_loading_or_cuda_initialization(self):
        for backend in ("eager", "sdpa"):
            with self.subTest(backend=backend):
                with self.assertRaisesRegex(ValueError, "requires flash_attention_2"):
                    TTSModel.from_assembled("unused", attn_implementation=backend)
                for component in ("talker", "predictor"):
                    config = make_config(tiny=True)
                    target = config if component == "talker" else config.code_predictor_config
                    target._attn_implementation = backend
                    with self.assertRaisesRegex(ValueError, "requires flash_attention_2"):
                        TTSModel(config)
                with tempfile.TemporaryDirectory() as folder:
                    path = Path(folder) / "config.yaml"
                    path.write_text(yaml.safe_dump({"model": {"attn_implementation": backend}}))
                    with patch.object(sys, "argv", ["train", "--config", str(path)]):
                        with self.assertRaisesRegex(ValueError, "must be flash_attention_2"):
                            main()

    @unittest.skipUnless(torch.cuda.is_available(), "Requires CUDA and FA2")
    def test_runtime_backend_switch_cannot_bypass_the_restriction(self):
        model = TTSModel(make_config(tiny=True))
        for config in (model.config, model.config.code_predictor_config):
            for backend in ("eager", "sdpa"):
                with self.subTest(backend=backend):
                    config._attn_implementation = backend
                    with self.assertRaisesRegex(ValueError, "requires flash_attention_2"):
                        model.hidden({})
            config._attn_implementation = "flash_attention_2"
