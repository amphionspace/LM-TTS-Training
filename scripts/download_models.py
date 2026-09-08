"""Download pinned official checkpoints into this independent project."""
from pathlib import Path
from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parents[1]
MODELS = [
    ("Qwen/Qwen3-0.6B", "c1899de289a04d12100db370d81485cdf75e47ca"),
    ("Qwen/Qwen3-TTS-Tokenizer-12Hz", "7dd38ad4e9bad454aae9cd937d0cd577604fe229"),
]
for repo, revision in MODELS:
    target = ROOT / "pretrained" / repo.split("/")[-1]
    snapshot_download(repo, revision=revision, local_dir=target,
                      allow_patterns=["*.json", "*.safetensors", "merges.txt"], max_workers=4)
    (target / "REVISION").write_text(f"{repo}@{revision}\n")
    print(target, flush=True)
