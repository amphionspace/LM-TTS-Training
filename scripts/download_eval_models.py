"""Fetch immutable official evaluation weights into the ignored assets directory."""

import argparse
import urllib.request
from pathlib import Path

from huggingface_hub import snapshot_download

MODELS = [
    (
        "microsoft/wavlm-base-plus-sv",
        "feb593a6c23c1cc3d9510425c29b0a14d2b07b1e",
        "wavlm-base-plus-sv",
        ["*.json", "pytorch_model.bin"],
    ),
    (
        "Systran/faster-whisper-small",
        "536b0662742c02347bc0e980a01041f333bce120",
        "faster-whisper-small",
        ["*.json", "model.bin", "vocabulary.txt"],
    ),
]
DNS_REVISION = "591184a9fcb2cbdec02520fed81a32bbbf9d73ff"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=Path, default=Path(__file__).resolve().parents[1] / "assets/evaluation"
    )
    args = parser.parse_args()
    for repo, revision, name, patterns in MODELS:
        target = args.output / name
        snapshot_download(
            repo, revision=revision, local_dir=target, allow_patterns=patterns, max_workers=4
        )
        (target / "REVISION").write_text(f"{repo}@{revision}\n")
    target = args.output / "DNSMOS"
    target.mkdir(parents=True, exist_ok=True)
    url = f"https://raw.githubusercontent.com/microsoft/DNS-Challenge/{DNS_REVISION}/DNSMOS/DNSMOS/sig_bak_ovr.onnx"
    temporary = target / "sig_bak_ovr.onnx.part"
    urllib.request.urlretrieve(url, temporary)
    temporary.replace(target / "sig_bak_ovr.onnx")
    (target / "REVISION").write_text(f"microsoft/DNS-Challenge@{DNS_REVISION}\n")


if __name__ == "__main__":
    main()
