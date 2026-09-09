"""Prepare a deterministic LJSpeech subset with the frozen official codec."""
import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from huggingface_hub import snapshot_download
from qwen_tts import Qwen3TTSTokenizer
from transformers import AutoTokenizer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", default="/ai_sds_wuzz/DATA_TTS/LJSpeech/LJSpeech-1.1")
    p.add_argument("--output", default="data/ljspeech-test")
    p.add_argument("--limit", type=int, default=64, help="0 uses all eligible utterances")
    p.add_argument("--val-count", type=int, default=8)
    p.add_argument("--max-seconds", type=float, default=12)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--tiny-text", action="store_true", help="Only for tiny-model integration checks")
    p.add_argument("--codec", default="pretrained/Qwen3-TTS-Tokenizer-12Hz")
    p.add_argument("--backbone", default="pretrained/Qwen3-0.6B")
    args = p.parse_args()
    if args.limit < 0 or args.val_count < 1 or args.max_seconds <= 0:
        p.error("Invalid subset size or duration")
    source, output = Path(args.source).resolve(), Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "codes").mkdir(exist_ok=True)
    metadata = source / "metadata.csv"
    entries = []
    lines = metadata.read_text().splitlines()
    random.Random(args.seed).shuffle(lines)
    print(f"Selecting up to {args.limit or len(lines)} examples from {len(lines)} records", flush=True)
    for line in lines:
        uid, raw, normalized = line.split("|", 2)
        wav = source / "wavs" / f"{uid}.wav"
        info = sf.info(wav)
        if 0 < info.duration <= args.max_seconds:
            entries.append((uid, normalized.strip() or raw.strip(), str(wav), info.duration))
            if args.limit and len(entries) >= args.limit:
                break
    if len(entries) <= args.val_count:
        raise ValueError("Need more examples than val-count")
    codec_path = str(Path(args.codec).resolve()) if Path(args.codec).is_dir() else snapshot_download(args.codec)
    text_path = None
    tokenizer = None
    if not args.tiny_text:
        text_path = str(Path(args.backbone).resolve()) if Path(args.backbone).is_dir() else snapshot_download(
            args.backbone, allow_patterns=["config.json", "tokenizer*", "vocab.json", "merges.txt", "special_tokens_map.json"])
        tokenizer = AutoTokenizer.from_pretrained(text_path, fix_mistral_regex=False)
    recipe = {**vars(args), "source": str(source), "output": str(output),
              "codec_path": codec_path, "backbone_path": text_path,
              "metadata_sha256": hashlib.sha256(metadata.read_bytes()).hexdigest()}
    recipe_file = output / "preparation.json"
    if recipe_file.exists() and json.loads(recipe_file.read_text()) != recipe:
        raise ValueError("Output was prepared with different settings; use a new output directory")
    recipe_file.write_text(json.dumps(recipe, indent=2))
    codec = Qwen3TTSTokenizer.from_pretrained(codec_path, device_map=args.device)
    codec.model.eval().requires_grad_(False)
    rows = []
    for i, (uid, text, wav, duration) in enumerate(entries):
        code_file = output / "codes" / f"{uid}.npz"
        if not code_file.exists():
            with torch.inference_mode():
                encoded = codec.encode(wav)
            codes = encoded.audio_codes[0].cpu().numpy().astype(np.uint16)
            if codes.ndim != 2 or codes.shape[1] != 16 or not len(codes) or codes.max() >= 2048:
                raise ValueError(f"Unexpected codec output {uid}: {codes.shape}")
            tmp = code_file.with_suffix(".tmp")
            with tmp.open("wb") as f:
                np.savez(f, codes=codes)
            tmp.replace(code_file)
        with np.load(code_file, allow_pickle=False) as f:
            frames = len(f["codes"])
        ids = [b % 250 + 1 for b in text.encode()] if args.tiny_text else tokenizer.encode(text, add_special_tokens=False)
        rows.append(dict(id=uid, text=text, text_ids=ids, audio=wav, duration=duration,
                         codes=f"codes/{uid}.npz", num_frames=frames,
                         codes_sha256=hashlib.sha256(code_file.read_bytes()).hexdigest(), speaker="ljspeech", language="en"))
        print(f"encoded {i + 1}/{len(entries)} {uid} frames={frames}", flush=True)
    for name, subset in [("val", rows[:args.val_count]), ("train", rows[args.val_count:])]:
        target = output / f"{name}.jsonl"
        tmp = target.with_suffix(".tmp")
        tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in subset))
        tmp.replace(target)
    with np.load(output / rows[0]["codes"], allow_pickle=False) as f:
        reference = torch.from_numpy(f["codes"].astype(np.int64))
    wavs, sr = codec.decode({"audio_codes": [reference]})
    sf.write(output / "codec-reconstruction.wav", wavs[0], sr)
    print(json.dumps({"train": len(rows) - args.val_count, "val": args.val_count,
                      "hours": sum(r["duration"] for r in rows) / 3600, "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
