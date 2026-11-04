"""Discrete-codec training data, objective and metrics for Qwen and LFM talkers."""

import hashlib

import torch
import torch.distributed as dist

from ..data.batch import collate
from ..data.build import tokenizer_identity, verify_tokenizer_compatibility
from ..data.reader import FeatureDataset
from ..evaluation.validation import validate
from ..models.protocol import language_codec_ids
from ..objectives.tts import loss_normalizers, tts_loss

STREAMING_KEYS = (
    "streaming_fraction",
    "streaming_max_text_ratio",
    "streaming_pause_probability",
    "streaming_max_pause_frames",
    "streaming_layout",
    "streaming_chunk",
)
LOSS_KEYS = ("first_sum", "residual_sum", "first_reduced_sum", "residual_reduced_sum")


def talker_lengths(batch, prefix_tokens, layout="overlay"):
    """Per-row Talker sequence lengths. Overlay streaming rows keep only bos of the text in the
    prefix; interleaved rows keep all text but drop codec_bos."""
    text = batch["text_lengths"]
    if "streaming" in batch:
        if layout == "interleaved":
            return text + batch["frame_lengths"] + prefix_tokens - batch["streaming"].long()
        text = text.masked_fill(batch["streaming"], 1)
    return text + batch["frame_lengths"] + prefix_tokens


def verify_conditioning(model, dataset):
    tokenizer = tokenizer_identity(dataset.manifest["tokenizer"]["path"])
    if tokenizer["sha256"] != dataset.manifest["tokenizer"]["sha256"]:
        raise ValueError("Build tokenizer artifact changed")
    if not getattr(model.config, "lm_tts_use_speaker_embedding", True):
        return
    if not getattr(model.config, "lm_tts_freeze_speaker_encoder", False):
        raise ValueError("Cached embeddings require a frozen speaker encoder")
    state = model.speaker_encoder.state_dict()
    profiles = [b["speaker_profile"] for b in dataset.bindings]
    profiles += [b["alternate"]["speaker_profile"] for b in dataset.bindings if "alternate" in b]
    for profile in profiles:
        tensors = profile["model"]["tensors"]
        if set(tensors) != {"speaker_encoder." + k for k in state}:
            raise ValueError("Cached speaker profile has a different architecture")
        for name, tensor in state.items():
            expected = tensors["speaker_encoder." + name]
            dtype = getattr(torch, expected["dtype"].removeprefix("torch."))
            value = tensor.detach().cpu().to(dtype).contiguous()
            actual = hashlib.sha256(
                value.reshape(-1).view(torch.uint8).numpy().tobytes()
            ).hexdigest()
            if list(value.shape) != expected["shape"] or actual != expected["sha256"]:
                raise ValueError(f"Cached speaker profile weights differ: {name}")


class CodecTask:
    collate = staticmethod(collate)

    def __init__(self, config, full_config, assembled, rank, backend):
        self.config, self.full_config, self.assembled = config, full_config, assembled
        self.settings, self.backend = config["train"], backend
        settings = self.settings
        self.layout = settings.get("streaming_layout", "overlay")
        specials = (full_config["tts_bos_token_id"], full_config["tts_eos_token_id"])
        self.language = config["model"].get("language_conditioning", "auto")
        reader_options = {}
        if self.language == "codec_language_id":
            reader_options = dict(
                language_codec_ids=language_codec_ids(
                    full_config["talker_config"]["codec_language_id"]
                ),
                language_auto_fraction=settings.get("language_auto_fraction", 0.0),
            )
        if settings.get("streaming_fraction"):
            reader_options.update({key: settings[key] for key in STREAMING_KEYS if key in settings})

        def dataset(path):
            return FeatureDataset(
                path,
                specials,
                verify_integrity=rank == 0,
                use_speaker_embedding=config["model"].get("use_speaker_embedding", True),
                mask_reference=settings.get("mask_reference", False),
                text_vocab_size=full_config["talker_config"]["text_vocab_size"],
                **reader_options,
            )

        self.train_data = dataset(config["data"]["build"])
        if rank == 0:
            self.train_data.validate_budgets(
                settings["max_batch_frames"], settings["max_batch_tokens"]
            )
        verify_tokenizer_compatibility(assembled, self.train_data.manifest["tokenizer"])
        val_build = config["data"].get("val_build")
        self.val_data = dataset(val_build) if val_build else None

    def configure(self, model):
        settings, full_config = self.settings, self.full_config
        if {"lora", "embedding_rows", "adapter_lr"} & set(settings):
            raise ValueError(
                "lora, embedding_rows and adapter_lr are implemented for dots.tts only"
            )
        if self.language == "codec_language_id":
            if full_config.get("model_type") != "qwen3_tts":
                raise ValueError("Language conditioning is implemented for Qwen talkers only")
            model.config.lm_tts_language_conditioning = self.language
        if settings.get("streaming_fraction"):
            protocol = getattr(model.config, "lm_tts_input_protocol", "legacy_prefix")
            if full_config.get("model_type") != "qwen3_tts" or protocol != "qwen3_non_streaming":
                raise ValueError("Streaming follows the Qwen3 prefill; it needs a Qwen3 talker")
            model.config.lm_tts_input_layout = "mixed"
            if settings.get("streaming_layout") == "interleaved":
                model.config.lm_tts_streaming_layout = "interleaved"
                model.config.lm_tts_streaming_chunk = settings["streaming_chunk"]
        verify_conditioning(model, self.train_data)
        if self.val_data is not None:
            verify_tokenizer_compatibility(self.assembled, self.val_data.manifest["tokenizer"])
            verify_conditioning(model, self.val_data)

    def describe(self, model):
        return {
            "predictor_attention": model.predictor_attention,
            "use_speaker_embedding": self.config["model"].get("use_speaker_embedding", True),
            "mask_reference": self.settings.get("mask_reference", False),
        }

    def begin_step(self, model, batches, device, split):
        """Global loss normalizers and token counts of one optimizer update."""
        settings = self.settings
        self.denominators = sum(
            loss_normalizers(
                b.get("supervised_frame_lengths", b["frame_lengths"]),
                settings["loss_reduction"],
            )
            for b in batches
        ).to(device)
        dist.all_reduce(self.denominators)
        self.counts = torch.tensor(
            [
                sum(
                    b.get("supervised_frame_lengths", b["frame_lengths"]).sum().item()
                    + len(b["frame_lengths"])
                    for b in batches
                ),
                sum(
                    b.get("supervised_frame_lengths", b["frame_lengths"]).sum().item() * 15
                    for b in batches
                ),
                sum(
                    talker_lengths(b, model.prefix_tokens, self.layout).sum().item()
                    for b in batches
                ),
                sum(
                    talker_lengths(b, model.prefix_tokens, self.layout).max().item()
                    * len(b["frame_lengths"])
                    for b in batches
                ),
            ],
            dtype=torch.float64,
            device=device,
        )
        dist.all_reduce(self.counts)
        # [0:4] every sample; [4:8] streaming samples; [8:12] their counts and normalizers.
        self.sums = torch.zeros(12, dtype=torch.float64, device=device)
        self.split = split

    def loss(self, model, batch, world):
        settings, sums = self.settings, self.sums
        predictions = model(batch)
        samples = batch.get("streaming") if self.split else None
        out = tts_loss(predictions, batch, model.eos, settings["loss_reduction"], samples)
        if samples is not None:
            part = out["selected"]
            lengths = batch.get("supervised_frame_lengths", batch["frame_lengths"])
            sums[4:8] += torch.stack([part[k] for k in LOSS_KEYS])
            sums[8] += part["first_count"]
            sums[9] += part["frame_count"] * 15
            sums[10:12] += loss_normalizers(lengths[samples], settings["loss_reduction"])
        del predictions
        loss = world * (
            out["first_reduced_sum"] / self.denominators[0]
            + settings["residual_weight"] * out["residual_reduced_sum"] / self.denominators[1]
        )
        sums[:4] += torch.stack([out[k].detach() for k in LOSS_KEYS])
        return loss

    def end_step(self):
        dist.all_reduce(self.sums)

    def metrics(self):
        settings, sums, counts, denominators = (
            self.settings,
            self.sums,
            self.counts,
            self.denominators,
        )
        metrics = {
            "first_ce": (sums[0] / counts[0]).item(),
            "residual_ce": (sums[1] / counts[1]).item(),
            "objective": (
                sums[2] / denominators[0] + settings["residual_weight"] * sums[3] / denominators[1]
            ).item(),
            "codec_tokens": counts[0].item() + counts[1].item(),
            "talker_tokens": counts[2].item(),
            "padding_efficiency": (counts[2] / counts[3]).item() if self.backend == "sdpa" else 1.0,
        }
        if settings.get("streaming_fraction"):
            # The same pooled statistics, split by each sample's input layout.
            parts = (
                ("streaming_", sums[4:8], sums[8:10], sums[10:12]),
                (
                    "non_streaming_",
                    sums[:4] - sums[4:8],
                    counts[:2] - sums[8:10],
                    denominators - sums[10:12],
                ),
            )
            for prefix, part, count, norm in parts:
                if count[0] > 0:
                    metrics[prefix + "first_ce"] = (part[0] / count[0]).item()
                    metrics[prefix + "residual_ce"] = (part[1] / count[1]).item()
                    metrics[prefix + "objective"] = (
                        part[2] / norm[0] + settings["residual_weight"] * part[3] / norm[1]
                    ).item()
        return metrics

    def validate(self, model, dataset, device):
        settings = self.settings
        return validate(
            model,
            dataset,
            self.config["eval"]["batch_size"],
            device,
            settings["loss_reduction"],
            settings["residual_weight"],
            loader_settings=settings,
        )
