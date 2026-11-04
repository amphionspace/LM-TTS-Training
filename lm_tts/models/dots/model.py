"""dots.tts continuous-latent TTS: Qwen2 LLM over text and patch embeddings, AR flow-matching DiT head.

The training path follows the official release: the LLM sees text tokens with each
<|audio_gen_span|> replaced by the PatchEncoder embedding of that 4-frame VAE patch; the hidden
state that predicts patch k conditions a DiT that denoises patch k while attending to the clean
history of patches 1..k-1. The VAE and the CAM++ speaker encoder are frozen and run offline, so the
batch carries VAE posteriors (mean, log std) and x-vectors."""

import json
from pathlib import Path

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.checkpoint import checkpoint
from transformers import Qwen2Config, Qwen2Model

from ...artifacts import file_hash
from ..assembly.common import load_prefix
from ..embedding import AddedRowsEmbedding
from ..lora import LoRASpec, apply_lora, lora_parameters, merge_lora
from .modules import DiT, PatchEncoder

CHUNK = 5  # One hidden state followed by the 4 latent frames of its patch.


class DotsLLM(nn.Module):
    """Keeps the release's `llm.model.*` names; the tied LM head is never used in training."""

    def __init__(self, config):
        super().__init__()
        self.model = Qwen2Model(config)


class DotsTTSModel(nn.Module):
    def __init__(self, config, llm_config, latent_mean, latent_var):
        super().__init__()
        if llm_config._attn_implementation not in {"flash_attention_2", "sdpa"}:
            raise ValueError("dots.tts requires flash_attention_2 or sdpa")
        self.config, self.llm_config = config, llm_config
        latent_dim, width = config["latent_dim"], config["DiT"]["hidden_size"]
        self.patch_size = config["patch_size"]
        self.cfg_droprate = config["cfg_droprate"]
        self.xvec_drop_rate = config["xvec_drop_rate"]
        self.llm = DotsLLM(llm_config)
        self.patch_encoder = PatchEncoder(
            latent_dim, llm_config.hidden_size, self.patch_size, config["PatchEncoder"]
        )
        self.hidden_proj = nn.Linear(llm_config.hidden_size, width)
        self.latent_proj = nn.Linear(latent_dim, width)
        self.coordinate_proj = nn.Linear(latent_dim, width)
        self.xvec_proj = nn.Sequential(
            nn.Linear(config["campplus_embedding_size"], width), nn.LayerNorm(width)
        )
        self.velocity_field_predictor = DiT(width, latent_dim, config["DiT"])
        self.eos_proj = nn.Sequential(
            nn.Linear(llm_config.hidden_size, llm_config.hidden_size),
            nn.SiLU(),
            nn.Linear(llm_config.hidden_size, 2),
        )
        # Fixed dataset statistics that put the DiT's history and targets on unit scale.
        self.register_buffer("latent_mean", torch.tensor(latent_mean), persistent=False)
        self.register_buffer("latent_std", torch.tensor(latent_var).sqrt(), persistent=False)
        self.checkpointing = False
        self.pretrained_rows = None  # Vocabulary size before assembly added the tag tokens.

    def sharding_modules(self):
        """Ordered FSDP units; module and parameter names remain checkpoint-stable."""
        return [
            *self.llm.model.layers,
            *self.patch_encoder.encoder.layers,
            *self.velocity_field_predictor.blocks,
        ]

    def enable_activation_checkpointing(self):
        self.llm.model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        self.checkpointing = True

    def adapt(self, settings):
        """Swap in the parameter-efficient layers a run trains. Exports repeat this before loading
        the checkpoint and then call merge_adapters, so the published layout never changes."""
        if settings.get("embedding_rows", "all") == "added":
            self.llm.model.embed_tokens = AddedRowsEmbedding(
                self.llm.model.embed_tokens, self.pretrained_rows
            )
        if "lora" in settings:
            apply_lora(self, LoRASpec(**settings["lora"]))

    def merge_adapters(self):
        merge_lora(self)
        if isinstance(self.llm.model.embed_tokens, AddedRowsEmbedding):
            self.llm.model.embed_tokens = self.llm.model.embed_tokens.merged()

    def parameter_groups(self, settings):
        """Pretrained weights fine-tune at lr; parameters this run introduces (LoRA adapters and
        added embedding rows) learn at adapter_lr without weight decay."""
        embedding = self.llm.model.embed_tokens
        new = lora_parameters(self)
        if isinstance(embedding, AddedRowsEmbedding):
            new.append(embedding.added)
        ids = {id(p) for p in new}
        pretrained = [p for p in self.parameters() if p.requires_grad and id(p) not in ids]
        groups = [{"params": pretrained, "lr": settings["lr"]}]
        if new:
            groups.append({"params": new, "lr": settings["adapter_lr"], "weight_decay": 0.0})
        return groups

    def _layers(self, layers, x, *args):
        for layer in layers:
            if self.checkpointing and self.training:
                x = checkpoint(layer, x, *args, use_reentrant=False)
            else:
                x = layer(x, *args)
        return x

    def encode_patches(self, latents, patch_lengths):
        """Raw VAE latents, packed (sum 4N, D), -> one LLM-width embedding per patch (sum N, H)."""
        rows = pad_sequence(latents.split((patch_lengths * self.patch_size).tolist()), True)
        embeddings = self.patch_encoder(rows.to(self.patch_encoder.in_proj.weight.dtype))
        valid = torch.arange(embeddings.shape[1], device=rows.device) < patch_lengths[:, None]
        return embeddings[valid]

    def hidden(self, batch, patch_embeddings):
        """LLM hidden states that predict each patch, packed (sum N, H)."""
        embeddings = self.llm.model.embed_tokens(batch["input_ids"])
        embeddings = embeddings.index_put(
            (batch["span_positions"],), patch_embeddings.to(embeddings.dtype)
        )
        lengths = batch["sequence_lengths"]
        offsets = torch.nn.functional.pad(lengths.cumsum(0), (1, 0))
        positions = torch.cat([torch.arange(n, device=lengths.device) for n in lengths.tolist()])
        if self.llm_config._attn_implementation == "flash_attention_2":
            cu_seqlens = offsets.to(torch.int32)
            hidden = self.llm.model(
                inputs_embeds=embeddings[None],
                position_ids=positions[None],
                cu_seq_lens_q=cu_seqlens,
                cu_seq_lens_k=cu_seqlens,
                max_length_q=int(lengths.max()),
                max_length_k=int(lengths.max()),
                use_cache=False,
            ).last_hidden_state[0]
            return hidden[batch["predictor_positions"]]
        rows = pad_sequence(embeddings.split(lengths.tolist()), True)
        mask = torch.arange(rows.shape[1], device=rows.device) < lengths[:, None]
        hidden = self.llm.model(inputs_embeds=rows, attention_mask=mask.long(), use_cache=False)
        hidden = hidden.last_hidden_state[mask]
        return hidden[batch["predictor_positions"]]

    def velocity(self, hidden, clean, noisy, t, condition, patch_lengths):
        """Predicted velocity for every noisy patch, packed (sum N, 4, D).

        Per utterance the DiT sees a causal prefix [h_k, clean_k] for k = 1..N and, for each
        patch, a generation chunk [h_k, noisy_k] that attends to itself and to the prefix chunks
        before k. Prefix and generation chunks of patch k share rotary positions. Utterances are
        right-padded; padded queries attend only to themselves."""
        p, n = self.patch_size, int(patch_lengths.max())
        valid = torch.arange(n, device=hidden.device) < patch_lengths[:, None]

        def chunks(latents, projection):
            frames = latents.new_zeros(len(patch_lengths), n, p, latents.shape[-1])
            frames[valid] = latents.view(-1, p, latents.shape[-1])
            values = hidden.new_zeros(len(patch_lengths), n, 1, hidden.shape[-1])
            values[valid] = hidden[:, None]
            return torch.cat([values, projection(frames.to(hidden.dtype))], dim=2).flatten(1, 2)

        sequence = torch.cat(
            [chunks(clean, self.latent_proj), chunks(noisy, self.coordinate_proj)], dim=1
        )
        index = torch.arange(2 * CHUNK * n, device=hidden.device)
        generation, local = index >= CHUNK * n, index % (CHUNK * n)
        chunk = local // CHUNK
        prefix_causal = ~generation[:, None] & ~generation[None] & (local[None] <= local[:, None])
        history = generation[:, None] & ~generation[None] & (local[None] < CHUNK * chunk[:, None])
        own = generation[:, None] & generation[None] & (chunk[None] == chunk[:, None])
        allowed = valid[:, chunk]
        mask = (prefix_causal | history | own)[None] & allowed[:, None, :] & allowed[:, :, None]
        mask = mask | (
            ~allowed[:, :, None] & torch.eye(len(index), dtype=torch.bool, device=index.device)
        )
        predictor = self.velocity_field_predictor
        c = predictor.time_embedder(t) + condition
        x = predictor.input_layer(sequence)
        x = self._layers(predictor.blocks, x, c, mask, local.expand(len(patch_lengths), -1))
        out = predictor.output_layer(x, c)[:, CHUNK * n :].unflatten(1, (n, CHUNK))
        return out[:, :, 1:][valid]

    def forward(self, batch):
        """Returns packed predictions and their flow-matching targets (sum N, 4, D) and EOS logits
        for every patch (sum N, 2). Noise, flow times and CFG dropout are drawn here."""
        lengths = batch["frame_lengths"]
        # Mixed precision casts floating inputs; posteriors and flow targets stay FP32 here.
        mean, log_std = batch["vae_mean"].float(), batch["vae_logstd"].float()
        latents = mean + torch.randn_like(mean) * log_std.exp()
        hidden = self.hidden(batch, self.encode_patches(latents, lengths))
        # The EOS head is trained on detached states, as in the release.
        eos_logits = self.eos_proj(hidden.detach())
        clean = (latents - self.latent_mean) / self.latent_std
        frames = lengths * self.patch_size
        t = torch.rand(len(lengths), device=clean.device)
        noise = torch.randn_like(clean)
        t_frames = t.repeat_interleave(frames)[:, None]
        noisy = t_frames * clean + (1 - t_frames) * noise
        drop_hidden = torch.zeros(len(lengths), dtype=torch.bool, device=clean.device)
        drop_speaker = drop_hidden
        if self.training:
            drop_hidden = torch.rand(len(lengths), device=clean.device) < self.cfg_droprate
            drop_speaker = torch.rand(len(lengths), device=clean.device) < self.xvec_drop_rate
        hidden = hidden.masked_fill(drop_hidden.repeat_interleave(lengths)[:, None], 0)
        speaker = self.xvec_proj(batch["xvectors"]).masked_fill(drop_speaker[:, None], 0)
        prediction = self.velocity(self.hidden_proj(hidden), clean, noisy, t, speaker, lengths)
        target = (clean - noise).view(-1, self.patch_size, clean.shape[-1])
        return {"velocity": prediction, "target": target, "eos_logits": eos_logits}

    @classmethod
    def from_config(cls, directory, attn_implementation="sdpa"):
        directory = Path(directory)
        config = json.loads((directory / "config.json").read_text())
        llm_config = Qwen2Config.from_dict(json.loads((directory / "llm_config.json").read_text()))
        llm_config._attn_implementation = attn_implementation
        stats = json.loads((directory / "latent_stats.json").read_text())
        return cls(config, llm_config, stats["mean"], stats["var"])

    @classmethod
    def from_assembled(
        cls,
        directory,
        load_weights=True,
        attn_implementation="flash_attention_2",
        use_speaker_embedding=None,
    ):
        directory = Path(directory)
        if not (directory / "ASSEMBLY_COMPLETE").is_file():
            raise ValueError("Require a completed dots.tts assembly")
        if use_speaker_embedding is False:
            raise ValueError("dots.tts always conditions the DiT on a speaker x-vector")
        report = json.loads((directory / "assembly_report.json").read_text())
        for name, expected in report["artifact_sha256"].items():
            if name.endswith(".json") or (load_weights and name == "model.safetensors"):
                if file_hash(directory / name) != expected:
                    raise ValueError(f"Assembled artifact changed: {name}")
        model = cls.from_config(directory, attn_implementation)
        model.pretrained_rows = report["text_vocabulary"]["source_token_count"]
        if load_weights:
            model.load_state_dict(load_prefix(directory, ""), strict=True)
        return model
