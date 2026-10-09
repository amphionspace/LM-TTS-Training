"""Shared codec input protocol and first/residual prediction objective interface."""

import torch
from torch import nn

from .sampling import sample_token


class CodecTTSModel(nn.Module):
    def sharding_modules(self):
        """Ordered FSDP units; module and parameter names remain checkpoint-stable."""
        return [*self.talker.model.layers, *self.talker.code_predictor.model.layers]

    def enable_activation_checkpointing(self):
        for module in (self.talker.model, self.talker.code_predictor.model):
            module.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )

    def pretrained_modules(self, include_text=False):
        raise NotImplementedError

    def parameter_groups(self, settings):
        pretrained = {
            id(p)
            for module in self.pretrained_modules(
                include_text=settings.get("text_embedding_lr_group", "fresh") == "backbone"
            )
            for p in module.parameters()
        }
        backbone, fresh = [], []
        # Preserve traversal order so older Qwen optimizer checkpoints still load.
        for parameter in self.parameters():
            if parameter.requires_grad:
                (backbone if id(parameter) in pretrained else fresh).append(parameter)
        return [
            {"params": backbone, "lr": settings["backbone_lr"]},
            {"params": fresh, "lr": settings["lr"]},
        ]

    @property
    def predictor_attention(self):
        return self.config.code_predictor_config._attn_implementation

    @property
    def prefix_tokens(self):
        speaker = int(getattr(self.config, "lm_tts_use_speaker_embedding", True))
        if getattr(self.config, "lm_tts_input_protocol", "legacy_prefix") == "legacy_prefix":
            return 1 + speaker
        return len(self.config.lm_tts_role_ids) + 4 + speaker

    def input_embeddings(self, batch):
        from .protocol import build_inputs

        return build_inputs(self.talker, self.speaker_encoder, self.config, batch)

    def hidden(self, batch):
        inputs, audio_positions = self.input_embeddings(batch)
        outputs = self.talker.model(**inputs, use_cache=False)
        return outputs.last_hidden_state.flatten(0, 1)[audio_positions]

    def forward(self, batch, mode="predict", suppress_eos=False, generation=None):
        hidden = self.hidden(batch)
        last_positions = (batch["frame_lengths"] + 1).cumsum(0) - 1
        if mode == "next_frame":
            # Generation uses one unpadded sample on every rank.
            h = hidden[last_positions]
            logits = self.talker.codec_head(h).float()
            allowed = torch.cat(
                [logits[:, : self.code_size], logits[:, self.eos : self.eos + 1]], dim=-1
            )
            if suppress_eos:
                allowed[:, -1] = -torch.inf
            settings = generation or {}
            first = sample_token(
                allowed,
                history=batch["codes"][:, 0].unsqueeze(0) if len(h) == 1 else None,
                **{
                    key: settings[key]
                    for key in ("do_sample", "temperature", "top_k", "top_p", "repetition_penalty")
                    if key in settings
                },
            )
            stop = first == self.code_size
            codes = [first.clamp_max(self.code_size - 1)]
            inputs = [h.unsqueeze(1), self.talker.model.codec_embedding(codes[0]).unsqueeze(1)]
            predictor = self.talker.code_predictor
            for g in range(1, self.groups):
                out = predictor.model(
                    inputs_embeds=predictor.small_to_mtp_projection(torch.cat(inputs, dim=1)),
                    use_cache=False,
                )
                code = sample_token(
                    predictor.lm_head[g - 1](out.last_hidden_state[:, -1]),
                    do_sample=settings.get("subtalker_dosample", False),
                    temperature=settings.get("subtalker_temperature", 1.0),
                    top_k=settings.get("subtalker_top_k", 0),
                    top_p=settings.get("subtalker_top_p", 1.0),
                )
                codes.append(code)
                if g < self.groups - 1:
                    inputs.append(predictor.get_input_embeddings()[g - 1](code).unsqueeze(1))
            return torch.stack(codes, dim=-1), stop
        if mode != "predict":
            raise ValueError(mode)
        valid = torch.ones(hidden.shape[0], dtype=torch.bool, device=hidden.device)
        valid[last_positions] = False
        h, target = hidden[valid], batch["codes"]
        first_logits = self.talker.codec_head(hidden)
        predictor = self.talker.code_predictor
        inputs = [h.unsqueeze(1), self.talker.model.codec_embedding(target[:, 0]).unsqueeze(1)]
        for g in range(1, self.groups - 1):
            inputs.append(predictor.get_input_embeddings()[g - 1](target[:, g]).unsqueeze(1))
        out = predictor.model(
            inputs_embeds=predictor.small_to_mtp_projection(torch.cat(inputs, dim=1)),
            use_cache=False,
        )
        residual_logits = tuple(
            predictor.lm_head[g - 1](out.last_hidden_state[:, g]) for g in range(1, self.groups)
        )
        return {"first_logits": first_logits, "residual_logits": residual_logits}
