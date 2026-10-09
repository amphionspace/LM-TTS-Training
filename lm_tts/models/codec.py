"""Shared codec input protocol and first/residual prediction objective interface."""

import torch
from torch import nn


class CodecTTSModel(nn.Module):
    def input_embeddings(self, batch):
        from .protocol import build_inputs

        return build_inputs(self.talker, self.speaker_encoder, self.config, batch)

    def hidden(self, batch):
        inputs, audio_positions = self.input_embeddings(batch)
        outputs = self.talker.model(**inputs, use_cache=False)
        return outputs.last_hidden_state.flatten(0, 1)[audio_positions]

    def forward(self, batch, mode="predict", suppress_eos=False):
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
            first = allowed.argmax(-1)
            stop = first == self.code_size
            codes = [first.clamp_max(self.code_size - 1)]
            inputs = [h.unsqueeze(1), self.talker.model.codec_embedding(codes[0]).unsqueeze(1)]
            predictor = self.talker.code_predictor
            for g in range(1, self.groups):
                out = predictor.model(
                    inputs_embeds=predictor.small_to_mtp_projection(torch.cat(inputs, dim=1)),
                    use_cache=False,
                )
                code = predictor.lm_head[g - 1](out.last_hidden_state[:, -1]).argmax(-1)
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
