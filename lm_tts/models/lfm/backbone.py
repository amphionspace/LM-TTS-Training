"""Native LFM2 backbone with convolution boundaries isolated for packed utterances."""

import torch
from torch import nn
from torch.nn import functional as F
from transformers.models.lfm2.modeling_lfm2 import (
    Lfm2DecoderLayer,
    Lfm2Model,
    Lfm2PreTrainedModel,
    Lfm2RMSNorm,
    Lfm2RotaryEmbedding,
    Lfm2ShortConv,
)


class PackedShortConv(Lfm2ShortConv):
    def forward(self, hidden_states, position_ids):
        # Position IDs restart at each utterance. Mask every lag before summation,
        # including sequences shorter than the kernel; FA2 only isolates attention.
        b, c, x = self.in_proj(hidden_states).chunk(3, dim=-1)
        bx = b * x
        result = torch.zeros_like(bx, dtype=torch.float32)
        weights = self.conv.weight[:, 0].float()
        for lag in range(self.L_cache):
            if lag >= bx.shape[1]:
                break
            shifted = bx if lag == 0 else F.pad(bx[:, :-lag], (0, 0, lag, 0))
            valid = (position_ids >= lag).unsqueeze(-1)
            result = result + shifted.float() * weights[:, self.L_cache - 1 - lag] * valid
        if self.conv.bias is not None:
            result = result + self.conv.bias.float()
        return self.out_proj(c * result.to(c.dtype))


class PackedLfmLayer(Lfm2DecoderLayer):
    def __init__(self, config, layer_idx):
        super().__init__(config, layer_idx)
        if not self.is_attention_layer:
            self.conv = PackedShortConv(config, layer_idx)

    def forward(self, hidden_states, position_embeddings, position_ids=None, **kwargs):
        if self.is_attention_layer:
            return super().forward(
                hidden_states, position_embeddings, position_ids=position_ids, **kwargs
            )
        hidden_states = hidden_states + self.conv(self.operator_norm(hidden_states), position_ids)
        return hidden_states + self.feed_forward(self.ffn_norm(hidden_states))


class LfmCodecBackbone(Lfm2Model):
    def __init__(self, config, codec_vocab_size):
        Lfm2PreTrainedModel.__init__(self, config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.codec_embedding = nn.Embedding(codec_vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [PackedLfmLayer(config, i) for i in range(config.num_hidden_layers)]
        )
        self.rotary_emb = Lfm2RotaryEmbedding(config=config)
        self.pos_emb = Lfm2RotaryEmbedding(config)
        self.embedding_norm = Lfm2RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.gradient_checkpointing = False
        self.post_init()

    @property
    def text_embedding(self):
        # Expose the shared input protocol without registering the same weight twice.
        return self.embed_tokens

    def forward(self, *args, use_cache=False, past_key_values=None, **kwargs):
        if use_cache or past_key_values is not None:
            raise ValueError(
                "LFM codec training uses complete prefixes, without a generation cache"
            )
        return super().forward(*args, use_cache=False, **kwargs)
