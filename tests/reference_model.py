"""Dense FP32 oracle for CPU checks and comparisons against production FA2."""

from qwen3_train.models.qwen import TTSModel
from qwen3_train.objectives.tts import tts_loss


class ReferenceTTSModel(TTSModel):
    def __init__(self, config, speaker_config=None):
        config._attn_implementation = "sdpa"
        config.code_predictor_config._attn_implementation = "sdpa"
        super().__init__(config, speaker_config)
        self.config._attn_implementation = "sdpa"
        self.config.code_predictor_config._attn_implementation = "sdpa"

    def forward(self, batch, mode="loss", suppress_eos=False, loss_reduction="token"):
        if mode == "next_frame":
            return super().forward(batch, mode=mode, suppress_eos=suppress_eos)
        return tts_loss(super().forward(batch), batch, self.eos, loss_reduction)

    def input_embeddings(self, batch):
        previous = self.config._attn_implementation
        self.config._attn_implementation = "flash_attention_2"
        try:
            return super().input_embeddings(batch)
        finally:
            self.config._attn_implementation = previous

    def hidden(self, batch):
        if self.config._attn_implementation == "flash_attention_2":
            return super().hidden(batch)
        inputs, audio_positions = self.input_embeddings(batch)
        positions = inputs["position_ids"][0]
        segments = (positions == 0).cumsum(0)
        inputs["attention_mask"] = (
            (segments[:, None] == segments[None, :]) & (positions[:, None] >= positions[None, :])
        )[None, None]
        outputs = self.talker.model(**inputs, use_cache=False)
        return outputs.last_hidden_state[0, audio_positions]
