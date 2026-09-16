"""Dense FP32 oracle for CPU checks and comparisons against production FA2."""
from qwen3_train.model import TTSModel


class ReferenceTTSModel(TTSModel):
    def __init__(self, config, speaker_config=None):
        config._attn_implementation = "flash_attention_2"
        config.code_predictor_config._attn_implementation = "flash_attention_2"
        super().__init__(config, speaker_config)
        self.config._attn_implementation = "sdpa"
        self.config.code_predictor_config._attn_implementation = "sdpa"

    def hidden(self, batch):
        if self.config._attn_implementation == "flash_attention_2":
            return super().hidden(batch)
        inputs, audio_positions = self.input_embeddings(batch)
        positions = inputs['position_ids'][0]
        segments = (positions == 0).cumsum(0)
        inputs['attention_mask'] = ((segments[:, None] == segments[None, :]) &
                                    (positions[:, None] >= positions[None, :]))[None, None]
        outputs = self.talker.model(**inputs, use_cache=False)
        return outputs.last_hidden_state[0, audio_positions]
