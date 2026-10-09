"""LFM backbone settings and codec protocol, without unused Qwen backbone fields."""

from dataclasses import asdict, dataclass, fields

from qwen_tts.core.models.configuration_qwen3_tts import Qwen3TTSTalkerCodePredictorConfig


@dataclass
class LfmTTSConfig:
    hidden_size: int
    text_hidden_size: int
    text_vocab_size: int
    vocab_size: int
    num_code_groups: int
    codec_bos_id: int
    codec_eos_token_id: int
    codec_pad_id: int
    codec_nothink_id: int
    codec_think_bos_id: int
    codec_think_eos_id: int
    lm_tts_lfm_config: dict
    lm_tts_role_ids: list
    lm_tts_pad_token_id: int
    code_predictor_config: Qwen3TTSTalkerCodePredictorConfig
    initializer_range: float = 0.02
    lm_tts_input_protocol: str = "qwen3_non_streaming"
    lm_tts_use_speaker_embedding: bool = False
    lm_tts_freeze_text_frontend: bool = False
    lm_tts_text_projection: str = "identity"
    _attn_implementation: str = "sdpa"

    @classmethod
    def from_dict(cls, raw):
        # Earlier artifacts used a Qwen Talker container. Its redundant backbone
        # fields are ignored; the original nested LFM config remains authoritative.
        names = {f.name for f in fields(cls)}
        values = {key: value for key, value in raw.items() if key in names}
        values["code_predictor_config"] = Qwen3TTSTalkerCodePredictorConfig.from_dict(
            raw["code_predictor_config"]
        )
        return cls(**values)

    def to_dict(self):
        values = asdict(self)
        values["code_predictor_config"] = self.code_predictor_config.to_dict()
        values.pop("_attn_implementation")
        return values
