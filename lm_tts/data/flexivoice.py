"""FlexiVoice-Instruct rows: speaker keys, style classes and instructions from upstream metadata.

Speakers and styles come only from evidence the upstream record carries: the relative wav_path
(game character, EARS / Expresso speaker and recording style, emotion-corpus actor) and the
emotion label list in `meta`. Nothing is inferred from the instruction text."""

import re

DEFAULT_INSTRUCTION = "Speak the following text."
# Ends in a newline: tokenizing prefix and text apart then equals tokenizing them together (the
# runtime tokenizes "Instruction: …\nText:\n<text>" as one string); "Text: " would not.
INSTRUCTION_TEMPLATE = "Instruction: {instruction}\nText:\n"

STYLES = (
    "neutral", "happy", "sad", "angry", "fear", "surprise", "disgust",
    "whisper", "loud", "fast", "slow", "high", "low", "other",
)  # fmt: skip
STYLE_ID = {name: i for i, name in enumerate(STYLES)}
_ALIASES = {
    "neutral": ("regular", "neutral", "default", "calm", "中立"),
    "happy": ("happy", "happy/joy", "amusement", "extasy", "excited", "laughing", "快乐", "开心"),
    "sad": (
        "sad", "saddened", "sadness", "disappointment", "guilt", "sympathetic",
        "sad-sympathetic", "sympathetic-sad", "难过", "伤心",
    ),
    "angry": ("anger", "angry", "生气"),
    "fear": ("fear", "fearful", "anxious", "scared", "害怕"),
    "surprise": ("surprise", "surprised", "amazement", "awe", "惊讶"),
    "disgust": ("disgust", "disgusted", "contempt", "厌恶"),
    "whisper": ("whisper",),
    "loud": ("loud", "projected"),
    "fast": ("fast",),
    "slow": ("slow",),
    "high": ("highpitch",),
    "low": ("lowpitch",),
}  # fmt: skip
_LABEL = {alias: style for style, aliases in _ALIASES.items() for alias in aliases}
_AROUSED = ("happy", "angry", "fear", "surprise", "loud")
# A reference in one of these styles conflicts with a target in the key style.
CONFLICTS = {
    "neutral": _AROUSED,
    "happy": ("sad", "angry", "disgust"),
    "sad": ("happy", "surprise", "loud"),
    "angry": ("happy", "neutral", "whisper"),
    "fear": ("happy", "angry"),
    "surprise": ("sad", "neutral"),
    "disgust": ("happy",),
    "whisper": ("loud",),
    "loud": ("whisper", "sad"),
    "fast": ("slow",),
    "slow": ("fast",),
    "high": ("low",),
    "low": ("high",),
    "other": tuple(s for s in STYLES if s not in ("neutral", "other")),
}


def speaker_key(config, wav_path):
    """Speaker key within FlexiVoice, or None where the path names no speaker (Emilia,
    nvspeech, MSP-PODCAST, MELD, m3ed)."""
    p = wav_path.split("/")
    fn = p[-1]
    if config in ("genshin", "starrail"):
        return f"{config}:{p[3]}:{p[4]}"  # language / character: one voice actor per language
    if config in ("ears", "l2"):
        return f"{config}:{p[3]}"
    if config == "expresso":
        if "_channel" in fn:  # conversational: <a>-<b>/…_channel1_… is speaker a
            a, b = p[5].split("-")
            return f"expresso:{a if '_channel1_' in fn else b}"
        return f"expresso:{p[5]}"
    if config == "kespeech":
        return f"kespeech:{p[4]}"
    if config == "voxceleb":
        return f"voxceleb:{p[6]}"
    if config == "childmandarin":
        return f"childmandarin:{fn.split('_')[0]}"
    if config == "WHSP":
        return f"WHSP:{p[4]}"
    if config == "debatts":
        return "debatts:" + "_".join(fn.split("_")[:2])
    if config == "emotion":
        src = p[2]
        if src == "Emotion_Speech_Dataset":
            return f"ESD:{p[3]}"
        if src == "CREMA-D":
            return f"CREMA-D:{fn.split('_')[0]}"
        if src == "RAVDESS":
            return f"RAVDESS:{p[3]}"
        if src == "EMNS":
            return "EMNS:single"
        if src == "IEMOCAP":  # Ses03M_impro05b_M029: session 3, utterance by the male speaker
            return f"IEMOCAP:{p[3]}-{fn.rsplit('_', 1)[1][0]}"
    return None


def _style_name(config, wav_path, meta):
    p = wav_path.split("/")
    if config == "ears":
        stem = re.sub(r"_\d+", "", p[-1].rsplit(".", 1)[0])
        parts = stem.split("_")
        return parts[1] if parts[0] == "emo" else parts[-1]
    if config == "expresso":
        return p[-3] if "/read/" in wav_path else p[-2]
    if config == "emotion" and isinstance(meta, list) and meta:
        return str(meta[0])
    return None


def style_class(config, wav_path, meta):
    """STYLE_ID of the recording's labelled style, or -1 where the source labels none."""
    name = _style_name(config, wav_path, meta)
    if name is None:
        return -1
    return STYLE_ID[_LABEL.get(name.strip().lower(), "other")]


def conflict_matrix():
    """bool[target_style, reference_style]: True where the reference style conflicts."""
    return [[ref in CONFLICTS[target] for ref in STYLES] for target in STYLES]


def instruction_prefix(instruction):
    instruction = (instruction or "").strip() or DEFAULT_INSTRUCTION
    return INSTRUCTION_TEMPLATE.format(instruction=instruction)
