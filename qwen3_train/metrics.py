"""Transparent English content metrics for the LJSpeech pilot."""
import re
import unicodedata


def normalize(text):
    text = unicodedata.normalize("NFKC", text).lower().replace("’", "'")
    return " ".join(re.findall(r"[^\W_]+(?:'[^\W_]+)*", text, flags=re.UNICODE))


def edits(reference, hypothesis):
    # (distance, substitutions, deletions, insertions), deterministic tie breaking.
    previous = [(j, 0, 0, j) for j in range(len(hypothesis) + 1)]
    for i, a in enumerate(reference, 1):
        current = [(i, 0, i, 0)]
        for j, b in enumerate(hypothesis, 1):
            if a == b:
                current.append(previous[j - 1])
            else:
                d, s, delete, insert = previous[j - 1]
                choices = [(d + 1, s + 1, delete, insert)]
                d, s, delete, insert = previous[j]
                choices.append((d + 1, s, delete + 1, insert))
                d, s, delete, insert = current[j - 1]
                choices.append((d + 1, s, delete, insert + 1))
                current.append(min(choices, key=lambda x: x[0]))
        previous = current
    return previous[-1]


def content_metrics(reference, hypothesis):
    ref, hyp = normalize(reference), normalize(hypothesis)
    words = ref.split()
    chars = list(ref.replace(" ", ""))
    if not words or not chars:
        raise ValueError("Reference text must contain words")
    wd, sub, delete, insert = edits(words, hyp.split())
    cd, *_ = edits(chars, list(hyp.replace(" ", "")))
    return {"wer": wd / len(words), "cer": cd / len(chars), "word_errors": wd,
            "reference_words": len(words), "character_errors": cd, "reference_characters": len(chars),
            "substitutions": sub, "deletions": delete, "insertions": insert,
            "normalized_reference": ref, "normalized_hypothesis": hyp}


class ASRScorer:
    def __init__(self, model="small.en"):
        from faster_whisper import WhisperModel
        self.model = WhisperModel(model, device="cpu", compute_type="int8", cpu_threads=4, num_workers=1)

    def score(self, audio, reference):
        segments, _ = self.model.transcribe(str(audio), language="en", beam_size=5,
                                            vad_filter=False, condition_on_previous_text=False)
        hypothesis = " ".join(segment.text.strip() for segment in segments)
        return {"transcript": hypothesis, **content_metrics(reference, hypothesis)}


def aggregate_content(results):
    if not results:
        raise ValueError("No scored samples")
    totals = {k: sum(r[k] for r in results) for k in [
        "word_errors", "reference_words", "character_errors", "reference_characters",
        "substitutions", "deletions", "insertions"]}
    return {**totals, "wer": totals["word_errors"] / totals["reference_words"],
            "cer": totals["character_errors"] / totals["reference_characters"], "samples": len(results)}
