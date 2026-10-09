"""Validate published reference intervals on the full codec target timeline."""

REFERENCE_COLUMNS = [
    "codec_feature_key",
    "speaker_reference_codec_start",
    "speaker_reference_codec_end",
    "speaker_reference_codec_feature_key",
    "speaker_reference_native_total_frames",
]


def is_reference(binding):
    return binding.get("merged_profile", {}).get("speaker_mode") == "codec_grid_reference"


def validate_reference(row):
    rate = row["codec_native_sample_rate"]
    total = row["codec_end_frame"]
    a, b = (row[f"speaker_reference_codec_{key}"] for key in ("start", "end"))
    start, end = row["speaker_start_frame"], row["speaker_end_frame"]
    if (
        type(a) is not int
        or type(b) is not int
        or rate <= 0
        or total <= 0
        or row["speaker_reference_codec_feature_key"] != row["codec_feature_key"]
        or row["speaker_reference_native_total_frames"] != total
        or row["speaker_native_sample_rate"] != rate
        or not 0 <= a < b <= total * 25 // (2 * rate)
        or b > row["codec_num_codec_frames"]
        or (start, end) != ((a * 2 * rate + 24) // 25, (b * 2 * rate + 24) // 25)
        or not max((rate + 1) // 2, (total + 9) // 10) <= end - start <= total // 2
    ):
        raise ValueError("Invalid reference-to-codec interval")
    return a, b
