"""
Run Qwen3-TTS on the existing prepared test texts.

Outputs are saved with a dedicated qwen3_* naming pattern so they are easy
to distinguish from previous Silero test files.
"""

import importlib.util
import argparse
import time
from pathlib import Path

import soundfile as sf
import torch
from qwen_tts import Qwen3TTSModel

from test_tts_comparison import TEST_TEXTS


MODEL_ID = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
OUTPUT_DIR = Path(__file__).resolve().parent / "test_results"
DEFAULT_VOICE_NAME = "Ryan"
DEFAULT_OUTPUT_PREFIX = "qwen3_test"
DEFAULT_SUMMARY_NAME = "QWEN3_SUMMARY.txt"


def _build_model() -> Qwen3TTSModel:
    use_cuda = torch.cuda.is_available()
    load_kwargs = {
        "device_map": "cuda:0" if use_cuda else "cpu",
        "dtype": torch.bfloat16 if use_cuda else torch.float32,
    }

    # Enable faster attention only when the dependency is available.
    if use_cuda and importlib.util.find_spec("flash_attn") is not None:
        load_kwargs["attn_implementation"] = "flash_attention_2"

    return Qwen3TTSModel.from_pretrained(MODEL_ID, **load_kwargs)


def _compact_text(text: str) -> str:
    return " ".join(text.split())


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Qwen3-TTS test set synthesis.")
    parser.add_argument(
        "--speaker",
        default=DEFAULT_VOICE_NAME,
        help="Qwen3 CustomVoice speaker name (e.g. Ryan, Vivian, Serena).",
    )
    parser.add_argument(
        "--output-prefix",
        default=DEFAULT_OUTPUT_PREFIX,
        help="Prefix for generated wav files.",
    )
    parser.add_argument(
        "--summary-name",
        default=DEFAULT_SUMMARY_NAME,
        help="Summary file name inside test_results directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    output_template = f"{args.output_prefix}" + "_{idx}_customvoice.wav"
    summary_path = OUTPUT_DIR / args.summary_name

    OUTPUT_DIR.mkdir(exist_ok=True)

    print("=" * 80)
    print("QWEN3-TTS TEST RUN")
    print("=" * 80)
    print(f"Model: {MODEL_ID}")
    print(f"Voice: {args.speaker}")
    print(f"Output dir: {OUTPUT_DIR}")
    print()

    model = _build_model()

    summary_lines = []
    synthesis_started = time.time()

    for idx, test_case in enumerate(TEST_TEXTS, start=1):
        test_name = test_case["name"]
        text = _compact_text(test_case["text"])

        print(f"[{idx}/{len(TEST_TEXTS)}] {test_name}")
        case_started = time.time()
        wavs, sample_rate = model.generate_custom_voice(
            text=text,
            language="Auto",
            speaker=args.speaker,
        )
        case_elapsed = time.time() - case_started

        output_file = OUTPUT_DIR / output_template.format(idx=idx)
        sf.write(output_file, wavs[0], sample_rate)

        duration_sec = len(wavs[0]) / sample_rate
        print(
            f"  saved: {output_file.name} | sr={sample_rate} | "
            f"duration={duration_sec:.2f}s | synth={case_elapsed:.2f}s"
        )

        summary_lines.append(
            f"{idx}. {test_name}\n"
            f"   file: {output_file.name}\n"
            f"   duration_sec: {duration_sec:.2f}\n"
            f"   synth_time_sec: {case_elapsed:.2f}\n"
        )

    total_elapsed = time.time() - synthesis_started
    header = [
        "QWEN3-TTS TEST SUMMARY",
        f"model: {MODEL_ID}",
        f"voice: {args.speaker}",
        f"total_synth_time_sec: {total_elapsed:.2f}",
        "",
    ]
    summary_path.write_text("\n".join(header + summary_lines), encoding="utf-8")

    print()
    print(f"Done. Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
