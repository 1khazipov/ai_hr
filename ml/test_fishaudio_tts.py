"""
Run FishAudio (fish-speech-1.5) tests on the prepared text set.

This script reuses TEST_TEXTS from test_tts_comparison.py and saves outputs with
a dedicated file naming pattern to keep them distinct from Silero/Qwen files.
"""

import subprocess
import sys
import time
from pathlib import Path

import soundfile as sf
import torch

from test_tts_comparison import TEST_TEXTS


ML_DIR = Path(__file__).resolve().parent
FISH_REPO_DIR = ML_DIR / "fish-speech"
FISH_CKPT_DIR = FISH_REPO_DIR / "checkpoints" / "fish-speech-1.5"
FISH_DECODER_CKPT = FISH_CKPT_DIR / "firefly-gan-vq-fsq-8x1024-21hz-generator.pth"

OUTPUT_DIR = ML_DIR / "test_results"
TMP_DIR = OUTPUT_DIR / "fishaudio_tmp"
OUTPUT_TEMPLATE = "fishaudio_v15_test_{idx}.wav"
SUMMARY_PATH = OUTPUT_DIR / "FISHAUDIO_SUMMARY.txt"


def _compact_text(text: str) -> str:
    return " ".join(text.split())


def _run(cmd: list[str], cwd: Path) -> None:
    subprocess.run(cmd, cwd=str(cwd), check=True)


def _build_text2semantic_cmd(text: str, codes_dir: Path, device: str) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "fish_speech.models.text2semantic.inference",
        "--text",
        text,
        "--checkpoint-path",
        str(FISH_CKPT_DIR),
        "--device",
        device,
        "--output-dir",
        str(codes_dir),
        "--num-samples",
        "1",
        "--seed",
        "42",
    ]
    if device == "cuda":
        cmd.append("--half")
    return cmd


def _build_decoder_cmd(codes_path: Path, output_wav: Path, device: str) -> list[str]:
    return [
        sys.executable,
        "-m",
        "fish_speech.models.vqgan.inference",
        "-i",
        str(codes_path),
        "-o",
        str(output_wav),
        "--checkpoint-path",
        str(FISH_DECODER_CKPT),
        "--device",
        device,
    ]


def main() -> None:
    if not FISH_REPO_DIR.exists():
        raise FileNotFoundError(f"Fish repo not found: {FISH_REPO_DIR}")
    if not FISH_CKPT_DIR.exists():
        raise FileNotFoundError(f"Fish checkpoint dir not found: {FISH_CKPT_DIR}")
    if not FISH_DECODER_CKPT.exists():
        raise FileNotFoundError(f"Fish decoder checkpoint not found: {FISH_DECODER_CKPT}")

    OUTPUT_DIR.mkdir(exist_ok=True)
    TMP_DIR.mkdir(exist_ok=True)

    preferred_device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 80)
    print("FISHAUDIO TEST RUN")
    print("=" * 80)
    print(f"Fish repo: {FISH_REPO_DIR}")
    print(f"Checkpoint: {FISH_CKPT_DIR}")
    print(f"Preferred device: {preferred_device}")
    print(f"Output dir: {OUTPUT_DIR}")
    print()

    summary_lines: list[str] = []
    total_started = time.time()

    for idx, test_case in enumerate(TEST_TEXTS, start=1):
        case_name = test_case["name"]
        text = "<|speaker:0|>" + _compact_text(test_case["text"])
        output_wav = OUTPUT_DIR / OUTPUT_TEMPLATE.format(idx=idx)
        codes_dir = TMP_DIR / f"case_{idx}"
        codes_dir.mkdir(parents=True, exist_ok=True)
        codes_path = codes_dir / "codes_0.npy"

        print(f"[{idx}/{len(TEST_TEXTS)}] {case_name}")

        used_device = preferred_device
        semantic_elapsed = 0.0
        decode_elapsed = 0.0

        try:
            t0 = time.time()
            _run(_build_text2semantic_cmd(text, codes_dir, used_device), FISH_REPO_DIR)
            semantic_elapsed = time.time() - t0

            t1 = time.time()
            _run(_build_decoder_cmd(codes_path, output_wav, used_device), FISH_REPO_DIR)
            decode_elapsed = time.time() - t1
        except subprocess.CalledProcessError:
            if used_device != "cuda":
                raise
            print("  GPU run failed, retrying on CPU...")
            used_device = "cpu"

            t0 = time.time()
            _run(_build_text2semantic_cmd(text, codes_dir, used_device), FISH_REPO_DIR)
            semantic_elapsed = time.time() - t0

            t1 = time.time()
            _run(_build_decoder_cmd(codes_path, output_wav, used_device), FISH_REPO_DIR)
            decode_elapsed = time.time() - t1

        wav, sr = sf.read(output_wav)
        duration_sec = len(wav) / sr
        total_case = semantic_elapsed + decode_elapsed

        print(
            f"  saved: {output_wav.name} | sr={sr} | duration={duration_sec:.2f}s | "
            f"semantic={semantic_elapsed:.2f}s | decode={decode_elapsed:.2f}s | "
            f"total={total_case:.2f}s | device={used_device}"
        )

        summary_lines.append(
            f"{idx}. {case_name}\n"
            f"   file: {output_wav.name}\n"
            f"   sample_rate: {sr}\n"
            f"   duration_sec: {duration_sec:.2f}\n"
            f"   semantic_time_sec: {semantic_elapsed:.2f}\n"
            f"   decode_time_sec: {decode_elapsed:.2f}\n"
            f"   total_case_time_sec: {total_case:.2f}\n"
            f"   device: {used_device}\n"
        )

    total_elapsed = time.time() - total_started
    header = [
        "FISHAUDIO TEST SUMMARY",
        "model: fishaudio/fish-speech-1.5",
        f"checkpoint_dir: {FISH_CKPT_DIR}",
        f"total_synth_time_sec: {total_elapsed:.2f}",
        "",
    ]
    SUMMARY_PATH.write_text("\n".join(header + summary_lines), encoding="utf-8")

    print()
    print(f"Done. Summary saved to: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
