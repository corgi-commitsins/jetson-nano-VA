#!/usr/bin/env python3
"""
train_wakeword.py
-----------------
Trains a wake word model using your recorded samples.
Run after record_wakeword.py.

Install:
    pip install torch torchaudio
    pip install "nanowakeword[train]"

Usage:
    python train_wakeword.py "hey jarvis"
    python train_wakeword.py
"""

import os
import sys
import subprocess

OUTPUT_BASE = os.path.join(os.path.expanduser("~"), "wakeword_training")


def print_banner(text):
    print("\n" + "─" * 60)
    print(f"  {text}")
    print("─" * 60)


def main():
    print_banner("Wake Word Trainer")

    if len(sys.argv) > 1:
        wake_word = " ".join(sys.argv[1:]).strip()
    else:
        wake_word = input("  Wake word/phrase to train: ").strip()

    if not wake_word:
        print("No wake word entered. Exiting.")
        sys.exit(1)

    safe_name   = wake_word.lower().replace(" ", "_")
    base_dir    = os.path.join(OUTPUT_BASE, safe_name)
    samples_dir = os.path.join(base_dir, "samples")
    model_dir   = os.path.join(base_dir, "model")
    os.makedirs(model_dir, exist_ok=True)

    if not os.path.isdir(samples_dir):
        print(f"\n  No samples found at: {samples_dir}")
        print("  Run record_wakeword.py first.\n")
        sys.exit(1)

    wav_files = [f for f in os.listdir(samples_dir) if f.endswith(".wav")]
    print(f"\n  Wake word  : '{wake_word}'")
    print(f"  Samples    : {len(wav_files)} recordings in {samples_dir}")
    print(f"  Model dir  : {model_dir}")

    print_banner("Training phases")
    print(
        f"\n  1. Generate ~3000 synthetic TTS samples of '{wake_word}'\n"
        f"  2. Train base model on synthetic data\n"
        f"  3. Fine-tune with your {len(wav_files)} personal recordings\n"
        f"  4. Export to ONNX\n"
        f"\n  This may take 10–30 minutes (GPU if available, else CPU).\n"
    )

    cmd = [
        sys.executable, "-m", "nanowakeword.train",
        "--phrase", wake_word,
        "--output-dir", model_dir,
        "--positive-samples-dir", samples_dir,
        "--n-synthetic", "3000",
        "--epochs", "30",
        "--auto-config",
        "--export-onnx"
    ]

    print("  Command:\n  " + " ".join(cmd) + "\n")

    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError:
        print(
            "\n  nanowakeword not found. Install with:\n"
            '      pip install "nanowakeword[train]"\n'
            "      pip install torch torchaudio\n"
        )
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"\n  Training failed (exit code {e.returncode})")
        print(f"  Recordings are safe in: {samples_dir}")
        print(
            "\n  Common fixes:\n"
            "  - Make sure torch is installed: pip install torch torchaudio\n"
            '  - Reinstall trainer: pip install --upgrade "nanowakeword[train]"\n'
            "  - Check nanowakeword docs for updated CLI flags\n"
        )
        sys.exit(e.returncode)

    onnx_files = [f for f in os.listdir(model_dir) if f.endswith(".onnx")]

    if onnx_files:
        model_path = os.path.join(model_dir, onnx_files[0])
        print_banner("Done!")
        print(f"\n  Model: {model_path}\n")
        print("  Next steps:")
        print("  1. Copy to Jetson:")
        print(f'       scp "{model_path}" kiran@<JETSON_IP>:/home/kiran/models/{safe_name}.onnx')
        print("\n  2. Load in assistant.py:")
        print("       wake_model = WakeModel(")
        print(f'           wakeword_models=["/home/kiran/models/{safe_name}.onnx"],')
        print('           inference_framework="onnx"')
        print("       )")
        print("\n  3. Update ALLOWED_WAKE_WORDS:")
        print(f'       ALLOWED_WAKE_WORDS = {{"{safe_name}"}}\n')
    else:
        print(f"\n  Training finished. Check output folder:\n  {model_dir}\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nStopped.")