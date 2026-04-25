#!/usr/bin/env python3
"""
wake_word_trainer.py  —  Windows compatible
--------------------------------------------
Records personal voice samples for a custom wake word, then trains
an ONNX model using nanowakeword.

Install once:
    pip install sounddevice soundfile numpy torch torchaudio
    pip install "nanowakeword[train]"
"""

import os
import sys
import json
import threading
import subprocess
import numpy as np
import sounddevice as sd
import soundfile as sf

# ─── Config ───────────────────────────────────────────────────────────────────
SAMPLE_RATE    = 16000
CHANNELS       = 1
NUM_RECORDINGS = 15
OUTPUT_BASE    = os.path.join(os.path.expanduser("~"), "wakeword_training")
MIC_DEVICE     = None  # set to an int if auto-detection picks the wrong mic

# ─── Recording state ──────────────────────────────────────────────────────────
recording_chunks = []
recording_active = False


# ─── Helpers ──────────────────────────────────────────────────────────────────
def print_banner(text):
    print("\n" + "─" * 60)
    print(f"  {text}")
    print("─" * 60)


def list_input_devices():
    print("\nAvailable input devices:")
    devices = sd.query_devices()
    for i, d in enumerate(devices):
        if d["max_input_channels"] > 0:
            print(f"  [{i}] {d['name']}")
    print()


# ─── Audio callback ───────────────────────────────────────────────────────────
def audio_callback(indata, frames, time_info, status):
    global recording_chunks, recording_active
    if status:
        print(f"[AUDIO WARNING] {status}", file=sys.stderr)
    if recording_active:
        recording_chunks.append(indata.copy())


# ─── Recording function ───────────────────────────────────────────────────────
def record_until_enter(device=None):
    global recording_chunks, recording_active

    recording_chunks = []
    recording_active = False
    stop_event = threading.Event()

    def wait_for_enter():
        input()
        stop_event.set()

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype="int16",
        callback=audio_callback,
        device=device,
        latency="low"
    ):
        input("Press ENTER to start recording...")
        recording_active = True
        print("Recording... say the wake word, then press ENTER to stop.")

        t = threading.Thread(target=wait_for_enter, daemon=True)
        t.start()
        stop_event.wait()

        recording_active = False

    if not recording_chunks:
        return None

    audio = np.concatenate(recording_chunks, axis=0).flatten()
    return audio


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    print_banner("Custom Wake Word Trainer (Windows)")

    list_input_devices()

    try:
        default_input = sd.query_devices(kind="input")
        print(f"Using default mic: {default_input['name']}")
    except Exception:
        print("Could not detect default microphone automatically.")

    print("\nIf the wrong mic is used, edit MIC_DEVICE at the top of this file.\n")

    wake_word = input("Type your wake word/phrase: ").strip()
    if not wake_word:
        print("No wake word entered. Exiting.")
        sys.exit(1)

    safe_name = wake_word.lower().replace(" ", "_")
    base_dir = os.path.join(OUTPUT_BASE, safe_name)
    samples_dir = os.path.join(base_dir, "samples")
    model_dir = os.path.join(base_dir, "model")

    os.makedirs(samples_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    print(f"\nWake word  : {wake_word}")
    print(f"Samples dir: {samples_dir}")
    print(f"Model dir  : {model_dir}")

    print_banner(f"Recording {NUM_RECORDINGS} samples")
    print(
        "Tips:\n"
        "- Say the phrase naturally\n"
        "- Vary tone and speed slightly\n"
        "- Keep background noise low\n"
        "- Use the same room where possible\n"
    )

    recorded_files = []

    while len(recorded_files) < NUM_RECORDINGS:
        idx = len(recorded_files) + 1
        print(f"\nSample {idx}/{NUM_RECORDINGS}")
        print(f"Say: '{wake_word}'")

        audio = record_until_enter(device=MIC_DEVICE)

        if audio is None or len(audio) < int(SAMPLE_RATE * 0.25):
            print("Recording too short or empty. Please try again.")
            continue

        out_path = os.path.join(samples_dir, f"sample_{idx:02d}.wav")
        sf.write(out_path, audio, SAMPLE_RATE)
        duration = len(audio) / SAMPLE_RATE
        print(f"Saved: {out_path} ({duration:.2f} s)")
        recorded_files.append(out_path)

    print(f"\nAll {NUM_RECORDINGS} recordings captured successfully.")

    config = {
        "wake_word": wake_word,
        "safe_name": safe_name,
        "samples": recorded_files,
        "model_dir": model_dir
    }

    config_path = os.path.join(base_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print_banner("Starting training")
    print(
        f"Phase 1: Generate synthetic samples for '{wake_word}'\n"
        f"Phase 2: Train base model\n"
        f"Phase 3: Fine-tune with your {NUM_RECORDINGS} recordings\n"
        f"Phase 4: Export ONNX\n"
    )
    print("This may take 10-30 minutes depending on CPU/GPU.\n")

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

    print("Running command:")
    print(" ".join(cmd))
    print()

    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError:
        print(
            "\nnanowakeword not found.\n"
            "Install with:\n"
            '  pip install "nanowakeword[train]"\n'
            "  pip install torch torchaudio\n"
        )
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"\nTraining failed with exit code {e.returncode}")
        print(f"Your recordings are saved in: {samples_dir}")
        sys.exit(e.returncode)

    onnx_files = [f for f in os.listdir(model_dir) if f.endswith(".onnx")]

    if onnx_files:
        model_path = os.path.join(model_dir, onnx_files[0])
        print_banner("Training complete")
        print(f"Model saved at:\n{model_path}\n")
        print("Next:")
        print(f'  Copy to Jetson as: /home/kiran/models/{safe_name}.onnx')
        print("  Then load it in your assistant.py")
    else:
        print(f"\nTraining finished. Check this folder manually:\n{model_dir}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped.")