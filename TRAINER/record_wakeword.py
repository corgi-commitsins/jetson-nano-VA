#!/usr/bin/env python3
"""
record_wakeword.py
------------------
Records personal voice samples for a custom wake word.
Run this first, then run train_wakeword.py.

Install:
    pip install sounddevice soundfile numpy
"""

import os
import sys
import json
import threading
import numpy as np
import sounddevice as sd
import soundfile as sf

# ─── Config ───────────────────────────────────────────────────────────────────
SAMPLE_RATE    = 16000
CHANNELS       = 1
NUM_RECORDINGS = 15
OUTPUT_BASE    = os.path.join(os.path.expanduser("~"), "wakeword_training")
MIC_DEVICE     = 44  # set to int if wrong mic is selected, e.g. MIC_DEVICE = 1

# ─── Recording state ──────────────────────────────────────────────────────────
recording_chunks = []
recording_active = False


def print_banner(text):
    print("\n" + "─" * 60)
    print(f"  {text}")
    print("─" * 60)


def audio_callback(indata, frames, time_info, status):
    global recording_chunks, recording_active
    if status:
        print(f"[AUDIO WARNING] {status}", file=sys.stderr)
    if recording_active:
        recording_chunks.append(indata.copy())


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
        input("   Press ENTER to start recording...")
        recording_active = True
        print("   Recording... press ENTER to stop.")

        t = threading.Thread(target=wait_for_enter, daemon=True)
        t.start()
        stop_event.wait()
        recording_active = False

    if not recording_chunks:
        return None

    return np.concatenate(recording_chunks, axis=0).flatten()


def main():
    print_banner("Wake Word Recorder")

    print("\n  Available input devices:")
    devices = sd.query_devices()
    for i, d in enumerate(devices):
        if d["max_input_channels"] > 0:
            print(f"    [{i}] {d['name']}")

    try:
        default_input = sd.query_devices(kind="input")
        print(f"\n  Default mic: {default_input['name']}")
    except Exception:
        pass

    print("\n  To change mic, edit MIC_DEVICE at the top of this file.\n")

    wake_word = input("  Type your wake word/phrase: ").strip()
    if not wake_word:
        print("No wake word entered. Exiting.")
        sys.exit(1)

    safe_name   = wake_word.lower().replace(" ", "_")
    base_dir    = os.path.join(OUTPUT_BASE, safe_name)
    samples_dir = os.path.join(base_dir, "samples")
    os.makedirs(samples_dir, exist_ok=True)

    print(f"\n  Wake word  : '{wake_word}'")
    print(f"  Saving to  : {samples_dir}")

    print_banner(f"Recording {NUM_RECORDINGS} samples")
    print(
        "\n  Tips:\n"
        "  - Say the phrase naturally, as you would to the device\n"
        "  - Vary your tone and speed slightly between takes\n"
        "  - Keep background noise low\n"
    )

    recorded_files = []

    while len(recorded_files) < NUM_RECORDINGS:
        idx = len(recorded_files) + 1
        print(f"\n  Sample {idx}/{NUM_RECORDINGS}  —  say: '{wake_word}'")

        audio = record_until_enter(device=MIC_DEVICE)

        if audio is None or len(audio) < int(SAMPLE_RATE * 0.25):
            print("  Too short or empty — try again.")
            continue

        out_path = os.path.join(samples_dir, f"sample_{idx:02d}.wav")
        sf.write(out_path, audio, SAMPLE_RATE)
        duration = len(audio) / SAMPLE_RATE
        print(f"  Saved: {out_path}  ({duration:.2f}s)")
        recorded_files.append(out_path)

    meta = {
        "wake_word": wake_word,
        "safe_name": safe_name,
        "samples_dir": samples_dir,
        "samples": recorded_files
    }
    meta_path = os.path.join(base_dir, "meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print_banner("Recording complete!")
    print(f"\n  {NUM_RECORDINGS} samples saved to:\n  {samples_dir}")
    print(f"\n  Now run:\n  python train_wakeword.py \"{wake_word}\"\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nStopped.")