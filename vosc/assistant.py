import sys, queue, json, time, subprocess, os
import numpy as np
import sounddevice as sd
from openwakeword.model import Model as WakeModel
import vosk

# ─── Config ───────────────────────────────────────────────────────────────────
VOSK_MODEL_PATH = "/home/kiran/models/vosk-model-small-en-in-0.4"
RATE            = 16000
CHANNELS        = 1
BLOCKSIZE       = 1280
WAKE_THRESHOLD  = 0.5
COMMAND_TIMEOUT = 6.0
MIC_DEVICE      = 11            # your USB mic index

# ─── Intent router ────────────────────────────────────────────────────────────
def handle_command(text):
    text = text.lower().strip()
    print(f"[CMD] {text}")

    if not text:
        speak("I didn't catch that, please try again.")
        return

    if any(w in text for w in ["time", "clock"]):
        import datetime
        now = datetime.datetime.now().strftime("%I:%M %p")
        speak(f"The time is {now}")

    elif any(w in text for w in ["date", "today"]):
        import datetime
        today = datetime.datetime.now().strftime("%A, %B %d")
        speak(f"Today is {today}")

    elif any(w in text for w in ["hello", "hi", "hey"]):
        speak("Hello! How can I help you?")

    elif any(w in text for w in ["stop", "exit", "quit", "shutdown"]):
        speak("Goodbye!")
        sys.exit(0)

    else:
        speak(f"You said: {text}. I don't know how to handle that yet.")

# ─── TTS (placeholder until Piper is installed) ───────────────────────────────
def speak(text):
    print(f"[SPEAK] {text}")
    # Uncomment after Piper is installed:
    # subprocess.run([
    #     "/home/kiran/piper/piper",
    #     "--model", "/home/kiran/piper/en_US-lessac-medium.onnx",
    #     "--output_raw"
    # ], input=text.encode(), check=True)

# ─── Audio queue ──────────────────────────────────────────────────────────────
audio_q = queue.Queue()

def audio_callback(indata, frames, time_info, status):
    if status:
        print(f"[AUDIO] {status}", file=sys.stderr)
    audio_q.put(bytes(indata))

# ─── Listen for a command after wake word ─────────────────────────────────────
def listen_for_command(recognizer):
    print("[INFO] Listening for command...")
    recognizer.Reset()
    deadline = time.time() + COMMAND_TIMEOUT
    text = ""

    # Drain buffered audio from wake word detection
    while not audio_q.empty():
        audio_q.get_nowait()

    while time.time() < deadline:
        try:
            data = audio_q.get(timeout=0.5)
        except queue.Empty:
            continue
        if recognizer.AcceptWaveform(data):
            result = json.loads(recognizer.Result())
            chunk = result.get("text", "")
            if chunk:
                text += " " + chunk

    final = json.loads(recognizer.FinalResult())
    text += " " + final.get("text", "")
    return text.strip()

# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    print("[INFO] Loading Vosk model...")
    vosk_model = vosk.Model(VOSK_MODEL_PATH)
    recognizer = vosk.KaldiRecognizer(vosk_model, RATE)

    print("[INFO] Loading openWakeWord model...")
    wake_model = WakeModel(inference_framework="onnx")
    wake_names = list(wake_model.models.keys())
    print(f"[INFO] Available wake words: {wake_names}")
    print(f"[INFO] Using mic device index: {MIC_DEVICE}")
    print("[INFO] Say a wake word to activate...\n")

    with sd.RawInputStream(
        samplerate=RATE,
        blocksize=BLOCKSIZE,
        dtype="int16",
        channels=CHANNELS,
        callback=audio_callback,
        device=MIC_DEVICE
    ):
        last_wake_time = 0
        COOLDOWN = 3.0  # seconds before wake word can trigger again

        while True:
            try:
                data = audio_q.get(timeout=1.0)
            except queue.Empty:
                continue

            # Wake word detection
            audio_np = np.frombuffer(data, dtype=np.int16)
            scores = wake_model.predict(audio_np)
            triggered = {k: v for k, v in scores.items() if v > WAKE_THRESHOLD}

            now = time.time()
            if triggered and (now - last_wake_time) > COOLDOWN:
                best = max(triggered, key=triggered.get)
                print(f"[WAKE] '{best}' detected (score={triggered[best]:.2f})")
                last_wake_time = now

                command = listen_for_command(recognizer)
                handle_command(command)

                # Hard drain — clear anything buffered during command listen
                drained = 0
                while not audio_q.empty():
                    try:
                        audio_q.get_nowait()
                        drained += 1
                    except queue.Empty:
                        break
                if drained:
                    print(f"[INFO] Drained {drained} buffered chunks.")

                # Reset wake word internal state to clear lingering scores
                wake_model.reset()

                print("\n[INFO] Listening for wake word again...\n")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[INFO] Stopped.")