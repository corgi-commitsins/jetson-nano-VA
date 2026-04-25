import sys, queue, json, time, subprocess, os
import numpy as np
import math
import sounddevice as sd
from openwakeword.model import Model as WakeModel
import vosk

# ─── Config ───────────────────────────────────────────────────────────────────
VOSK_MODEL_PATH  = "/home/kiran/models/vosk-model-small-en-in-0.4"
RATE             = 16000
CHANNELS         = 1
BLOCKSIZE        = 1280
WAKE_THRESHOLD   = 0.5
COMMAND_TIMEOUT  = 8.0
MIC_DEVICE       = 11

PIPER_BIN        = "/home/kiran/VA/jetson-nano-VA/piper-src/piper"
PIPER_MODEL      = "/home/kiran/VA/jetson-nano-VA/piper/en_US-lessac-medium.onnx"
PIPER_LIBS       = "/home/kiran/VA/jetson-nano-VA/piper-src/pi/lib"
PIPER_ESPEAK     = "/home/kiran/VA/jetson-nano-VA/piper-src/pi/share/espeak-ng-data"
PIPER_RATE       = 22050

# ─── TTS ──────────────────────────────────────────────────────────────────────
def speak(text):
    print(f"[SPEAK] {text}")
    try:
        env = os.environ.copy()
        env["LD_LIBRARY_PATH"] = PIPER_LIBS
        env["ESPEAK_DATA_PATH"] = PIPER_ESPEAK
        proc = subprocess.Popen(
            [PIPER_BIN, "--model", PIPER_MODEL, "--output_raw"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            env=env
        )
        raw_audio, _ = proc.communicate(input=text.encode())
        audio_np = np.frombuffer(raw_audio, dtype=np.int16).astype(np.float32) / 32768.0
        sd.play(audio_np, samplerate=PIPER_RATE)
        sd.wait()
    except Exception as e:
        print(f"[TTS ERROR] {e}")

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

# ─── Audio queue ──────────────────────────────────────────────────────────────
audio_q = queue.Queue()

def audio_callback(indata, frames, time_info, status):
    if status:
        print(f"[AUDIO] {status}", file=sys.stderr)
    audio_q.put(bytes(indata))

# beep after wake word to signal listening
def play_beep(freq=880, duration=0.15, volume=0.3):
    """Play a short beep to signal Jarvis is listening."""
    t = np.linspace(0, duration, int(PIPER_RATE * duration), False)
    tone = (np.sin(2 * np.pi * freq * t) * volume * 32767).astype(np.int16)
    sd.play(tone.astype(np.float32) / 32767, samplerate=PIPER_RATE)
    sd.wait()

# ─── Listen for a command after wake word ─────────────────────────────────────
def listen_for_command(recognizer):
    print("[INFO] Listening for command...")
    recognizer.Reset()

    # Drain only the wake word audio (fixed small amount)
    for _ in range(3):
        try:
            audio_q.get_nowait()
        except queue.Empty:
            break

    play_beep()  # signals user: "speak now"

    deadline = time.time() + COMMAND_TIMEOUT
    text = ""

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
        COOLDOWN = 3.0
        MAX_QUEUE_SIZE = 10

        while True:
            qsize = audio_q.qsize()
            if qsize > MAX_QUEUE_SIZE:
                while audio_q.qsize() > MAX_QUEUE_SIZE:
                    try:
                        audio_q.get_nowait()
                    except queue.Empty:
                        break

            try:
                data = audio_q.get(timeout=1.0)
            except queue.Empty:
                continue

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

                while not audio_q.empty():
                    try:
                        audio_q.get_nowait()
                    except queue.Empty:
                        break

                wake_model.reset()
                print("\n[INFO] Listening for wake word again...\n")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[INFO] Stopped.")