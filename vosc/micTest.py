import sounddevice as sd
import vosk, json, sys, queue

q = queue.Queue()

def callback(indata, frames, time, status):
    if status:
        print(status, file=sys.stderr)
    q.put(bytes(indata))

# Download a small Vosk English model first (from a browser on your PC and scp,
# or directly on the Nano if you prefer). Example path:
#   /home/nvidia/models/vosk-model-small-en-us-0.15
model_path = "/home/kiran/models/vosk-model-small-en-us-0.15"

model = vosk.Model(model_path)
recognizer = vosk.KaldiRecognizer(model, 16000)

with sd.RawInputStream(samplerate=16000, blocksize=8000, dtype='int16',
                       channels=1, callback=callback, device=11):  # change 2 to your mic index
    print("Speak into the microphone (Ctrl+C to stop)…")
    try:
        while True:
            data = q.get()
            if recognizer.AcceptWaveform(data):
                res = json.loads(recognizer.Result())
                print("FINAL:", res.get("text", ""))
            else:
                res = json.loads(recognizer.PartialResult())
                if res.get("partial"):
                    print("PARTIAL:", res["partial"])
    except KeyboardInterrupt:
        print("\nStopping.")