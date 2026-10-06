import os
import librosa
from transformers import pipeline

audio_classifier = pipeline("audio-classification", model="superb/wav2vec2-base-superb-er")

audio_path = os.path.join(os.path.dirname(__file__), "..", "samples", "test_audio.wav")

print(f"Loading file: {audio_path}")

audio_array, sampling_rate = librosa.load(audio_path, sr=16000)

result = audio_classifier({"raw": audio_array, "sampling_rate": sampling_rate})

print("\nResult of emotion detection:")
for item in result:
    print(f"- {item['label']}: {item['score']:.4f}")