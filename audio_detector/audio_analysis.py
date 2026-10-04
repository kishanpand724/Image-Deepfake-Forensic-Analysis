import os
import sys
import json
import torch
import librosa
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

# =========================
# CONFIGURATION
# =========================

MODEL_NAME = "Hemgg/Deepfake-audio-detection"
AUDIO_FILE = sys.argv[1] if len(sys.argv) > 1 else "sample.wav"

if not os.path.exists(AUDIO_FILE):
    print(json.dumps({"error": f"Audio file not found: {AUDIO_FILE}"}))
    sys.exit(1)

# =========================
# LOAD MODEL
# =========================

processor = AutoFeatureExtractor.from_pretrained(MODEL_NAME)
model = AutoModelForAudioClassification.from_pretrained(MODEL_NAME)

# =========================
# LOAD AUDIO
# =========================

# Load original audio for forensic analysis
original_audio, original_sample_rate = librosa.load(
    AUDIO_FILE,
    sr=None,
    mono=False
)

# Convert to mono + 16 kHz for the AI model
model_audio, model_sample_rate = librosa.load(
    AUDIO_FILE,
    sr=16000,
    mono=True
)

file_size = os.path.getsize(AUDIO_FILE)
original_channels = 1 if original_audio.ndim == 1 else original_audio.shape[0]
duration = len(model_audio) / 16000.0

# =========================
# AUDIO FORENSICS
# =========================

rms = librosa.feature.rms(y=model_audio)
average_rms = float(rms.mean())

zcr = librosa.feature.zero_crossing_rate(y=model_audio)
average_zcr = float(zcr.mean())

# =========================
# AI DETECTION
# =========================

inputs = processor(
    model_audio,
    sampling_rate=16000,
    return_tensors="pt"
)

with torch.no_grad():
    outputs = model(**inputs)

probabilities = torch.softmax(
    outputs.logits,
    dim=-1
)[0]

prediction_index = torch.argmax(probabilities).item()
prediction = model.config.id2label[prediction_index]

scores = {}
for i, probability in enumerate(probabilities):
    label = model.config.id2label[i]
    scores[label] = float(probability)

# =========================
# ASSESSMENT STATUS
# =========================

if "AIVoice" in scores and scores["AIVoice"] >= 0.80:
    status = "REQUIRES REVIEW — SYNTHETIC SIGNAL"
elif "HumanVoice" in scores and scores["HumanVoice"] >= 0.80:
    status = "NO STRONG SYNTHETIC SIGNAL DETECTED"
else:
    status = "INCONCLUSIVE"

result = {
    "modality": "audio",
    "file": os.path.basename(AUDIO_FILE),
    "prediction": prediction,
    "confidence": round(float(probabilities[prediction_index]), 4),
    "metadata": {
        "duration_seconds": round(float(duration), 2),
        "sample_rate": int(original_sample_rate),
        "channels": int(original_channels),
        "file_size_kb": round(file_size / 1024, 2)
    },
    "forensic_features": {
        "rms_energy": round(average_rms, 6),
        "zero_crossing_rate": round(average_zcr, 6)
    },
    "evidence": [
        "Model prediction generated via Hemgg/Deepfake-audio-detection",
        "Audio acoustic & technical characteristics extracted"
    ],
    "status": status
}

print(json.dumps(result, indent=4))
