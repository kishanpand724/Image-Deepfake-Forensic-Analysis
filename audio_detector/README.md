# TrustLayer Audio Deepfake Detection Module

Detects AI-generated synthetic voices, voice cloning, and audio deepfakes using acoustic forensics and the `Hemgg/Deepfake-audio-detection` classifier.

## Output Schema
```json
{
    "modality": "audio",
    "file": "sample.wav",
    "prediction": "HumanVoice",
    "confidence": 0.9618,
    "metadata": {
        "duration_seconds": 9.19,
        "sample_rate": 44100,
        "channels": 2,
        "file_size_kb": 4363.1
    },
    "forensic_features": {
        "rms_energy": 0.242003,
        "zero_crossing_rate": 0.217611
    },
    "evidence": [
        "Model prediction generated",
        "Audio technical characteristics extracted"
    ],
    "status": "NO STRONG SYNTHETIC SIGNAL DETECTED"
}
```

## Running Standalone via Python CLI
```bash
cd audio_detector
pip install -r requirements.txt
python audio_analysis.py your_audio_file.wav
```

## Running as FastAPI Microservice
```bash
cd audio_detector
uvicorn app:app --host 0.0.0.0 --port 8000
```
Then send POST requests to `http://localhost:8000/analyze/audio` with form-data `file`.
