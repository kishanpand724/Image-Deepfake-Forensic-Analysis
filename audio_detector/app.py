import os
import uuid
import json
import subprocess
import sys
from fastapi import FastAPI, UploadFile, File

app = FastAPI(title="TrustLayer Audio API")

@app.get("/")
def home():
    return {
        "service": "TrustLayer Audio Module",
        "status": "running"
    }

@app.post("/analyze/audio")
async def analyze_audio(file: UploadFile = File(...)):
    # Create a temporary filename
    extension = os.path.splitext(file.filename)[1] or ".wav"
    temp_filename = f"temp_{uuid.uuid4().hex}{extension}"

    try:
        # Temporarily save uploaded audio
        with open(temp_filename, "wb") as buffer:
            content = await file.read()
            buffer.write(content)

        # Run audio analysis
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"

        result = subprocess.run(
            [sys.executable, "audio_analysis.py", temp_filename],
            capture_output=True,
            text=True,
            encoding="utf-8",
            env=env
        )

        if result.returncode != 0:
            return {
                "error": "Audio analysis failed",
                "details": result.stderr
            }

        # Extract JSON from analysis output
        output = result.stdout
        json_start = output.find("{")

        if json_start == -1:
            return {
                "error": "No analysis result generated",
                "details": output
            }

        json_output = output[json_start:]
        analysis_result = json.loads(json_output)
        return analysis_result

    finally:
        # Delete temporary audio
        if os.path.exists(temp_filename):
            os.remove(temp_filename)
            print(f"Deleted temporary file: {temp_filename}")
