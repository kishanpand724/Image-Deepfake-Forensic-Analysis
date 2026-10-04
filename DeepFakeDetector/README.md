# Full-Image DeepFakeDetector & Forensic Analysis Studio

## 1. Executive Summary & Root Cause Analysis

This repository resolves the core architectural limitations of legacy face-only deepfake detectors:

| Problem in Legacy Detector | Root Cause | Solution Implemented |
| :--- | :--- | :--- |
| **1. Face-only cropping (MTCNN dependency)** | Pipeline forced MTCNN detection. Non-faces, full-body fakes, background replacements, and diffusion scenes failed or got cropped to random facial artifacts. | Replaced MTCNN with `FullImagePreprocessor` which preserves full scene context (background, neck/hair boundaries, clothing, and scene geometry) using aspect-preserving letterboxing. |
| **2. Cropped face output only** | Grad-CAM and display logic were strictly bound to MTCNN bounding box coordinates. | Grad-CAM now operates across the complete convolutional feature map, inverse-maps letterbox padding, and overlays heatmaps on the full original image. |
| **3. Constant 50% confidence score bug** | Caused by: (a) uninitialized linear head bias (`sigmoid(0.0) = 0.50`), (b) silent state_dict key mismatches (e.g. `module.` prefixes from DataParallel), or (c) MTCNN fallback `except: return 0.50`. | Implemented strict state dictionary key verification in `load_detector_model`, calibrated logit scaling with temperature parameter, and explicit error handling rather than silent 50% fallbacks. |
| **4. Forensic report inconsistencies** | Verbal verdicts frequently contradicted confidence percentages or claimed authenticity based on the absence of facial bounding boxes. | Implemented `ForensicReportGenerator` enforcing mathematical consistency between probabilities, verdicts, localized anomaly bounding boxes, and an explicit **INCONCLUSIVE / UNCERTAIN** decision boundary (42%–58%). |

---

## 2. Checkpoint Compatibility & The "Face-Crop" Domain Shift

### ⚠️ Critical Machine Learning Reality
Simply removing MTCNN from inference does **not** instantly make a face-trained checkpoint optimal for full images:
- **Domain Shift**: Models trained exclusively on cropped faces (e.g. *FaceForensics++*, *Celeb-DF*) have learned convolutional filters tuned specifically to facial landmark distances, iris reflections, and mouth blending boundaries.
- When exposed to a full image containing trees, skies, architectural lines, or text, an unadapted face-crop model may produce false-positive activations on high-frequency foliage or compression edges.
- **The Solution**: 
  1. Our `FullImageForensicNet` integrates a **Spatial Attention** mechanism (`SpatialAttention`) to suppress uniform background noise and emphasize spliced boundary transitions.
  2. For production deployment, **Fine-Tuning on Full-Scene Datasets** (e.g., *GenImage*, *OpenForensics*, *DiffusionFaces*, and *In-the-Wild Deepfake Benchmark*) is required for optimal recall. The checkpoint loader automatically alerts the forensic operator if a face-crop checkpoint is detected.

---

## 3. Project Structure

```
DeepFakeDetector/
├── app.py                      # Gradio Web Interface (Full Image & Video tabs)
├── models/
│   └── detector_model.py       # FullImageForensicNet architecture & checkpoint loader
├── utils/
│   ├── preprocessor.py         # Full-image aspect-preserving letterbox preprocessor
│   ├── gradcam.py              # Full-frame Grad-CAM generator & colormap visualizer
│   ├── video_analyzer.py       # Temporal multi-frame video analyzer
│   └── forensics_report.py     # Consistent forensic dossier builder
├── test_detector.py            # Comprehensive automated test suite
├── requirements.txt            # Python dependencies
└── README.md                   # Complete architectural guide & reproduction instructions
```

---

## 4. Installation & Environment Setup

```bash
# 1. Clone repository & navigate to directory
cd DeepFakeDetector

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## 5. Running the Application

### Launch Gradio Web Interface
```bash
python app.py
```
*The Gradio app will start at `http://localhost:7860`.*

### Run Automated Test Suite
```bash
python -m unittest test_detector.py -v
# or with pytest:
pytest test_detector.py -v
```

---

## 6. API Usage Example

```python
from models.detector_model import load_detector_model
from utils.preprocessor import FullImagePreprocessor
from utils.gradcam import FullImageGradCAM
from utils.forensics_report import ForensicReportGenerator
from PIL import Image

# 1. Load model with diagnostics
model, diag = load_detector_model("checkpoints/full_image_forensicnet.pth")

# 2. Preprocess full image (aspect preserved, no crop)
preprocessor = FullImagePreprocessor(target_size=(256, 256))
tensor, orig_img, meta = preprocessor.process("path/to/suspect_image.jpg")

# 3. Calibrated prediction
prob_real, prob_fake, conf = model.predict_probability(tensor)

# 4. Generate full-scene Grad-CAM overlay
gradcam = FullImageGradCAM(model)
heatmap_2d = gradcam.generate_heatmap(tensor)
overlay, colored_heat, anomalies = gradcam.overlay_on_image(
    orig_img, heatmap_2d, meta["pad_box"], alpha=0.55, colormap="turbo"
)

# 5. Build consistent forensic dossier
report = ForensicReportGenerator.generate_report(
    prob_fake=prob_fake,
    model_diagnostics=diag,
    image_metadata=meta,
    suspicious_regions=anomalies
)

print("Verdict:", report["verdict"])
print("Calibrated Confidence:", report["confidence_percentage"], "%")
print("Anomaly Clusters:", len(report["suspicious_regions"]))
```
