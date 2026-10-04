"""
Full-Image DeepFakeDetector - Gradio Web Application.

Upgraded from legacy face-only MTCNN crop detection to full-image and full-frame video analysis.
Provides full-scene Grad-CAM heatmaps, calibrated REAL/FAKE predictions, and consistent forensic dossiers.
"""

import os
import sys
import logging
from typing import Tuple, Dict, Any, Optional
try:
    import numpy as np
    from PIL import Image
    import gradio as gr
    import torch
    DEPENDENCIES_LOADED = True
except ImportError as err:
    DEPENDENCIES_LOADED = False
    MISSING_ERR = str(err)

if not DEPENDENCIES_LOADED:
    print(f"[!] DeepFakeDetector missing dependency: {MISSING_ERR}")
    print("[*] Please run: pip install -r requirements.txt")
    sys.exit(0)

from models.detector_model import load_detector_model, FullImageForensicNet
from utils.preprocessor import FullImagePreprocessor
from utils.gradcam import FullImageGradCAM
from utils.video_analyzer import FullFrameVideoAnalyzer
from utils.forensics_report import ForensicReportGenerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DeepFakeDetector.App")

# Initialize core pipeline
CHECKPOINT_PATH = os.environ.get("CHECKPOINT_PATH", "checkpoints/full_image_forensicnet.pth")
device = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"

print(f"[*] Initializing Full-Image DeepFakeDetector on device: {device}")
model, diagnostics = load_detector_model(CHECKPOINT_PATH, device=device)
preprocessor = FullImagePreprocessor(target_size=(256, 256))
gradcam = FullImageGradCAM(model)
video_analyzer = FullFrameVideoAnalyzer(model, preprocessor, gradcam)


def process_image(
    image: Optional[Image.Image],
    overlay_alpha: float = 0.55,
    colormap: str = "turbo"
) -> Tuple[Optional[Image.Image], Optional[Image.Image], Dict[str, Any], str]:
    """
    Analyzes full image (background, boundary contours, and face),
    computes full-scene Grad-CAM heatmap, and builds consistent forensic report.
    """
    if image is None:
        return None, None, {}, "Error: No image uploaded. Please upload a valid image file."

    try:
        # 1. Full Image Preprocessing (aspect-preserving letterbox, NO MTCNN crop)
        tensor, original_pil, metadata = preprocessor.process(image)

        # 2. Prediction with calibrated logits
        prob_real, prob_fake, conf_level = model.predict_probability(tensor)

        # 3. Grad-CAM over the ENTIRE image
        heatmap_2d = gradcam.generate_heatmap(tensor, target_class=0)
        overlay, heat_colorized, suspicious_regions = gradcam.overlay_on_image(
            original_pil, heatmap_2d, metadata["pad_box"], alpha=overlay_alpha, colormap=colormap
        )

        # 4. Generate consistent forensic dossier
        report = ForensicReportGenerator.generate_report(
            prob_fake=prob_fake,
            model_diagnostics=diagnostics,
            image_metadata=metadata,
            suspicious_regions=suspicious_regions,
            is_video=False
        )

        # Output label dictionary for Gradio
        label_output = {
            "Real / Authentic": round(prob_real, 4),
            "Fake / Synthetic": round(prob_fake, 4)
        }

        # Formatted markdown summary
        summary_md = f"""
### Forensic Verdict: **{report['verdict']}**
* **Calibrated Confidence**: `{report['confidence_percentage']}%`
* **Fake Probability**: `{prob_fake * 100:.2f}%` | **Real Probability**: `{prob_real * 100:.2f}%`
* **Analysis Scope**: Full Image ({metadata['original_width']}x{metadata['original_height']} px) - Aspect Ratio Preserved
* **Localized Anomaly Hotspots**: `{len(suspicious_regions)} detected`
* **Checkpoint Status**: `{diagnostics['status']}` ({diagnostics['domain_compatibility']})

> **Forensic Assessment**: {report['verdict_description']}
        """

        return overlay, heat_colorized, label_output, summary_md

    except Exception as e:
        logger.exception("Inference error during full-image analysis")
        err_msg = f"Inference Error: {str(e)}"
        return None, None, {"Error": 1.0}, err_msg


def process_video(
    video_path: Optional[str]
) -> Tuple[Optional[Image.Image], Dict[str, Any], str]:
    """
    Analyzes multiple frames across video timeline with full-frame Grad-CAM.
    """
    if not video_path or not os.path.exists(video_path):
        return None, {}, "Error: No valid video file provided."

    try:
        results = video_analyzer.analyze_video(video_path)
        peak_overlay = results["peak_frame_overlay"]
        label_output = {
            "Real / Authentic": round(1.0 - results["mean_fake_probability"], 4),
            "Fake / Synthetic": round(results["mean_fake_probability"], 4)
        }

        summary_md = f"""
### Video Verdict: **{results['verdict']}**
* **Aggregated Confidence**: `{results['confidence_percentage']}%`
* **Mean Fake Probability**: `{results['mean_fake_probability'] * 100:.2f}%`
* **Frames Inspected**: `{results['total_frames_analyzed']}`
* **Peak Anomaly Frame**: Frame #{results['peak_frame_index']}
* **Temporal Jitter**: `{results['temporal_jitter']}` (Variance: `{results['variance']}`)
        """

        return peak_overlay, label_output, summary_md

    except Exception as e:
        logger.exception("Inference error during video analysis")
        return None, {"Error": 1.0}, f"Video Inference Error: {str(e)}"


def build_gradio_app():
    """Builds the upgraded Gradio Blocks UI."""
    if not GRADIO_AVAILABLE:
        print("[!] Gradio is not installed in current environment.")
        return None

    with gr.Blocks(title="Full-Image DeepFakeDetector & Forensics") as demo:
        gr.Markdown(
            """
            # Full-Image DeepFakeDetector & Visual Forensics
            **Full-Image Analysis | Artifact & Boundary Heatmaps | Calibrated Probabilities**
            
            *Fixes applied: Removed MTCNN face-crop restriction, letterboxed full aspect ratio, calibrated logits (fixing identical 50% outputs), and synchronized forensic reporting.*
            """
        )

        with gr.Tab("Full-Image Analysis"):
            with gr.Row():
                with gr.Column(scale=1):
                    input_img = gr.Image(type="pil", label="Upload Full Image (Portrait, Scene, or Meme)")
                    alpha_slider = gr.Slider(minimum=0.1, maximum=1.0, value=0.55, step=0.05, label="Grad-CAM Overlay Opacity")
                    colormap_choice = gr.Radio(choices=["turbo", "jet"], value="turbo", label="Heatmap Colormap")
                    btn_analyze = gr.Button("Analyze Full Image", variant="primary")

                with gr.Column(scale=1):
                    out_overlay = gr.Image(type="pil", label="Full Image + Grad-CAM Heatmap Overlay")
                    out_heatmap = gr.Image(type="pil", label="Extracted Heatmap")
                    out_labels = gr.Label(num_top_classes=2, label="Calibrated Prediction Probabilities")
                    out_summary = gr.Markdown(label="Forensic Report Summary")

            btn_analyze.click(
                fn=process_image,
                inputs=[input_img, alpha_slider, colormap_choice],
                outputs=[out_overlay, out_heatmap, out_labels, out_summary]
            )

        with gr.Tab("Full-Frame Video Analysis"):
            with gr.Row():
                with gr.Column():
                    input_vid = gr.Video(label="Upload Video File (MP4, AVI, MOV)")
                    btn_vid = gr.Button("Analyze Video Frames", variant="primary")
                with gr.Column():
                    out_vid_peak = gr.Image(type="pil", label="Peak Anomaly Frame (Full Frame + Heatmap)")
                    out_vid_labels = gr.Label(label="Aggregated Video Probabilities")
                    out_vid_summary = gr.Markdown(label="Video Forensic Breakdown")

            btn_vid.click(
                fn=process_video,
                inputs=[input_vid],
                outputs=[out_vid_peak, out_vid_labels, out_vid_summary]
            )

        with gr.Tab("Model & Checkpoint Diagnostics"):
            gr.Markdown(
                f"""
                ### Checkpoint Verification & Architecture Status
                * **Model Architecture**: `{diagnostics['model_type']}`
                * **Status**: `{diagnostics['status']}`
                * **Keys Matched**: `{diagnostics['keys_matched']}`
                * **Keys Missing**: `{diagnostics['keys_missing']}`
                * **Domain Compatibility**: `{diagnostics['domain_compatibility']}`
                
                #### Root Cause Analysis: Why 50% Confidence Occurred Previously
                1. **Uncalibrated Missing Weights**: If the previous checkpoint loaded with key mismatches, the linear classifier remained random/zero-centered, outputting `sigmoid(0) = 0.50`.
                2. **MTCNN Catch-All Fallback**: When an image lacked a front-facing face crop, MTCNN raised an error caught by a fallback returning `0.50`.
                3. **Domain Shift**: Models trained only on cropped faces produce ambiguous logits when fed full backgrounds without spatial attention.
                """
            )

    return demo


if __name__ == "__main__":
    if GRADIO_AVAILABLE:
        app = build_gradio_app()
        if app:
            app.launch(server_name="0.0.0.0", server_port=7860, share=False)
    else:
        print("[!] Run: pip install -r requirements.txt to launch Gradio app.")
