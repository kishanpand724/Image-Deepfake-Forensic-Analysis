import React, { useState } from 'react';
import { 
  Bug, 
  CheckCircle2, 
  XCircle, 
  AlertTriangle, 
  Cpu, 
  ArrowRight, 
  Terminal, 
  Layers,
  Code2,
  HelpCircle
} from 'lucide-react';

export const BugAutopsyView: React.FC = () => {
  const [selectedBugIdx, setSelectedBugIdx] = useState<number>(0);

  const bugs = [
    {
      id: 'bug-1',
      title: 'Problem 1: Mandatory MTCNN Face Cropping Dependency',
      severity: 'Critical Flaw',
      symptom: 'Non-faces, full-body poses, diffusion-generated scenes, and background manipulations failed or were cropped blindly.',
      rootCause: `
The legacy pipeline included a hard dependency:
\`\`\`python
# Broken Legacy Implementation:
faces, _ = mtcnn.detect(image)
if faces is None or len(faces) == 0:
    return 0.5 # Silent failure!
crop = image.crop(faces[0]) # Throws away entire background
output = model(crop)
\`\`\`
This discarded all environmental context, blending borders at the collar/hairline, and background document splices.`,
      solution: `
We replaced MTCNN face cropping with \`FullImagePreprocessor\`:
1. Resizes whole image preserving aspect ratio with symmetric letterbox padding.
2. Directs the full convolutional receptive field across both face and background scene.
3. Grad-CAM computes activation gradients across the complete image resolution without downsampling loss.`,
      status: 'Fixed & Verified'
    },
    {
      id: 'bug-2',
      title: 'Problem 2: Constant 50.0% Confidence Score Collapse',
      severity: 'Critical Bug',
      symptom: 'Completely different uploaded images yielded the exact same 50% or 0.5000 confidence score.',
      rootCause: `
In forensic deep learning architectures, three distinct issues cause the 50% collapse:
1. **Uncalibrated Missing Linear Layer**:
   When loading checkpoints saved with \`torch.nn.DataParallel\`, layer keys had \`module.\` prefixes. 
   When loaded with \`model.load_state_dict(checkpoint, strict=False)\`, the classification head was silently skipped! 
   The uninitialized linear layer had weights around zero and bias=0, producing logit = 0.0. 
   Since \`sigmoid(0.0) = 0.5000\`, every image output 50%!
2. **MTCNN Silent Exception Fallback**:
   When MTCNN failed on stylized, side-profile, or diffusion images, an outer \`try...except\` returned a default score of 0.5.
3. **CrossEntropy Softmax Zero-Vector**:
   Output logits \`[0.0, 0.0]\` fed into \`softmax()\` yield \`[0.5, 0.5]\`.`,
      solution: `
1. Implemented strict state_dict key stripping (removing \`module.\`, \`backbone.\`, \`net.\`).
2. Verified layer key counts (\`keys_matched\` vs \`keys_missing\`) in \`load_detector_model\`.
3. Applied temperature calibration (\`logits / T\`) with non-zero Xavier/Kaiming initialization.
4. Raised explicit diagnostic warnings instead of catching exceptions and returning 0.5.`,
      status: 'Fixed & Verified'
    },
    {
      id: 'bug-3',
      title: 'Problem 3: Cropped Face Visualization in Grad-CAM',
      severity: 'Visualization Limitation',
      symptom: 'Grad-CAM heatmaps were restricted to a tiny cropped face box, rendering the rest of the image black or invisible.',
      rootCause: `
Grad-CAM was invoked only on the tensor extracted by MTCNN (e.g. 160x160). 
The resulting heatmap was blended only with the cropped face thumbnail, making it impossible to audit background tampering, hair seams, or collar artifacts.`,
      solution: `
1. Grad-CAM hooks are attached to \`layer4\` of \`FullImageForensicNet\`.
2. Activation heatmaps span the entire normalized tensor.
3. Letterbox padding is inversely un-mapped to align precisely with the original high-resolution image dimensions.`,
      status: 'Fixed & Verified'
    },
    {
      id: 'bug-4',
      title: 'Problem 4: Prediction & Forensic Report Inconsistency',
      severity: 'Reliability Issue',
      symptom: 'Verdicts like "Authentic" were declared when the confidence was inconclusive, or verbal reports contradicted the confidence meter.',
      rootCause: `
Heuristic report generators used disconnected rule engines:
- If no face was found: report said "Authentic".
- If confidence was 51%: report said "High Confidence Deepfake".
- Lack of an explicit uncertainty boundary resulted in misleading binary judgments on low-quality media.`,
      solution: `
Built \`ForensicReportGenerator\` with:
- Formal mathematical consistency validator.
- Explicit **INCONCLUSIVE / UNCERTAIN** boundary between 42% and 58%.
- Multi-domain forensic pillars (spatial gradients, ELA, FFT, PRNU, and model calibration).
- Clear domain shift warning when checkpoints were trained solely on cropped faces.`,
      status: 'Fixed & Verified'
    }
  ];

  const currentBug = bugs[selectedBugIdx];

  return (
    <div className="space-y-6">
      <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-2xl flex flex-col md:flex-row items-start md:items-center justify-between gap-4">
        <div>
          <div className="flex items-center gap-2 mb-1">
            <Bug className="w-5 h-5 text-rose-400" />
            <h2 className="text-base font-bold text-slate-100">
              Technical Bug Autopsy & Architecture Diagnosis
            </h2>
          </div>
          <p className="text-xs text-slate-400">
            Comprehensive diagnostic analysis of why legacy detectors failed and how full-image analysis fixes each issue.
          </p>
        </div>

        <div className="flex items-center gap-2">
          <span className="px-2.5 py-1 text-xs font-mono font-semibold rounded-full bg-emerald-950 text-emerald-300 border border-emerald-800 flex items-center gap-1.5">
            <CheckCircle2 className="w-3.5 h-3.5" /> 4 of 4 Root Causes Fixed
          </span>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Navigation list */}
        <div className="bg-slate-900 border border-slate-800 rounded-xl p-3 shadow-xl space-y-2">
          {bugs.map((b, idx) => (
            <button
              key={b.id}
              onClick={() => setSelectedBugIdx(idx)}
              className={`w-full text-left p-3 rounded-lg transition-all border text-xs ${
                selectedBugIdx === idx
                  ? 'bg-indigo-950/60 border-indigo-600 text-white shadow-md'
                  : 'bg-slate-950/60 border-slate-800 text-slate-400 hover:border-slate-700 hover:text-slate-200'
              }`}
            >
              <div className="flex items-center justify-between mb-1">
                <span className="font-mono text-[10px] text-rose-400 uppercase font-semibold">{b.severity}</span>
                <span className="font-mono text-[10px] text-emerald-400">{b.status}</span>
              </div>
              <div className="font-semibold text-slate-200 line-clamp-1">{b.title}</div>
            </button>
          ))}

          {/* Model Domain Shift Callout */}
          <div className="mt-4 p-3 bg-slate-950 rounded-lg border border-slate-800 text-[11px] text-slate-400 space-y-2">
            <div className="font-semibold text-amber-300 flex items-center gap-1.5">
              <AlertTriangle className="w-3.5 h-3.5" />
              Face-Crop Domain Shift Alert
            </div>
            <p className="leading-relaxed">
              Models trained only on cropped faces (e.g., FaceForensics++) have no exposure to trees, skies, or wallpaper. 
              Applying them to full images without spatial attention or full-scene fine-tuning causes false positives on natural textures.
            </p>
          </div>
        </div>

        {/* Selected Bug Deep Dive */}
        <div className="lg:col-span-2 bg-slate-900 border border-slate-800 rounded-xl p-6 shadow-2xl space-y-5">
          <div className="border-b border-slate-800 pb-3">
            <span className="text-[10px] uppercase font-mono font-bold text-rose-400 px-2 py-0.5 rounded bg-rose-950/80 border border-rose-800 inline-block mb-1.5">
              {currentBug.severity}
            </span>
            <h3 className="text-lg font-bold text-slate-100">{currentBug.title}</h3>
            <p className="text-xs text-slate-400 mt-1">
              <strong>Observed Symptom:</strong> {currentBug.symptom}
            </p>
          </div>

          {/* Root Cause Section */}
          <div className="space-y-2">
            <h4 className="text-xs font-semibold text-rose-300 flex items-center gap-1.5">
              <XCircle className="w-4 h-4 text-rose-400" />
              Root Cause Identification
            </h4>
            <div className="bg-slate-950 p-4 rounded-xl border border-slate-800 text-xs font-mono text-slate-300 leading-relaxed whitespace-pre-line">
              {currentBug.rootCause}
            </div>
          </div>

          {/* Solution Implemented */}
          <div className="space-y-2">
            <h4 className="text-xs font-semibold text-emerald-300 flex items-center gap-1.5">
              <CheckCircle2 className="w-4 h-4 text-emerald-400" />
              Architectural Solution Implemented
            </h4>
            <div className="bg-slate-950 p-4 rounded-xl border border-slate-800 text-xs font-mono text-emerald-300/90 leading-relaxed whitespace-pre-line">
              {currentBug.solution}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
