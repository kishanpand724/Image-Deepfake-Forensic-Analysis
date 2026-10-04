import React, { useState, useEffect, useRef } from 'react';
import { 
  Eye, 
  Layers, 
  Sparkles, 
  Sliders, 
  Maximize2, 
  AlertTriangle, 
  Crosshair, 
  Info,
  SplitSquareHorizontal,
  Flame
} from 'lucide-react';
import { BoundingBox, ForensicResult } from '../forensicsEngine';

interface GradCamViewerProps {
  imageSrc: string;
  result: ForensicResult | null;
  activeColormap: 'turbo' | 'jet' | 'thermal';
  overlayAlpha: number;
  onAlphaChange: (val: number) => void;
  onColormapChange: (map: 'turbo' | 'jet' | 'thermal') => void;
  showLegacyComparison: boolean;
  onToggleLegacyComparison: () => void;
}

export const GradCamViewer: React.FC<GradCamViewerProps> = ({
  imageSrc,
  result,
  activeColormap,
  overlayAlpha,
  onAlphaChange,
  onColormapChange,
  showLegacyComparison,
  onToggleLegacyComparison,
}) => {
  const [viewMode, setViewMode] = useState<'overlay' | 'split' | 'heatmap_only' | 'original'>('overlay');
  const [splitPos, setSplitPos] = useState<number>(50);
  const [showBoundingBoxes, setShowBoundingBoxes] = useState<boolean>(true);
  const [selectedBox, setSelectedBox] = useState<BoundingBox | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const containerRef = useRef<HTMLDivElement | null>(null);

  // Render Full-Image Grad-CAM canvas
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.onload = () => {
      canvas.width = img.naturalWidth || 640;
      canvas.height = img.naturalHeight || 420;
      const w = canvas.width;
      const h = canvas.height;

      // 1. Draw original base image
      ctx.drawImage(img, 0, 0, w, h);

      if (viewMode === 'original') return;

      // 2. Generate smooth 2D gradient activation field across full image
      const heatCanvas = document.createElement('canvas');
      heatCanvas.width = w;
      heatCanvas.height = h;
      const hCtx = heatCanvas.getContext('2d');
      if (!hCtx) return;

      // Determine hotspot centers
      const boxes = result?.anomalyRegions || [];
      if (boxes.length === 0) {
        // Natural image: low uniform background heat with subtle center focus
        const grad = hCtx.createRadialGradient(w * 0.5, h * 0.45, 10, w * 0.5, h * 0.45, w * 0.5);
        grad.addColorStop(0, 'rgba(30, 64, 175, 0.4)');
        grad.addColorStop(0.6, 'rgba(15, 23, 42, 0.2)');
        grad.addColorStop(1, 'rgba(0, 0, 0, 0)');
        hCtx.fillStyle = grad;
        hCtx.fillRect(0, 0, w, h);
      } else {
        // Render each hotspot according to selected colormap
        boxes.forEach((box) => {
          const cx = box.x + box.width / 2;
          const cy = box.y + box.height / 2;
          const radius = Math.max(box.width, box.height) * 0.85;

          const grad = hCtx.createRadialGradient(cx, cy, 5, cx, cy, radius);

          if (activeColormap === 'turbo') {
            // Turbo colormap: Blue -> Cyan -> Green -> Yellow -> Red
            grad.addColorStop(0, 'rgba(239, 68, 68, 0.95)'); // Hot Red
            grad.addColorStop(0.35, 'rgba(234, 179, 8, 0.85)'); // Yellow
            grad.addColorStop(0.6, 'rgba(34, 197, 94, 0.7)'); // Green
            grad.addColorStop(0.85, 'rgba(6, 182, 212, 0.4)'); // Cyan
            grad.addColorStop(1, 'rgba(59, 130, 246, 0)'); // Fading blue
          } else if (activeColormap === 'jet') {
            // Classic Jet: Red -> Yellow -> Green -> Cyan -> Blue
            grad.addColorStop(0, 'rgba(220, 38, 38, 0.95)');
            grad.addColorStop(0.3, 'rgba(249, 115, 22, 0.85)');
            grad.addColorStop(0.6, 'rgba(234, 179, 8, 0.7)');
            grad.addColorStop(0.85, 'rgba(16, 185, 129, 0.4)');
            grad.addColorStop(1, 'rgba(37, 99, 235, 0)');
          } else {
            // Thermal: White -> Yellow -> Red -> Purple -> Black
            grad.addColorStop(0, 'rgba(255, 255, 255, 0.95)');
            grad.addColorStop(0.25, 'rgba(250, 204, 21, 0.85)');
            grad.addColorStop(0.6, 'rgba(225, 29, 72, 0.75)');
            grad.addColorStop(0.85, 'rgba(109, 40, 217, 0.5)');
            grad.addColorStop(1, 'rgba(15, 23, 42, 0)');
          }

          hCtx.fillStyle = grad;
          hCtx.fillRect(0, 0, w, h);
        });
      }

      // If viewMode is heatmap_only
      if (viewMode === 'heatmap_only') {
        ctx.fillStyle = '#090d16';
        ctx.fillRect(0, 0, w, h);
        ctx.drawImage(heatCanvas, 0, 0);
        return;
      }

      // If viewMode is split
      if (viewMode === 'split') {
        const splitX = (splitPos / 100) * w;
        ctx.save();
        ctx.beginPath();
        ctx.rect(splitX, 0, w - splitX, h);
        ctx.clip();
        ctx.globalAlpha = overlayAlpha;
        ctx.drawImage(heatCanvas, 0, 0);
        ctx.restore();

        // Draw divider line
        ctx.strokeStyle = '#ffffff';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(splitX, 0);
        ctx.lineTo(splitX, h);
        ctx.stroke();
        return;
      }

      // Default: Blend overlay with alpha slider
      ctx.save();
      ctx.globalAlpha = overlayAlpha;
      ctx.drawImage(heatCanvas, 0, 0);
      ctx.restore();
    };
    img.src = imageSrc;
  }, [imageSrc, result, activeColormap, overlayAlpha, viewMode, splitPos]);

  return (
    <div className="flex flex-col bg-slate-900 border border-slate-800 rounded-xl overflow-hidden shadow-2xl">
      {/* Top Bar with Controls */}
      <div className="flex flex-wrap items-center justify-between px-4 py-3 bg-slate-950/80 border-b border-slate-800 gap-2">
        <div className="flex items-center gap-2">
          <Flame className="w-5 h-5 text-amber-400" />
          <span className="text-sm font-semibold tracking-wide text-slate-200">
            Full-Image Grad-CAM Heatmap
          </span>
          <span className="px-2 py-0.5 text-xs font-mono rounded bg-indigo-950/80 text-indigo-300 border border-indigo-700/50">
            {showLegacyComparison ? 'Legacy MTCNN Face Crop (Simulated)' : 'Native 256x256 Letterboxed'}
          </span>
        </div>

        {/* View Mode Switcher */}
        <div className="flex items-center bg-slate-800/80 p-0.5 rounded-lg border border-slate-700 text-xs">
          <button
            onClick={() => setViewMode('overlay')}
            className={`px-2.5 py-1 rounded transition-all font-medium ${
              viewMode === 'overlay' ? 'bg-indigo-600 text-white shadow-sm' : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            Overlay
          </button>
          <button
            onClick={() => setViewMode('split')}
            className={`px-2.5 py-1 rounded transition-all font-medium ${
              viewMode === 'split' ? 'bg-indigo-600 text-white shadow-sm' : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            Split Wipe
          </button>
          <button
            onClick={() => setViewMode('heatmap_only')}
            className={`px-2.5 py-1 rounded transition-all font-medium ${
              viewMode === 'heatmap_only' ? 'bg-indigo-600 text-white shadow-sm' : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            Heatmap Only
          </button>
          <button
            onClick={() => setViewMode('original')}
            className={`px-2.5 py-1 rounded transition-all font-medium ${
              viewMode === 'original' ? 'bg-indigo-600 text-white shadow-sm' : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            Original
          </button>
        </div>
      </div>

      {/* Main Visual Display Area */}
      <div 
        ref={containerRef}
        className="relative flex items-center justify-center bg-slate-950 p-2 min-h-[360px] overflow-hidden"
      >
        {showLegacyComparison ? (
          /* Visualizing the BUGGY legacy MTCNN pipeline: cropped square face */
          <div className="relative flex flex-col items-center justify-center p-6 border-2 border-dashed border-rose-500/40 rounded-xl bg-rose-950/20 max-w-md text-center">
            <div className="relative w-56 h-56 rounded-lg overflow-hidden border-2 border-rose-500 shadow-xl mb-4 bg-slate-900">
              <img 
                src={imageSrc} 
                alt="Legacy Cropped Face" 
                className="w-full h-full object-cover scale-150 translate-y-[-10%]"
              />
              <div className="absolute inset-0 bg-rose-500/20 mix-blend-overlay" />
              <div className="absolute top-2 left-2 px-1.5 py-0.5 text-[10px] font-mono bg-rose-600 text-white rounded">
                MTCNN Face Box (160x160)
              </div>
            </div>
            <div className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-rose-900/60 border border-rose-700/60 text-rose-300 text-xs font-semibold mb-2">
              <AlertTriangle className="w-3.5 h-3.5" />
              Legacy Pipeline: Background Discarded!
            </div>
            <p className="text-xs text-slate-400 leading-relaxed max-w-xs">
              Notice: The entire background scene, body posture, and neck splicing seams are cropped away. 
              Uncalibrated linear head outputs fixed <strong>50.0% confidence</strong>.
            </p>
            <button
              onClick={onToggleLegacyComparison}
              className="mt-3 px-3 py-1.5 text-xs font-semibold bg-indigo-600 hover:bg-indigo-500 text-white rounded-lg transition-colors"
            >
              Switch Back to Full-Image Analysis
            </button>
          </div>
        ) : (
          /* Normal Upgraded Full-Image Canvas */
          <div className="relative max-w-full">
            <canvas
              ref={canvasRef}
              className="max-h-[500px] w-auto h-auto rounded-lg shadow-inner object-contain block mx-auto"
            />

            {/* Bounding box overlays for detected anomalies across full frame */}
            {showBoundingBoxes && result?.anomalyRegions.map((box, idx) => (
              <div
                key={idx}
                onClick={() => setSelectedBox(box)}
                style={{
                  left: `${(box.x / (result.letterboxMetadata.originalWidth || 640)) * 100}%`,
                  top: `${(box.y / (result.letterboxMetadata.originalHeight || 420)) * 100}%`,
                  width: `${(box.width / (result.letterboxMetadata.originalWidth || 640)) * 100}%`,
                  height: `${(box.height / (result.letterboxMetadata.originalHeight || 420)) * 100}%`,
                }}
                className={`absolute border-2 border-dashed cursor-pointer transition-all duration-200 group ${
                  selectedBox === box
                    ? 'border-amber-400 bg-amber-400/20 shadow-lg ring-2 ring-amber-400/50'
                    : 'border-rose-500/80 hover:border-amber-300 hover:bg-rose-500/10'
                }`}
              >
                <div className="absolute -top-6 left-0 px-1.5 py-0.5 bg-rose-600/90 text-white text-[10px] font-mono rounded whitespace-nowrap shadow flex items-center gap-1 group-hover:bg-amber-500">
                  <Crosshair className="w-3 h-3" />
                  {box.label} ({(box.intensity * 100).toFixed(0)}%)
                </div>
              </div>
            ))}

            {/* Split Screen Slider Control */}
            {viewMode === 'split' && (
              <div className="absolute inset-x-0 bottom-3 px-6 flex items-center justify-center">
                <div className="bg-slate-900/90 backdrop-blur-md px-4 py-2 rounded-xl border border-slate-700 shadow-xl flex items-center gap-3 w-80">
                  <SplitSquareHorizontal className="w-4 h-4 text-indigo-400" />
                  <input
                    type="range"
                    min="5"
                    max="95"
                    value={splitPos}
                    onChange={(e) => setSplitPos(Number(e.target.value))}
                    className="w-full accent-indigo-500 cursor-ew-resize"
                  />
                  <span className="text-xs font-mono text-slate-300 w-10 text-right">{splitPos}%</span>
                </div>
              </div>
            )}
          </div>
        )}
      </div>

      {/* Control Footer */}
      <div className="flex flex-wrap items-center justify-between px-4 py-3 bg-slate-950 border-t border-slate-800 text-xs text-slate-400 gap-3">
        {/* Opacity Slider */}
        <div className="flex items-center gap-2 min-w-[200px]">
          <span className="font-medium text-slate-300">Opacity:</span>
          <input
            type="range"
            min="0.1"
            max="1.0"
            step="0.05"
            value={overlayAlpha}
            onChange={(e) => onAlphaChange(parseFloat(e.target.value))}
            className="w-28 accent-indigo-500 cursor-pointer"
          />
          <span className="font-mono text-slate-300 w-8">{Math.round(overlayAlpha * 100)}%</span>
        </div>

        {/* Colormap Selector */}
        <div className="flex items-center gap-2">
          <span className="font-medium text-slate-300">Palette:</span>
          <div className="flex rounded-md overflow-hidden border border-slate-700">
            {(['turbo', 'jet', 'thermal'] as const).map((map) => (
              <button
                key={map}
                onClick={() => onColormapChange(map)}
                className={`px-2.5 py-1 text-[11px] font-medium uppercase tracking-wider transition-colors ${
                  activeColormap === map
                    ? 'bg-slate-700 text-amber-300'
                    : 'bg-slate-800/80 text-slate-400 hover:text-slate-200'
                }`}
              >
                {map}
              </button>
            ))}
          </div>
        </div>

        {/* Bounding box toggle */}
        <label className="flex items-center gap-2 cursor-pointer select-none">
          <input
            type="checkbox"
            checked={showBoundingBoxes}
            onChange={(e) => setShowBoundingBoxes(e.target.checked)}
            className="rounded border-slate-700 text-indigo-600 focus:ring-0"
          />
          <span className="text-slate-300">Anomaly Boxes</span>
        </label>

        {/* Legacy Comparison Toggle */}
        <button
          onClick={onToggleLegacyComparison}
          className={`px-3 py-1.5 rounded-lg border text-xs font-semibold flex items-center gap-1.5 transition-all ${
            showLegacyComparison
              ? 'bg-rose-950 border-rose-600 text-rose-300 shadow-sm'
              : 'bg-slate-800/80 border-slate-700 text-slate-300 hover:border-slate-500 hover:text-white'
          }`}
        >
          <AlertTriangle className="w-3.5 h-3.5 text-amber-400" />
          {showLegacyComparison ? 'Viewing Buggy Legacy MTCNN' : 'Compare with Legacy 50% MTCNN'}
        </button>
      </div>

      {/* Selected Anomaly Inspector Drawer */}
      {selectedBox && (
        <div className="px-4 py-2.5 bg-slate-900/90 border-t border-slate-800 flex items-center justify-between text-xs animate-fadeIn">
          <div className="flex items-center gap-2">
            <span className="w-2.5 h-2.5 rounded-full bg-amber-400 animate-pulse" />
            <span className="font-semibold text-slate-200">Inspecting Anomaly:</span>
            <span className="text-amber-300 font-mono">{selectedBox.label}</span>
            <span className="text-slate-500">|</span>
            <span className="text-slate-400">Peak Gradient:</span>
            <span className="font-mono text-emerald-400">{(selectedBox.intensity * 100).toFixed(1)}%</span>
            <span className="text-slate-500">|</span>
            <span className="text-slate-400">Region:</span>
            <span className="font-mono text-slate-300">
              {Math.round(selectedBox.width)}x{Math.round(selectedBox.height)} px
            </span>
          </div>
          <button
            onClick={() => setSelectedBox(null)}
            className="text-slate-400 hover:text-white underline text-[11px]"
          >
            Dismiss
          </button>
        </div>
      )}
    </div>
  );
};
