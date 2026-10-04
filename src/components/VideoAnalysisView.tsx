import React, { useState } from 'react';
import { 
  Film, 
  Play, 
  Pause, 
  AlertTriangle, 
  CheckCircle2, 
  ChevronRight, 
  ChevronLeft,
  Flame,
  Activity,
  BarChart3,
  Sliders
} from 'lucide-react';
import { createSyntheticSampleImage } from '../forensicsEngine';

interface VideoFrame {
  frameNumber: number;
  timestamp: string;
  fakeProbability: number;
  isAnomaly: boolean;
  imageUrl: string;
  hasHeatmap: boolean;
}

export const VideoAnalysisView: React.FC = () => {
  const [selectedFrameIdx, setSelectedFrameIdx] = useState<number>(4);
  const [isPlaying, setIsPlaying] = useState<boolean>(false);
  const [overlayAlpha, setOverlayAlpha] = useState<number>(0.6);
  const [selectedColormap, setSelectedColormap] = useState<'turbo' | 'jet'>('turbo');

  // Simulated 12 sampled full-scene video frames (demonstrating deepfake video artifacts)
  const videoFrames: VideoFrame[] = [
    { frameNumber: 1, timestamp: '00:00.2', fakeProbability: 0.18, isAnomaly: false, imageUrl: createSyntheticSampleImage('genuine'), hasHeatmap: false },
    { frameNumber: 2, timestamp: '00:00.8', fakeProbability: 0.22, isAnomaly: false, imageUrl: createSyntheticSampleImage('genuine'), hasHeatmap: false },
    { frameNumber: 3, timestamp: '00:01.4', fakeProbability: 0.44, isAnomaly: false, imageUrl: createSyntheticSampleImage('uncertain'), hasHeatmap: false },
    { frameNumber: 4, timestamp: '00:02.0', fakeProbability: 0.89, isAnomaly: true, imageUrl: createSyntheticSampleImage('faceswap'), hasHeatmap: true },
    { frameNumber: 5, timestamp: '00:02.6', fakeProbability: 0.94, isAnomaly: true, imageUrl: createSyntheticSampleImage('faceswap'), hasHeatmap: true }, // Peak Anomaly Frame
    { frameNumber: 6, timestamp: '00:03.2', fakeProbability: 0.91, isAnomaly: true, imageUrl: createSyntheticSampleImage('faceswap'), hasHeatmap: true },
    { frameNumber: 7, timestamp: '00:03.8', fakeProbability: 0.85, isAnomaly: true, imageUrl: createSyntheticSampleImage('faceswap'), hasHeatmap: true },
    { frameNumber: 8, timestamp: '00:04.4', fakeProbability: 0.62, isAnomaly: false, imageUrl: createSyntheticSampleImage('faceswap'), hasHeatmap: true },
    { frameNumber: 9, timestamp: '00:05.0', fakeProbability: 0.35, isAnomaly: false, imageUrl: createSyntheticSampleImage('uncertain'), hasHeatmap: false },
    { frameNumber: 10, timestamp: '00:05.6', fakeProbability: 0.21, isAnomaly: false, imageUrl: createSyntheticSampleImage('genuine'), hasHeatmap: false },
    { frameNumber: 11, timestamp: '00:06.2', fakeProbability: 0.19, isAnomaly: false, imageUrl: createSyntheticSampleImage('genuine'), hasHeatmap: false },
    { frameNumber: 12, timestamp: '00:06.8', fakeProbability: 0.15, isAnomaly: false, imageUrl: createSyntheticSampleImage('genuine'), hasHeatmap: false },
  ];

  const currentFrame = videoFrames[selectedFrameIdx];
  const meanScore = videoFrames.reduce((acc, f) => acc + f.fakeProbability, 0) / videoFrames.length;
  const maxScore = Math.max(...videoFrames.map(f => f.fakeProbability));
  const peakFrame = videoFrames.find(f => f.fakeProbability === maxScore) || videoFrames[4];

  // Calculate temporal jitter: average frame-to-frame diff
  let jitterSum = 0;
  for (let i = 1; i < videoFrames.length; i++) {
    jitterSum += Math.abs(videoFrames[i].fakeProbability - videoFrames[i - 1].fakeProbability);
  }
  const temporalJitter = jitterSum / (videoFrames.length - 1);

  return (
    <div className="space-y-6">
      {/* Video Aggregated Banner */}
      <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-2xl flex flex-col md:flex-row items-start md:items-center justify-between gap-4">
        <div>
          <div className="flex items-center gap-2 mb-1">
            <Film className="w-5 h-5 text-indigo-400" />
            <h2 className="text-base font-bold text-slate-100">
              Full-Frame Video Temporal Analysis
            </h2>
            <span className="px-2 py-0.5 rounded text-[11px] font-mono bg-rose-950 text-rose-300 border border-rose-800">
              AGGREGATED VERDICT: SYNTHETIC VIDEO (FAKE)
            </span>
          </div>
          <p className="text-xs text-slate-400">
            Multi-frame full-scene sampling across 12 temporal keyframes without facial cropping distortion.
          </p>
        </div>

        {/* Aggregated Quick Metrics */}
        <div className="flex items-center gap-4 bg-slate-950 px-4 py-2 rounded-xl border border-slate-800 text-xs">
          <div>
            <div className="text-slate-400 text-[10px]">Aggregated Conf:</div>
            <div className="text-base font-mono font-bold text-rose-400">{(meanScore * 100).toFixed(1)}%</div>
          </div>
          <div className="w-px h-8 bg-slate-800" />
          <div>
            <div className="text-slate-400 text-[10px]">Peak Anomaly:</div>
            <div className="text-base font-mono font-bold text-amber-400">Frame #{peakFrame.frameNumber}</div>
          </div>
          <div className="w-px h-8 bg-slate-800" />
          <div>
            <div className="text-slate-400 text-[10px]">Temporal Jitter:</div>
            <div className="text-base font-mono font-bold text-indigo-300">{temporalJitter.toFixed(3)}</div>
          </div>
        </div>
      </div>

      {/* Main Video Frame & Grad-CAM Inspector */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left: Active Full-Frame with Grad-CAM Overlay */}
        <div className="lg:col-span-2 bg-slate-900 border border-slate-800 rounded-xl overflow-hidden shadow-2xl flex flex-col">
          <div className="flex items-center justify-between px-4 py-3 bg-slate-950 border-b border-slate-800">
            <div className="flex items-center gap-2">
              <span className="text-xs font-semibold text-slate-200">
                Inspecting Frame #{currentFrame.frameNumber} ({currentFrame.timestamp})
              </span>
              {currentFrame.isAnomaly && (
                <span className="px-1.5 py-0.5 text-[10px] font-bold bg-rose-600 text-white rounded animate-pulse">
                  PEAK ANOMALY SPIKE
                </span>
              )}
            </div>

            <div className="flex items-center gap-2 text-xs">
              <span className="text-slate-400">Alpha:</span>
              <input
                type="range"
                min="0.1"
                max="1.0"
                step="0.1"
                value={overlayAlpha}
                onChange={(e) => setOverlayAlpha(parseFloat(e.target.value))}
                className="w-20 accent-indigo-500"
              />
            </div>
          </div>

          {/* Frame Display */}
          <div className="relative flex-1 bg-slate-950 flex items-center justify-center p-3 min-h-[380px]">
            <img
              src={currentFrame.imageUrl}
              alt={`Frame ${currentFrame.frameNumber}`}
              className="max-h-[360px] w-auto rounded border border-slate-800 object-contain shadow-2xl"
            />

            {/* If frame has synthetic anomalies, render simulated full-scene Grad-CAM overlay */}
            {currentFrame.fakeProbability > 0.5 && (
              <div 
                style={{ opacity: overlayAlpha }}
                className="absolute inset-x-8 inset-y-6 pointer-events-none rounded mix-blend-screen flex items-center justify-center"
              >
                <div 
                  className="w-48 h-36 rounded-full blur-2xl"
                  style={{
                    background: selectedColormap === 'turbo' 
                      ? 'radial-gradient(circle, rgba(239,68,68,0.9) 0%, rgba(234,179,8,0.7) 40%, rgba(59,130,246,0) 80%)'
                      : 'radial-gradient(circle, rgba(220,38,38,0.9) 0%, rgba(249,115,22,0.7) 40%, rgba(37,99,235,0) 80%)'
                  }}
                />
              </div>
            )}
          </div>

          {/* Frame Navigation Controls */}
          <div className="flex items-center justify-between px-4 py-3 bg-slate-950 border-t border-slate-800">
            <button
              onClick={() => setSelectedFrameIdx(Math.max(0, selectedFrameIdx - 1))}
              disabled={selectedFrameIdx === 0}
              className="px-3 py-1.5 rounded-lg bg-slate-800 text-slate-300 hover:bg-slate-700 disabled:opacity-40 text-xs font-semibold flex items-center gap-1"
            >
              <ChevronLeft className="w-4 h-4" /> Prev Frame
            </button>

            <span className="text-xs font-mono text-slate-300">
              Frame {selectedFrameIdx + 1} of {videoFrames.length}
            </span>

            <button
              onClick={() => setSelectedFrameIdx(Math.min(videoFrames.length - 1, selectedFrameIdx + 1))}
              disabled={selectedFrameIdx === videoFrames.length - 1}
              className="px-3 py-1.5 rounded-lg bg-slate-800 text-slate-300 hover:bg-slate-700 disabled:opacity-40 text-xs font-semibold flex items-center gap-1"
            >
              Next Frame <ChevronRight className="w-4 h-4" />
            </button>
          </div>
        </div>

        {/* Right: Temporal Anomaly Timeline & Diagnostics */}
        <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-2xl flex flex-col space-y-4">
          <div className="flex items-center gap-2 border-b border-slate-800 pb-3">
            <BarChart3 className="w-4 h-4 text-amber-400" />
            <h3 className="text-sm font-semibold text-slate-200">Temporal Anomaly Curve</h3>
          </div>

          {/* Interactive Timeline Bar Chart */}
          <div className="space-y-1.5">
            <div className="text-[11px] text-slate-400 flex justify-between">
              <span>Frame-by-Frame Fake Score</span>
              <span className="text-amber-400">Click bar to select frame</span>
            </div>

            <div className="h-32 flex items-end gap-1.5 bg-slate-950 p-2.5 rounded-xl border border-slate-800">
              {videoFrames.map((frame, idx) => {
                const heightPercent = Math.round(frame.fakeProbability * 100);
                const isSelected = idx === selectedFrameIdx;
                const isHigh = frame.fakeProbability > 0.6;

                return (
                  <div
                    key={frame.frameNumber}
                    onClick={() => setSelectedFrameIdx(idx)}
                    title={`Frame #${frame.frameNumber}: ${(frame.fakeProbability * 100).toFixed(1)}%`}
                    className="flex-1 h-full flex flex-col justify-end items-center cursor-pointer group"
                  >
                    <div
                      style={{ height: `${heightPercent}%` }}
                      className={`w-full rounded-t transition-all ${
                        isSelected
                          ? 'bg-amber-400 ring-2 ring-amber-300 shadow-lg'
                          : isHigh
                          ? 'bg-rose-500/80 group-hover:bg-rose-400'
                          : 'bg-emerald-500/70 group-hover:bg-emerald-400'
                      }`}
                    />
                    <span className="text-[9px] font-mono text-slate-500 mt-1">#{frame.frameNumber}</span>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Temporal Consistency Audit */}
          <div className="bg-slate-950 p-3.5 rounded-xl border border-slate-800 space-y-2 text-xs">
            <div className="font-semibold text-slate-200 flex items-center gap-1.5">
              <Activity className="w-4 h-4 text-indigo-400" />
              Temporal Deepfake Signatures
            </div>
            <p className="text-[11px] text-slate-400 leading-relaxed">
              Deepfakes generated with frame-by-frame models suffer from high-frequency facial jitter and boundary flashing 
              between seconds 02.0 and 03.8. Our full-frame temporal aggregator captures this flicker across consecutive frames.
            </p>
            <div className="pt-2 border-t border-slate-800/80 flex justify-between text-[11px]">
              <span className="text-slate-400">Anomaly Cluster:</span>
              <span className="font-mono text-rose-400 font-semibold">Frames #4 - #8 (Duration: 1.8s)</span>
            </div>
          </div>

          {/* Video upload notice */}
          <div className="p-3 bg-indigo-950/30 rounded-xl border border-indigo-900/40 text-[11px] text-indigo-300">
            <strong>Gradio Integration Note:</strong> The accompanying Python backend supports direct MP4/AVI/MOV video ingestion 
            via OpenCV and Decord, with batched full-image tensor normalization.
          </div>
        </div>
      </div>
    </div>
  );
};
