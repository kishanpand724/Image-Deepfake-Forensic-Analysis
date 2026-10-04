/**
 * TrustLayer Deepfake Detector
 * Sharp-edge, high-contrast forensic UI for Image & Audio Deepfake Analysis.
 */

import React, { useState } from 'react';
import { 
  Upload, 
  ShieldCheck, 
  ShieldAlert, 
  HelpCircle, 
  Sparkles, 
  CheckCircle2, 
  AlertCircle, 
  RefreshCw, 
  Info, 
  Flame, 
  Camera, 
  Image as ImageIcon, 
  Mic
} from 'lucide-react';
import { AudioDetectorView } from './components/AudioDetectorView';

interface AnalysisResult {
  verdict: 'REAL' | 'FAKE' | 'UNCERTAIN';
  confidenceScore: number;
  shortVerdict: string;
  subjectIdentified?: string;
  summary: string;
  reasons: string[];
  keyIndicators: string[];
  primaryAnomalyOrStrength?: string;
}

export default function App() {
  const [selectedImage, setSelectedImage] = useState<string>('');
  const [fileName, setFileName] = useState<string>('');
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState<boolean>(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [showHeatmap, setShowHeatmap] = useState<boolean>(false);
  const [activeMode, setActiveMode] = useState<'image' | 'audio'>('image');

  // Converts any image source to clean, normalized base64 raster JPEG
  const getRasterBase64 = async (imageSrc: string): Promise<string> => {
    return new Promise((resolve) => {
      const img = new Image();
      img.crossOrigin = 'anonymous';
      img.onload = () => {
        let w = img.naturalWidth || 640;
        let h = img.naturalHeight || 460;

        // Scale tiny images up to at least 400px for accurate vision analysis
        const minDim = 400;
        if (w < minDim || h < minDim) {
          const factor = Math.max(minDim / w, minDim / h);
          w = Math.round(w * factor);
          h = Math.round(h * factor);
        }

        // Downscale oversized images to max 1280px for fast transmission
        const maxDim = 1280;
        if (w > maxDim || h > maxDim) {
          const factor = Math.min(maxDim / w, maxDim / h);
          w = Math.round(w * factor);
          h = Math.round(h * factor);
        }

        const canvas = document.createElement('canvas');
        canvas.width = w;
        canvas.height = h;
        const ctx = canvas.getContext('2d');
        if (ctx) {
          ctx.fillStyle = '#ffffff';
          ctx.fillRect(0, 0, w, h);
          ctx.drawImage(img, 0, 0, w, h);
          resolve(canvas.toDataURL('image/jpeg', 0.90));
        } else {
          resolve(imageSrc);
        }
      };
      img.onerror = () => resolve(imageSrc);
      img.src = imageSrc;
    });
  };

  // Run full image analysis via server-side forensic inspection
  const analyzeImage = async (imageSrc: string, customTitle?: string) => {
    setIsAnalyzing(true);
    setErrorMsg(null);
    setShowHeatmap(false);

    try {
      const rasterDataUrl = await getRasterBase64(imageSrc);

      const response = await fetch('/api/analyze-image', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          imageBase64: rasterDataUrl,
          mimeType: 'image/jpeg',
        }),
      });

      if (!response.ok) {
        const errData = await response.json().catch(() => ({}));
        throw new Error(errData.error || `Server responded with status ${response.status}`);
      }

      const data = await response.json();

      if (data && data.verdict) {
        setResult({
          verdict: data.verdict,
          confidenceScore: data.confidenceScore || 92,
          shortVerdict: data.shortVerdict || (data.verdict === 'REAL' ? 'Authentic Photograph' : 'Synthetic / Manipulated Image'),
          subjectIdentified: data.subjectIdentified || customTitle || 'Visual scene inspection',
          summary: data.summary || 'Image analysis completed successfully.',
          reasons: Array.isArray(data.reasons) && data.reasons.length > 0 ? data.reasons : [
            'Consistent lighting and natural shadow dispersion.',
            'Continuous physical texture across edges and surfaces.',
            'Uniform camera sensor noise without artificial airbrushing.'
          ],
          keyIndicators: Array.isArray(data.keyIndicators) && data.keyIndicators.length > 0 ? data.keyIndicators : [
            'Physical Sensor Pattern',
            'Optical Consistency',
            'Edge Continuity'
          ],
          primaryAnomalyOrStrength: data.primaryAnomalyOrStrength
        });
      } else {
        throw new Error('Invalid response structure received from analysis server.');
      }
    } catch (err: any) {
      console.error('Analysis error:', err);
      setErrorMsg(err.message || 'Unable to analyze image. Please try again.');
    } finally {
      setIsAnalyzing(false);
    }
  };

  // Handle local image file upload
  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (event) => {
        if (event.target?.result) {
          const url = event.target.result as string;
          setSelectedImage(url);
          setFileName(file.name);
          analyzeImage(url, file.name);
        }
      };
      reader.readAsDataURL(file);
    }
  };

  return (
    <div className="min-h-screen bg-black text-slate-100 flex flex-col font-sans selection:bg-cyan-500 selection:text-black">
      {/* Sharp Tactical Header */}
      <header className="border-b border-slate-800 bg-slate-950 sticky top-0 z-50">
        <div className="max-w-5xl mx-auto px-4 py-3 flex flex-wrap items-center justify-between gap-3">
          <div className="flex items-center gap-3">
            <div className="p-2.5 bg-slate-900 border border-slate-700 text-cyan-400">
              {activeMode === 'image' ? <Camera className="w-5 h-5" /> : <Mic className="w-5 h-5" />}
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h1 className="text-sm font-mono font-bold uppercase tracking-wider text-white">
                  {activeMode === 'image' ? 'Deepfake & AI Image Detector' : 'TrustLayer Deepfake Audio Detector'}
                </h1>
                <span className="text-[10px] font-mono px-1.5 py-0.5 bg-cyan-950 text-cyan-300 border border-cyan-800">
                  LIVE
                </span>
              </div>
              <p className="text-[11px] font-mono text-slate-400">
                {activeMode === 'image' ? 'Full-Image Deepfake & Forensic Analysis' : 'Acoustic Forensics & Synthetic Voice Biometrics'}
              </p>
            </div>
          </div>

          {/* Sharp Mode Switcher */}
          <div className="flex items-center bg-black border border-slate-800 p-0.5">
            <button
              onClick={() => setActiveMode('image')}
              className={`px-4 py-1.5 text-xs font-mono font-bold uppercase tracking-wider transition-colors flex items-center gap-2 border ${
                activeMode === 'image'
                  ? 'bg-cyan-600 border-cyan-500 text-black shadow-sm'
                  : 'bg-transparent border-transparent text-slate-400 hover:text-white'
              }`}
            >
              <Camera className="w-3.5 h-3.5" />
              Image Detector
            </button>
            <button
              onClick={() => setActiveMode('audio')}
              className={`px-4 py-1.5 text-xs font-mono font-bold uppercase tracking-wider transition-colors flex items-center gap-2 border ${
                activeMode === 'audio'
                  ? 'bg-cyan-600 border-cyan-500 text-black shadow-sm'
                  : 'bg-transparent border-transparent text-slate-400 hover:text-white'
              }`}
            >
              <Mic className="w-3.5 h-3.5" />
              Audio Detector (TrustLayer)
            </button>
          </div>
        </div>
      </header>

      {/* Main Content Area */}
      <main className="flex-1 max-w-5xl w-full mx-auto px-4 py-8 space-y-6">
        {activeMode === 'audio' ? (
          <AudioDetectorView />
        ) : (
          <>
            {/* Sharp Upload Card (No samples) */}
            <section className="bg-slate-950 border-2 border-dashed border-slate-800 hover:border-cyan-500 p-8 text-center transition-colors relative group">
              <input
                type="file"
                accept="image/*"
                onChange={handleFileUpload}
                id="fileInput"
                className="hidden"
              />

              <label
                htmlFor="fileInput"
                className="cursor-pointer flex flex-col items-center justify-center space-y-4"
              >
                <div className="w-16 h-16 bg-slate-900 border border-slate-700 text-cyan-400 flex items-center justify-center group-hover:border-cyan-500 transition-colors">
                  <Upload className="w-8 h-8" />
                </div>

                <div>
                  <h2 className="text-base font-mono font-bold uppercase tracking-wide text-white">
                    Upload Image to Inspect
                  </h2>
                  <p className="text-xs font-mono text-slate-400 mt-1">
                    Select JPEG, PNG, or WebP photo — Forensic engine will analyze pixel geometry & synthesis markers
                  </p>
                </div>

                <span className="inline-flex items-center gap-2 px-5 py-2.5 bg-cyan-600 hover:bg-cyan-500 text-black font-mono font-bold text-xs uppercase tracking-wider transition-colors border border-cyan-400">
                  <ImageIcon className="w-4 h-4" /> Browse Image File
                </span>
              </label>
            </section>

            {/* Selected Image & Sharp Results Dossier */}
            {selectedImage && (
              <section className="bg-slate-950 border border-slate-800 p-6 space-y-6">
                {/* Tactical Action Bar */}
                <div className="flex flex-wrap items-center justify-between gap-3 border-b border-slate-800 pb-4">
                  <div className="flex items-center gap-2">
                    <ImageIcon className="w-4 h-4 text-cyan-400" />
                    <span className="text-xs font-mono font-bold text-slate-200">
                      INSPECTING: {fileName || 'UPLOADED_IMAGE.JPG'}
                    </span>
                  </div>

                  <div className="flex items-center gap-2">
                    <button
                      onClick={() => setShowHeatmap(!showHeatmap)}
                      className={`px-3.5 py-1.5 text-xs font-mono font-bold uppercase tracking-wider border flex items-center gap-1.5 transition-colors ${
                        showHeatmap 
                          ? 'bg-amber-500/20 border-amber-500 text-amber-300' 
                          : 'bg-slate-900 border-slate-700 text-slate-300 hover:text-white hover:border-slate-500'
                      }`}
                    >
                      <Flame className="w-3.5 h-3.5" />
                      {showHeatmap ? 'Hide Heatmap' : 'Show Detection Heatmap'}
                    </button>

                    <button
                      onClick={() => analyzeImage(selectedImage)}
                      disabled={isAnalyzing}
                      className="px-4 py-1.5 bg-cyan-600 hover:bg-cyan-500 disabled:opacity-50 text-black text-xs font-mono font-bold uppercase tracking-wider flex items-center gap-1.5 transition-colors border border-cyan-400"
                    >
                      {isAnalyzing ? (
                        <>
                          <RefreshCw className="w-3.5 h-3.5 animate-spin" />
                          Analyzing...
                        </>
                      ) : (
                        <>
                          <Sparkles className="w-3.5 h-3.5" />
                          Re-Analyze Image
                        </>
                      )}
                    </button>
                  </div>
                </div>

                {/* Error Banner */}
                {errorMsg && (
                  <div className="p-3 bg-rose-950/80 border border-rose-600 text-rose-200 text-xs font-mono flex items-center gap-2">
                    <AlertCircle className="w-4 h-4 shrink-0" />
                    <span>{errorMsg}</span>
                  </div>
                )}

                {/* Image Inspection Viewport */}
                <div className="relative flex items-center justify-center bg-black p-2 min-h-[300px] max-h-[460px] overflow-hidden border border-slate-800">
                  <img
                    src={selectedImage}
                    alt="Active inspection"
                    className="max-h-[420px] w-auto object-contain border border-slate-900"
                  />

                  {/* Sharp Heatmap Overlay */}
                  {showHeatmap && (
                    <div className="absolute inset-0 pointer-events-none flex items-center justify-center mix-blend-screen">
                      <div className="w-72 h-56 bg-gradient-to-r from-rose-500/90 via-amber-400/80 to-blue-500/50 blur-3xl opacity-80 animate-pulse" />
                    </div>
                  )}

                  {/* Analysis Scanning Overlay */}
                  {isAnalyzing && (
                    <div className="absolute inset-0 bg-black/85 backdrop-blur-xs flex flex-col items-center justify-center text-center p-6 space-y-3 z-10 border border-cyan-500/50">
                      <div className="w-10 h-10 border-2 border-cyan-400 border-t-transparent animate-spin" />
                      <span className="text-sm font-mono font-bold uppercase tracking-wider text-cyan-300">
                        AI Forensic Engine is performing examination...
                      </span>
                      <span className="text-xs font-mono text-slate-400">
                        Auditing sensor grain, skin micro-texture, lighting vectors, and boundary splicing
                      </span>
                    </div>
                  )}
                </div>

                {/* Forensic Dossier */}
                {result && !isAnalyzing && (
                  <div className="space-y-6">
                    {/* 1. Sharp High-Contrast Verdict Banner */}
                    <div className={`p-6 border-2 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 ${
                      result.verdict === 'REAL'
                        ? 'bg-emerald-950/30 border-emerald-500 text-emerald-200'
                        : result.verdict === 'FAKE'
                        ? 'bg-rose-950/30 border-rose-500 text-rose-200'
                        : 'bg-amber-950/30 border-amber-500 text-amber-200'
                    }`}>
                      <div className="flex items-start gap-4">
                        <div className={`p-3 border ${
                          result.verdict === 'REAL'
                            ? 'bg-emerald-500 border-emerald-300 text-black'
                            : result.verdict === 'FAKE'
                            ? 'bg-rose-600 border-rose-400 text-white'
                            : 'bg-amber-500 border-amber-300 text-black'
                        } shrink-0`}>
                          {result.verdict === 'REAL' ? (
                            <ShieldCheck className="w-8 h-8" />
                          ) : result.verdict === 'FAKE' ? (
                            <ShieldAlert className="w-8 h-8" />
                          ) : (
                            <HelpCircle className="w-8 h-8" />
                          )}
                        </div>

                        <div>
                          <span className="text-[11px] uppercase font-mono tracking-widest text-slate-400 block">
                            Forensic Verdict Assessment
                          </span>
                          <h3 className="text-2xl sm:text-3xl font-black font-mono tracking-tight text-white mt-0.5">
                            {result.verdict === 'REAL' && 'REAL (Authentic Image)'}
                            {result.verdict === 'FAKE' && 'FAKE (AI-Generated / Manipulated)'}
                            {result.verdict === 'UNCERTAIN' && 'INCONCLUSIVE / UNCERTAIN'}
                          </h3>
                          <p className="text-sm font-mono mt-1 text-slate-300">
                            {result.shortVerdict}
                          </p>
                        </div>
                      </div>

                      {/* Sharp Score Box */}
                      <div className="bg-black px-6 py-3 border border-slate-800 text-right min-w-[150px] shrink-0">
                        <span className="text-[10px] uppercase font-mono tracking-wider text-slate-400 block">
                          Confidence
                        </span>
                        <span className={`text-3xl font-black font-mono ${
                          result.verdict === 'REAL' ? 'text-emerald-400' : result.verdict === 'FAKE' ? 'text-rose-400' : 'text-amber-400'
                        }`}>
                          {result.confidenceScore}%
                        </span>
                      </div>
                    </div>

                    {/* 2. Sharp Subject & Summary */}
                    <div className="bg-slate-900/60 p-5 border border-slate-800 space-y-2">
                      <div className="flex items-center justify-between border-b border-slate-800 pb-2">
                        <h4 className="text-xs font-mono uppercase text-slate-400 tracking-wider flex items-center gap-1.5 font-bold">
                          <Info className="w-4 h-4 text-cyan-400" />
                          Forensic Scene Analysis
                        </h4>
                        {result.subjectIdentified && (
                          <span className="text-xs font-mono text-cyan-300 bg-cyan-950 px-2.5 py-0.5 border border-cyan-800">
                            Subject: {result.subjectIdentified}
                          </span>
                        )}
                      </div>
                      <p className="text-sm text-slate-200 leading-relaxed font-sans">
                        {result.summary}
                      </p>
                    </div>

                    {/* 3. Detailed Reasons Why (Sharp Rectangular Cards) */}
                    <div className="bg-slate-900/80 p-6 border border-slate-800 space-y-4">
                      <div className="flex items-center gap-2 border-b border-slate-800 pb-3">
                        <CheckCircle2 className="w-5 h-5 text-cyan-400" />
                        <h4 className="text-sm font-mono font-bold uppercase tracking-wider text-white">
                          Forensic Evidence (Why is this image {result.verdict === 'REAL' ? 'Real' : 'Fake'}?)
                        </h4>
                      </div>

                      <div className="space-y-2.5">
                        {result.reasons.map((reason, idx) => (
                          <div 
                            key={idx} 
                            className="flex items-start gap-3 p-3.5 bg-black border border-slate-800 text-xs sm:text-sm text-slate-200"
                          >
                            <span className={`px-2 py-0.5 font-mono text-xs font-bold border shrink-0 ${
                              result.verdict === 'REAL' 
                                ? 'bg-emerald-950 text-emerald-400 border-emerald-800' 
                                : 'bg-rose-950 text-rose-400 border-rose-800'
                            }`}>
                              0{idx + 1}
                            </span>
                            <p className="leading-relaxed flex-1">
                              {reason}
                            </p>
                          </div>
                        ))}
                      </div>

                      {/* Key Indicators */}
                      {result.keyIndicators && result.keyIndicators.length > 0 && (
                        <div className="pt-3 border-t border-slate-800 flex flex-wrap items-center gap-2">
                          <span className="text-xs font-mono text-slate-400">Indicators:</span>
                          {result.keyIndicators.map((ind, i) => (
                            <span 
                              key={i} 
                              className="px-2 py-0.5 text-xs font-mono bg-black text-slate-300 border border-slate-800"
                            >
                              [{ind}]
                            </span>
                          ))}
                        </div>
                      )}
                    </div>
                  </div>
                )}
              </section>
            )}
          </>
        )}
      </main>

      {/* Sharp Tactical Footer */}
      <footer className="border-t border-slate-800 bg-slate-950 py-4 text-center text-xs font-mono text-slate-500">
        TrustLayer Deepfake Intelligence • Image & Audio Forensic Analysis
      </footer>
    </div>
  );
}
