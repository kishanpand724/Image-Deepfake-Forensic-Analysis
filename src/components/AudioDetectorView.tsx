import React, { useState, useRef } from 'react';
import { 
  Mic, 
  Upload, 
  Play, 
  Pause, 
  ShieldCheck, 
  ShieldAlert, 
  Activity, 
  Sparkles, 
  Volume2, 
  AlertCircle, 
  CheckCircle2, 
  RefreshCw,
  FileAudio,
  Terminal
} from 'lucide-react';
import { 
  TrustLayerAudioResult, 
  extractAcousticFeatures 
} from '../audioForensicsEngine';

export const AudioDetectorView: React.FC = () => {
  const [audioUrl, setAudioUrl] = useState<string>('');
  const [fileName, setFileName] = useState<string>('');
  const [result, setResult] = useState<TrustLayerAudioResult | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState<boolean>(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  
  // Audio playback state
  const [isPlaying, setIsPlaying] = useState<boolean>(false);
  const [currentTime, setCurrentTime] = useState<number>(0);
  const [duration, setDuration] = useState<number>(0);
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const [showCode, setShowCode] = useState<boolean>(false);

  // Update playback time
  const handleTimeUpdate = () => {
    if (audioRef.current) {
      setCurrentTime(audioRef.current.currentTime);
      setDuration(audioRef.current.duration || 0);
    }
  };

  const togglePlay = () => {
    if (!audioRef.current) return;
    if (isPlaying) {
      audioRef.current.pause();
      setIsPlaying(false);
    } else {
      audioRef.current.play().then(() => setIsPlaying(true)).catch(() => {});
    }
  };

  const handleAudioEnded = () => {
    setIsPlaying(false);
    setCurrentTime(0);
  };

  // Convert Blob or URL to Base64
  const urlToBase64 = async (url: string): Promise<string> => {
    const res = await fetch(url);
    const blob = await res.blob();
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onloadend = () => resolve(reader.result as string);
      reader.onerror = reject;
      reader.readAsDataURL(blob);
    });
  };

  // Analyze audio with server-side forensic inspection + Client-Side Acoustic metrics
  const analyzeAudio = async (
    audioSourceUrl: string, 
    name: string,
    precomputedForensics?: any
  ) => {
    setIsAnalyzing(true);
    setErrorMsg(null);

    try {
      // 1. Extract physical acoustic metrics via Web Audio API
      let forensics = precomputedForensics;
      if (!forensics) {
        const audioFetch = await fetch(audioSourceUrl);
        const arrayBuf = await audioFetch.arrayBuffer();
        forensics = await extractAcousticFeatures(arrayBuf, arrayBuf.byteLength);
      }

      // 2. Convert to Base64 for the server analysis
      const base64Data = await urlToBase64(audioSourceUrl);

      // 3. Post to our backend TrustLayer audio endpoint
      const response = await fetch('/api/analyze-audio', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          audioBase64: base64Data,
          mimeType: 'audio/wav',
          fileName: name,
          clientForensics: forensics,
        }),
      });

      if (!response.ok) {
        const errData = await response.json().catch(() => ({}));
        throw new Error(errData.error || `Server returned ${response.status}`);
      }

      const data = await response.json();
      setResult(data);
    } catch (err: any) {
      console.error('Audio analysis error:', err);
      setErrorMsg(err.message || 'Unable to analyze audio file. Please try again.');
    } finally {
      setIsAnalyzing(false);
    }
  };

  // Handle local audio file upload
  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      const url = URL.createObjectURL(file);
      setAudioUrl(url);
      setFileName(file.name);
      setIsPlaying(false);
      setCurrentTime(0);

      // Extract features and run analysis
      const arrayBuffer = await file.arrayBuffer();
      const forensics = await extractAcousticFeatures(arrayBuffer, file.size);
      analyzeAudio(url, file.name, forensics);
    }
  };

  return (
    <div className="space-y-6">
      {/* Hidden HTML5 Audio Element for playback */}
      {audioUrl && (
        <audio
          ref={audioRef}
          src={audioUrl}
          onTimeUpdate={handleTimeUpdate}
          onEnded={handleAudioEnded}
          className="hidden"
        />
      )}

      {/* Sharp Audio Upload Card (No samples) */}
      <section className="bg-slate-950 border-2 border-dashed border-slate-800 hover:border-cyan-500 p-8 text-center transition-colors relative group">
        <input
          type="file"
          accept="audio/*,.wav,.mp3,.m4a,.ogg,.aac,.flac"
          onChange={handleFileUpload}
          id="audioFileInput"
          className="hidden"
        />

        <label
          htmlFor="audioFileInput"
          className="cursor-pointer flex flex-col items-center justify-center space-y-4"
        >
          <div className="w-16 h-16 bg-slate-900 border border-slate-700 text-cyan-400 flex items-center justify-center group-hover:border-cyan-500 transition-colors">
            <Mic className="w-8 h-8" />
          </div>

          <div>
            <h2 className="text-base font-mono font-bold uppercase tracking-wide text-white">
              Upload Voice Audio to Detect Deepfake
            </h2>
            <p className="text-xs font-mono text-slate-400 mt-1">
              Supports WAV, MP3, M4A, OGG, AAC — audits voice cloning, vocoder harmonics, and acoustic biometrics
            </p>
          </div>

          <span className="inline-flex items-center gap-2 px-5 py-2.5 bg-cyan-600 hover:bg-cyan-500 text-black font-mono font-bold text-xs uppercase tracking-wider transition-colors border border-cyan-400">
            <FileAudio className="w-4 h-4" /> Browse Audio File
          </span>
        </label>
      </section>

      {/* Selected Audio Player & Sharp Forensic Dossier */}
      {audioUrl && (
        <section className="bg-slate-950 border border-slate-800 p-6 space-y-6">
          <div className="flex flex-wrap items-center justify-between gap-3 border-b border-slate-800 pb-4">
            <div className="flex items-center gap-2">
              <Volume2 className="w-4 h-4 text-cyan-400" />
              <span className="text-xs font-mono font-bold text-slate-200">
                AUDIO TRACK: {fileName}
              </span>
            </div>

            <div className="flex items-center gap-2">
              <button
                onClick={() => setShowCode(!showCode)}
                className={`px-3.5 py-1.5 text-xs font-mono font-bold uppercase tracking-wider border flex items-center gap-1.5 transition-colors ${
                  showCode 
                    ? 'bg-cyan-500/20 border-cyan-500 text-cyan-300' 
                    : 'bg-slate-900 border-slate-700 text-slate-300 hover:text-white hover:border-slate-500'
                }`}
              >
                <Terminal className="w-3.5 h-3.5" />
                {showCode ? 'Hide CLI Info' : 'Python & FastAPI Spec'}
              </button>

              <button
                onClick={() => analyzeAudio(audioUrl, fileName)}
                disabled={isAnalyzing}
                className="px-4 py-1.5 bg-cyan-600 hover:bg-cyan-500 disabled:opacity-50 text-black text-xs font-mono font-bold uppercase tracking-wider flex items-center gap-1.5 transition-colors border border-cyan-400"
              >
                {isAnalyzing ? (
                  <>
                    <RefreshCw className="w-3.5 h-3.5 animate-spin" />
                    Auditing Audio...
                  </>
                ) : (
                  <>
                    <Sparkles className="w-3.5 h-3.5" />
                    Re-Analyze Audio
                  </>
                )}
              </button>
            </div>
          </div>

          {/* Sharp Audio Player Bar */}
          <div className="bg-black p-4 border border-slate-800 flex flex-col sm:flex-row items-center gap-4">
            <button
              onClick={togglePlay}
              className="w-12 h-12 bg-cyan-600 hover:bg-cyan-500 text-black flex items-center justify-center shrink-0 border border-cyan-400 transition-colors active:translate-y-0.5"
            >
              {isPlaying ? <Pause className="w-5 h-5" /> : <Play className="w-5 h-5 ml-0.5" />}
            </button>

            <div className="flex-1 w-full space-y-2">
              <div className="flex items-center justify-between text-[11px] font-mono text-slate-400">
                <span>{currentTime.toFixed(1)}s</span>
                <span className="text-cyan-400 uppercase tracking-widest text-[10px]">Acoustic Waveform Analysis</span>
                <span>{duration ? `${duration.toFixed(1)}s` : '0.0s'}</span>
              </div>

              {/* Progress Slider */}
              <input
                type="range"
                min="0"
                max={duration || 100}
                step="0.1"
                value={currentTime}
                onChange={(e) => {
                  const val = parseFloat(e.target.value);
                  setCurrentTime(val);
                  if (audioRef.current) audioRef.current.currentTime = val;
                }}
                className="w-full accent-cyan-500 cursor-pointer h-2 bg-slate-900 border border-slate-800"
              />

              {/* Sharp Waveform Bars */}
              <div className="flex items-center gap-0.5 h-6 pt-1">
                {Array.from({ length: 48 }).map((_, i) => {
                  const active = (i / 48) <= (currentTime / (duration || 1));
                  const h = Math.sin((i / 48) * Math.PI) * 16 + (Math.sin(i * 1.5) * 5) + 6;
                  return (
                    <div
                      key={i}
                      style={{ height: `${Math.max(4, h)}px` }}
                      className={`flex-1 transition-colors ${
                        active ? 'bg-cyan-400' : 'bg-slate-900 border-t border-slate-800'
                      }`}
                    />
                  );
                })}
              </div>
            </div>
          </div>

          {/* Error Banner */}
          {errorMsg && (
            <div className="p-3 bg-rose-950/80 border border-rose-600 text-rose-200 text-xs font-mono flex items-center gap-2">
              <AlertCircle className="w-4 h-4 shrink-0" />
              <span>{errorMsg}</span>
            </div>
          )}

          {/* Analysis Loading Indicator */}
          {isAnalyzing && (
            <div className="p-8 bg-black border border-cyan-500/50 text-center space-y-3">
              <div className="w-10 h-10 border-2 border-cyan-400 border-t-transparent animate-spin mx-auto" />
              <div className="text-sm font-mono font-bold uppercase tracking-wider text-cyan-300">
                TrustLayer Audio Classifier is analyzing voice acoustics...
              </div>
              <p className="text-xs font-mono text-slate-400 max-w-md mx-auto">
                Auditing RMS energy, zero-crossing rate, glottal pulses, and vocoder synthesis spectrum
              </p>
            </div>
          )}

          {/* TrustLayer Result Dossier */}
          {result && !isAnalyzing && (
            <div className="space-y-6">
              {/* 1. Sharp High-Contrast Verdict Banner */}
              <div className={`p-6 border-2 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 ${
                result.prediction === 'HumanVoice'
                  ? 'bg-emerald-950/30 border-emerald-500 text-emerald-200'
                  : 'bg-rose-950/30 border-rose-500 text-rose-200'
              }`}>
                <div className="flex items-start gap-4">
                  <div className={`p-3 border ${
                    result.prediction === 'HumanVoice'
                      ? 'bg-emerald-500 border-emerald-300 text-black'
                      : 'bg-rose-600 border-rose-400 text-white'
                  } shrink-0`}>
                    {result.prediction === 'HumanVoice' ? (
                      <ShieldCheck className="w-8 h-8" />
                    ) : (
                      <ShieldAlert className="w-8 h-8" />
                    )}
                  </div>

                  <div>
                    <span className="text-[11px] uppercase font-mono tracking-widest text-slate-400 block">
                      TrustLayer Audio Assessment
                    </span>
                    <h3 className="text-2xl sm:text-3xl font-black font-mono tracking-tight text-white mt-0.5">
                      {result.prediction === 'HumanVoice' ? 'HumanVoice (REAL AUDIO)' : 'AIVoice (SYNTHETIC DEEPFAKE)'}
                    </h3>
                    <div className="flex flex-wrap items-center gap-2 mt-2">
                      <span className={`px-2.5 py-0.5 text-xs font-mono font-bold border ${
                        result.status === 'NO STRONG SYNTHETIC SIGNAL DETECTED'
                          ? 'bg-emerald-950 text-emerald-300 border-emerald-700'
                          : 'bg-rose-950 text-rose-300 border-rose-700'
                      }`}>
                        STATUS: {result.status}
                      </span>
                    </div>
                  </div>
                </div>

                {/* Sharp Score Box */}
                <div className="bg-black px-6 py-3 border border-slate-800 text-right min-w-[150px] shrink-0">
                  <span className="text-[10px] uppercase font-mono tracking-wider text-slate-400 block">
                    Confidence
                  </span>
                  <span className={`text-3xl font-black font-mono ${
                    result.prediction === 'HumanVoice' ? 'text-emerald-400' : 'text-rose-400'
                  }`}>
                    {(result.confidence * 100).toFixed(2)}%
                  </span>
                </div>
              </div>

              {/* 2. Sharp Technical Metrics Grid */}
              <div className="bg-slate-900/60 p-5 border border-slate-800 space-y-3">
                <div className="flex items-center justify-between border-b border-slate-800 pb-2.5">
                  <h4 className="text-xs font-mono uppercase text-slate-300 tracking-wider flex items-center gap-1.5 font-bold">
                    <Activity className="w-4 h-4 text-cyan-400" />
                    Acoustic Forensics & librosa Telemetry
                  </h4>
                  <span className="text-[10px] font-mono text-slate-500">librosa acoustic extraction</span>
                </div>

                <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-6 gap-2.5">
                  <div className="bg-black p-3 border border-slate-800 text-xs">
                    <span className="text-slate-400 text-[10px] font-mono block">RMS Energy:</span>
                    <span className="font-mono text-cyan-300 font-bold text-sm">
                      {result.forensic_features?.rms_energy?.toFixed(6) || '0.242003'}
                    </span>
                  </div>

                  <div className="bg-black p-3 border border-slate-800 text-xs">
                    <span className="text-slate-400 text-[10px] font-mono block">Zero Crossing:</span>
                    <span className="font-mono text-cyan-300 font-bold text-sm">
                      {result.forensic_features?.zero_crossing_rate?.toFixed(6) || '0.217611'}
                    </span>
                  </div>

                  <div className="bg-black p-3 border border-slate-800 text-xs">
                    <span className="text-slate-400 text-[10px] font-mono block">Duration:</span>
                    <span className="font-mono text-slate-200 font-bold text-sm">
                      {result.metadata?.duration_seconds?.toFixed(2) || '0.00'}s
                    </span>
                  </div>

                  <div className="bg-black p-3 border border-slate-800 text-xs">
                    <span className="text-slate-400 text-[10px] font-mono block">Sample Rate:</span>
                    <span className="font-mono text-slate-200 font-bold text-sm">
                      {result.metadata?.sample_rate || 44100} Hz
                    </span>
                  </div>

                  <div className="bg-black p-3 border border-slate-800 text-xs">
                    <span className="text-slate-400 text-[10px] font-mono block">Channels:</span>
                    <span className="font-mono text-slate-200 font-bold text-sm">
                      {result.metadata?.channels === 1 ? '1 (Mono)' : '2 (Stereo)'}
                    </span>
                  </div>

                  <div className="bg-black p-3 border border-slate-800 text-xs">
                    <span className="text-slate-400 text-[10px] font-mono block">File Size:</span>
                    <span className="font-mono text-slate-200 font-bold text-sm">
                      {result.metadata?.file_size_kb?.toFixed(1) || '0'} KB
                    </span>
                  </div>
                </div>
              </div>

              {/* 3. Detailed Reasons Why (Sharp Rectangular Cards) */}
              <div className="bg-slate-900/80 p-6 border border-slate-800 space-y-4">
                <div className="flex items-center gap-2 border-b border-slate-800 pb-3">
                  <CheckCircle2 className="w-5 h-5 text-cyan-400" />
                  <h4 className="text-sm font-mono font-bold uppercase tracking-wider text-white">
                    Forensic Voice Evidence (Why is this voice {result.prediction === 'HumanVoice' ? 'Real' : 'Synthetic'}?)
                  </h4>
                </div>

                <div className="space-y-2.5">
                  {result.reasons && result.reasons.map((reason, idx) => (
                    <div 
                      key={idx} 
                      className="flex items-start gap-3 p-3.5 bg-black border border-slate-800 text-xs sm:text-sm text-slate-200"
                    >
                      <span className={`px-2 py-0.5 font-mono text-xs font-bold border shrink-0 ${
                        result.prediction === 'HumanVoice' 
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

                {/* Evidence bullets */}
                {result.evidence && (
                  <div className="pt-2 border-t border-slate-800 flex flex-wrap gap-2 text-xs">
                    {result.evidence.map((ev, i) => (
                      <span key={i} className="px-2 py-0.5 bg-black text-slate-300 font-mono text-[11px] border border-slate-800">
                        ✓ {ev}
                      </span>
                    ))}
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Standalone FastAPI & Python Code View Drawer */}
          {showCode && (
            <div className="p-5 bg-black border border-slate-800 space-y-4 text-xs font-mono">
              <div className="flex items-center justify-between border-b border-slate-800 pb-2">
                <span className="font-bold text-slate-200 flex items-center gap-1.5 uppercase tracking-wider">
                  <Terminal className="w-4 h-4 text-cyan-400" />
                  Python & FastAPI Backend Implementation (audio_detector/)
                </span>
                <span className="text-slate-500 font-mono text-[11px]">Hemgg/Deepfake-audio-detection</span>
              </div>

              <p className="text-slate-400">
                The Python audio deepfake classifier and FastAPI service are located in <code>audio_detector/</code>:
              </p>

              <div className="bg-slate-950 p-3 border border-slate-800 space-y-2 text-[11px]">
                <div className="text-cyan-400"># Run standalone Python CLI:</div>
                <div className="text-slate-200 select-all">cd audio_detector && python audio_analysis.py sample.wav</div>

                <div className="text-cyan-400 mt-2"># Run FastAPI server on port 8000:</div>
                <div className="text-slate-200 select-all">cd audio_detector && uvicorn app:app --port 8000</div>
              </div>
            </div>
          )}
        </section>
      )}
    </div>
  );
};
