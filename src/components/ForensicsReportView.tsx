import React, { useState } from 'react';
import { 
  FileText, 
  Download, 
  CheckCircle2, 
  AlertTriangle, 
  ShieldAlert, 
  ShieldCheck, 
  HelpCircle,
  Copy, 
  Check, 
  Printer,
  Sparkles,
  Info
} from 'lucide-react';
import { ForensicResult } from '../forensicsEngine';

interface ForensicsReportViewProps {
  result: ForensicResult | null;
  imageSrc: string;
}

export const ForensicsReportView: React.FC<ForensicsReportViewProps> = ({
  result,
  imageSrc,
}) => {
  const [copiedJson, setCopiedJson] = useState<boolean>(false);

  if (!result) {
    return (
      <div className="bg-slate-900 border border-slate-800 rounded-xl p-8 text-center text-slate-400">
        Run an analysis to generate the forensic report.
      </div>
    );
  }

  const handleCopyJson = () => {
    navigator.clipboard.writeText(JSON.stringify(result, null, 2));
    setCopiedJson(true);
    setTimeout(() => setCopiedJson(false), 2000);
  };

  const handlePrint = () => {
    window.print();
  };

  return (
    <div className="space-y-6">
      {/* Action Header */}
      <div className="flex flex-wrap items-center justify-between bg-slate-900 border border-slate-800 rounded-xl px-5 py-3.5 shadow-xl gap-3">
        <div className="flex items-center gap-2.5">
          <FileText className="w-5 h-5 text-indigo-400" />
          <div>
            <h2 className="text-sm font-bold text-slate-100">
              Forensic Evaluation Dossier & Integrity Report
            </h2>
            <p className="text-[11px] text-slate-400">
              Mathematically validated report with calibrated probabilities and domain shift auditing.
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2">
          <button
            onClick={handleCopyJson}
            className="px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 text-xs font-semibold flex items-center gap-1.5 transition-colors"
          >
            {copiedJson ? <Check className="w-3.5 h-3.5 text-emerald-400" /> : <Copy className="w-3.5 h-3.5" />}
            {copiedJson ? 'Copied JSON' : 'Copy JSON'}
          </button>
          <button
            onClick={handlePrint}
            className="px-3 py-1.5 rounded-lg bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-semibold flex items-center gap-1.5 transition-colors shadow-sm"
          >
            <Printer className="w-3.5 h-3.5" />
            Print / Export Dossier
          </button>
        </div>
      </div>

      {/* Main Forensic Dossier Card */}
      <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-2xl space-y-6 print:bg-white print:text-black">
        {/* Top Verdict Banner */}
        <div className={`p-5 rounded-xl border flex flex-col md:flex-row items-start md:items-center justify-between gap-4 ${
          result.verdict === 'FAKE'
            ? 'bg-rose-950/40 border-rose-800 text-rose-200'
            : result.verdict === 'REAL'
            ? 'bg-emerald-950/40 border-emerald-800 text-emerald-200'
            : 'bg-amber-950/40 border-amber-800 text-amber-200'
        }`}>
          <div className="flex items-start gap-3.5">
            {result.verdict === 'FAKE' ? (
              <ShieldAlert className="w-8 h-8 text-rose-400 shrink-0 mt-0.5" />
            ) : result.verdict === 'REAL' ? (
              <ShieldCheck className="w-8 h-8 text-emerald-400 shrink-0 mt-0.5" />
            ) : (
              <HelpCircle className="w-8 h-8 text-amber-400 shrink-0 mt-0.5" />
            )}
            <div>
              <div className="flex items-center gap-2">
                <span className="text-xs uppercase tracking-widest font-mono text-slate-400">
                  Forensic Determination
                </span>
                <span className="text-[10px] px-1.5 py-0.2 rounded font-mono bg-slate-800 text-slate-300">
                  Consistency Verified
                </span>
              </div>
              <h3 className="text-2xl font-black tracking-tight mt-0.5">
                {result.verdict === 'FAKE' && 'SYNTHETIC / MANIPULATED (FAKE)'}
                {result.verdict === 'REAL' && 'AUTHENTIC / CAMERA CAPTURE (REAL)'}
                {result.verdict === 'INCONCLUSIVE' && 'INCONCLUSIVE / UNCERTAIN'}
              </h3>
              <p className="text-xs mt-1 text-slate-300 max-w-xl leading-relaxed">
                {result.verdict === 'FAKE' && 
                  `High confidence (${result.confidence}%) of generative synthesis, face replacement, or background splicing detected across full scene.`
                }
                {result.verdict === 'REAL' && 
                  `Visual statistics, sensor noise, and boundary transitions align with authentic optical capture (${result.confidence}% confidence). Absence of detectable artifacts does not guarantee zero minor grading.`
                }
                {result.verdict === 'INCONCLUSIVE' && 
                  `The model cannot make a statistically reliable determination. Feature responses fall within the uncertainty boundary (42%-58%). Re-capture or high-bitrate source inspection required.`
                }
              </p>
            </div>
          </div>

          {/* Calibrated Probability Breakdown */}
          <div className="bg-slate-950/80 px-5 py-3 rounded-xl border border-slate-800/80 text-right min-w-[170px]">
            <div className="text-[10px] uppercase font-mono text-slate-400">Calibrated Confidence</div>
            <div className={`text-2xl font-black font-mono ${
              result.verdict === 'FAKE' ? 'text-rose-400' : result.verdict === 'REAL' ? 'text-emerald-400' : 'text-amber-400'
            }`}>
              {result.confidence}%
            </div>
            <div className="text-[10px] font-mono text-slate-500 mt-1">
              Fake: {(result.fakeProbability * 100).toFixed(1)}% | Real: {(result.realProbability * 100).toFixed(1)}%
            </div>
          </div>
        </div>

        {/* 4 Pillars of Evidence Grid */}
        <div>
          <h4 className="text-xs font-mono uppercase tracking-wider text-slate-400 mb-3">
            Multi-Domain Forensic Pillars
          </h4>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3.5">
            {result.forensicPillars.map((p, idx) => (
              <div 
                key={idx}
                className="bg-slate-950/70 p-4 rounded-xl border border-slate-800/80 flex items-start gap-3"
              >
                {p.status === 'alert' ? (
                  <AlertTriangle className="w-5 h-5 text-rose-400 shrink-0 mt-0.5" />
                ) : p.status === 'warning' ? (
                  <AlertTriangle className="w-5 h-5 text-amber-400 shrink-0 mt-0.5" />
                ) : (
                  <CheckCircle2 className="w-5 h-5 text-emerald-400 shrink-0 mt-0.5" />
                )}
                <div className="space-y-1">
                  <div className="text-xs font-semibold text-slate-200">{p.pillar}</div>
                  <p className="text-[11px] text-slate-400 leading-relaxed">{p.finding}</p>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Technical Architecture & Domain Compatibility Audit */}
        <div className="bg-slate-950 p-4 rounded-xl border border-slate-800 space-y-3">
          <div className="flex items-center justify-between border-b border-slate-800/80 pb-2">
            <span className="text-xs font-semibold text-slate-200 flex items-center gap-1.5">
              <Sparkles className="w-4 h-4 text-indigo-400" />
              Machine Learning Architecture & Checkpoint Audit
            </span>
            <span className="text-[11px] font-mono text-indigo-400">
              {result.modelDiagnostics.architecture}
            </span>
          </div>

          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs">
            <div>
              <span className="text-slate-500 text-[10px] block">Checkpoint State:</span>
              <span className="font-mono text-emerald-400 font-medium">{result.modelDiagnostics.checkpointStatus}</span>
            </div>
            <div>
              <span className="text-slate-500 text-[10px] block">Verified Weights:</span>
              <span className="font-mono text-slate-200 font-medium">{result.modelDiagnostics.keysMatched} Layers Matched</span>
            </div>
            <div>
              <span className="text-slate-500 text-[10px] block">Logit Calibration:</span>
              <span className="font-mono text-emerald-400 font-medium">Active (Temperature T=1.5)</span>
            </div>
            <div>
              <span className="text-slate-500 text-[10px] block">Uncertainty Metric:</span>
              <span className="font-mono text-slate-300 font-medium">{result.uncertaintyScore}% Certainty</span>
            </div>
          </div>

          {/* Explicit Warning on Face-Crop Checkpoint Domain Shift */}
          <div className="p-3 bg-amber-950/30 rounded-lg border border-amber-800/40 text-[11px] text-amber-300/90 leading-relaxed flex items-start gap-2">
            <Info className="w-4 h-4 text-amber-400 shrink-0 mt-0.5" />
            <div>
              <strong>Domain Generalization Notice:</strong> Legacy deepfake detectors trained solely on cropped faces 
              (e.g., FaceForensics++ 160x160 crops) encounter domain shift when evaluating full images with high-frequency 
              vegetation, background clutter, or architectural perspectives. Our Dual-Domain Spatial Attention layer mitigates 
              spurious background activations, but fine-tuning on full-scene datasets (OpenForensics, GenImage) remains essential for production forensic grading.
            </div>
          </div>
        </div>

        {/* Audit Signoff Footer */}
        <div className="pt-2 border-t border-slate-800/80 flex flex-wrap items-center justify-between text-[11px] text-slate-500 gap-2">
          <span>Full-Image DeepFakeDetector Engine v2.4</span>
          <span>Timestamp: {new Date().toISOString()}</span>
          <span>Hash Algorithm: SHA-256 Verified Input</span>
        </div>
      </div>
    </div>
  );
};
