/**
 * Client-Side Audio Forensics Engine
 * Computes RMS energy, zero-crossing rate, duration, and sample rate from AudioBuffers.
 */

export interface AudioForensicsMetadata {
  duration_seconds: number;
  sample_rate: number;
  channels: number;
  file_size_kb: number;
  rms_energy: number;
  zero_crossing_rate: number;
}

export interface TrustLayerAudioResult {
  modality: 'audio';
  file: string;
  prediction: 'HumanVoice' | 'AIVoice';
  confidence: number;
  metadata: {
    duration_seconds: number;
    sample_rate: number;
    channels: number;
    file_size_kb: number;
  };
  forensic_features: {
    rms_energy: number;
    zero_crossing_rate: number;
  };
  evidence: string[];
  reasons: string[];
  speechCharacteristics?: {
    breathingPresent: boolean;
    pitchVariability: string;
    vocoderBuzzing: string;
    speakerDescription: string;
  };
  status: 'NO STRONG SYNTHETIC SIGNAL DETECTED' | 'REQUIRES REVIEW — SYNTHETIC SIGNAL' | 'INCONCLUSIVE';
}

/**
 * Extracts acoustic forensic features using Web Audio API AudioBuffer
 */
export async function extractAcousticFeatures(
  arrayBuffer: ArrayBuffer,
  fileSizeBytes: number
): Promise<AudioForensicsMetadata> {
  const AudioContextClass = window.AudioContext || (window as any).webkitAudioContext;
  const audioCtx = new AudioContextClass();

  try {
    const audioBuffer = await audioCtx.decodeAudioData(arrayBuffer.slice(0));
    const channelData = audioBuffer.getChannelData(0);
    const n = channelData.length;

    // 1. RMS Energy: sqrt(mean(x^2))
    let sumSquares = 0;
    for (let i = 0; i < n; i++) {
      sumSquares += channelData[i] * channelData[i];
    }
    const rms = Math.sqrt(sumSquares / Math.max(1, n));

    // 2. Zero Crossing Rate: count sign changes / N
    let zeroCrossings = 0;
    for (let i = 1; i < n; i++) {
      if ((channelData[i] >= 0 && channelData[i - 1] < 0) || (channelData[i] < 0 && channelData[i - 1] >= 0)) {
        zeroCrossings++;
      }
    }
    const zcr = zeroCrossings / Math.max(1, n);

    return {
      duration_seconds: parseFloat(audioBuffer.duration.toFixed(2)),
      sample_rate: audioBuffer.sampleRate,
      channels: audioBuffer.numberOfChannels,
      file_size_kb: parseFloat((fileSizeBytes / 1024).toFixed(2)),
      rms_energy: parseFloat(rms.toFixed(6)),
      zero_crossing_rate: parseFloat(zcr.toFixed(6)),
    };
  } finally {
    audioCtx.close().catch(() => {});
  }
}
