export function cleanGatewayUrl(
  rawUrl: string,
  fallback = ""
): string {
  if (!rawUrl || !rawUrl.trim()) return fallback;
  let cleaned = rawUrl.trim().replace(/\/+$/, "");
  if (cleaned.endsWith("/v1")) {
    cleaned = cleaned.slice(0, -3);
  }
  return cleaned;
}

export function fileToDataUri(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result));
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
}

function encodeAudioBufferToWavBlob(audioBuffer: AudioBuffer): Blob {
  const numChannels = 1;
  const sampleRate = audioBuffer.sampleRate;
  const samples = audioBuffer.getChannelData(0);
  const bytesPerSample = 2; // 16-bit PCM
  const blockAlign = numChannels * bytesPerSample;
  const dataSize = samples.length * bytesPerSample;
  const buffer = new ArrayBuffer(44 + dataSize);
  const view = new DataView(buffer);

  const writeString = (offset: number, str: string) => {
    for (let i = 0; i < str.length; i++) {
      view.setUint8(offset + i, str.charCodeAt(i));
    }
  };

  writeString(0, "RIFF");
  view.setUint32(4, 36 + dataSize, true);
  writeString(8, "WAVE");
  writeString(12, "fmt ");
  view.setUint32(16, 16, true); // Subchunk1Size (PCM)
  view.setUint16(20, 1, true); // AudioFormat = 1 (PCM)
  view.setUint16(22, numChannels, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * blockAlign, true);
  view.setUint16(32, blockAlign, true);
  view.setUint16(34, bytesPerSample * 8, true);
  writeString(36, "data");
  view.setUint32(40, dataSize, true);

  let offset = 44;
  for (let i = 0; i < samples.length; i++, offset += 2) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true);
  }

  return new Blob([buffer], { type: "audio/wav" });
}

/**
 * Normalizes any browser-decodable audio file (.mp3, .m4a, .webm, .ogg, .aac, .flac, .wav)
 * into a clean 16-bit PCM Mono WAV file so backend libsndfile/soundfile never chokes
 * on container headers, ID3 tags, or non-WAV extensions.
 */
export async function normalizeAudioFileToWav(
  file: File,
  targetSampleRate = 16000
): Promise<File> {
  if (typeof window === "undefined") return file;
  try {
    const arrayBuffer = await file.arrayBuffer();
    const AudioContextClass =
      window.AudioContext ||
      (window as unknown as { webkitAudioContext: typeof AudioContext })
        .webkitAudioContext;
    const audioCtx = new AudioContextClass();
    try {
      const decoded = await audioCtx.decodeAudioData(arrayBuffer.slice(0));
      const lengthInSamples = Math.max(
        1,
        Math.ceil(decoded.duration * targetSampleRate)
      );
      const offlineCtx = new OfflineAudioContext(
        1,
        lengthInSamples,
        targetSampleRate
      );
      const source = offlineCtx.createBufferSource();
      source.buffer = decoded;
      source.connect(offlineCtx.destination);
      source.start(0);
      const rendered = await offlineCtx.startRendering();
      const wavBlob = encodeAudioBufferToWavBlob(rendered);
      const baseName = file.name.replace(/\.[^/.]+$/, "") || "audio";
      return new File([wavBlob], `${baseName}.wav`, { type: "audio/wav" });
    } finally {
      await audioCtx.close().catch(() => {});
    }
  } catch {
    // Fallback to original file if Web Audio API cannot decode
    return file;
  }
}

export async function renderWaveformToCanvas(
  blob: Blob,
  canvas: HTMLCanvasElement | null
): Promise<void> {
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  if (!ctx) return;

  ctx.clearRect(0, 0, canvas.width, canvas.height);

  try {
    const arrayBuffer = await blob.arrayBuffer();
    const AudioContextClass =
      window.AudioContext ||
      (window as unknown as { webkitAudioContext: typeof AudioContext })
        .webkitAudioContext;
    const audioCtx = new AudioContextClass();
    const audioBuffer = await audioCtx.decodeAudioData(arrayBuffer);
    const channelData = audioBuffer.getChannelData(0);

    const bars = 96;
    const blockSize = Math.max(1, Math.floor(channelData.length / bars));
    const barWidth = canvas.width / bars;

    ctx.fillStyle = "#111111";
    for (let i = 0; i < bars; i++) {
      let sum = 0;
      for (let j = 0; j < blockSize; j++) {
        sum += Math.abs(channelData[i * blockSize + j] || 0);
      }
      const average = sum / blockSize;
      const barHeight = Math.max(
        4,
        Math.min(canvas.height - 8, average * canvas.height * 2.8)
      );
      const x = i * barWidth;
      const y = (canvas.height - barHeight) / 2;
      ctx.fillRect(x + 1, y, Math.max(2, barWidth - 2), barHeight);
    }
    await audioCtx.close();
  } catch {
    // Fallback decorative bar when browser cannot decode raw format
    ctx.fillStyle = "#787774";
    for (let i = 0; i < 48; i++) {
      const h = 8 + ((i * 17) % 36);
      ctx.fillRect(i * 9 + 4, (canvas.height - h) / 2, 5, h);
    }
  }
}
