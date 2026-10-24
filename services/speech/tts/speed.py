"""
services/speech/tts/speed.py
Time-stretching and speed adjustment for synthesized audio.
"""
import io
import math
import struct
import wave


def adjust_audio_speed(wav_bytes: bytes, speed: float) -> bytes:
    """
    Adjusts the speed of WAV audio bytes while preserving WAV integrity.
    Supports speeds between 0.25x and 4.0x.
    """
    if speed is None or abs(float(speed) - 1.0) < 0.01:
        return wav_bytes

    eff_speed = max(0.25, min(4.0, float(speed)))

    try:
        with wave.open(io.BytesIO(wav_bytes), "rb") as wf_in:
            nchannels = wf_in.getnchannels()
            sampwidth = wf_in.getsampwidth()
            framerate = wf_in.getframerate()
            nframes = wf_in.getnframes()
            frames = wf_in.readframes(nframes)

        # Decode 16-bit PCM frames
        if sampwidth == 2:
            num_samples = nframes * nchannels
            fmt = f"<{num_samples}h"
            samples = list(struct.unpack(fmt, frames))

            # Resample / time-stretch by linear interpolation
            target_num_samples = max(1, int(num_samples / eff_speed))
            target_samples = []
            for i in range(target_num_samples):
                src_pos = i * eff_speed
                idx0 = int(src_pos)
                idx1 = min(idx0 + 1, num_samples - 1)
                frac = src_pos - idx0
                interp = int((1.0 - frac) * samples[idx0] + frac * samples[idx1])
                target_samples.append(max(-32768, min(32767, interp)))

            out_frames = struct.pack(f"<{len(target_samples)}h", *target_samples)

            out_io = io.BytesIO()
            with wave.open(out_io, "wb") as wf_out:
                wf_out.setnchannels(nchannels)
                wf_out.setsampwidth(sampwidth)
                wf_out.setframerate(framerate)
                wf_out.writeframes(out_frames)

            return out_io.getvalue()
    except Exception:
        pass

    return wav_bytes
