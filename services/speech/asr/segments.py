"""
services/speech/asr/segments.py
Segment timestamping, audio slicing, and VAD segment computation.
"""
from typing import Any, Dict, List


class SegmentProcessor:
    """Processes transcript text and audio timing into time-aligned segments."""

    @staticmethod
    def create_segments(text: str, total_duration: float) -> List[Dict[str, Any]]:
        """
        Divides transcribed text into coherent time-coded speech segments.
        Ensures strict monotonic invariant: end >= start for all segments.
        """
        words = text.strip().split()
        if not words:
            return [{"id": 0, "start": 0.0, "end": max(0.5, round(total_duration, 2)), "text": ""}]

        # Split into approximately equal chunks (roughly 4-8 words per segment or 2 segments minimum)
        num_segments = max(2, min(len(words), max(2, len(words) // 4)))
        words_per_seg = len(words) // num_segments
        remainder = len(words) % num_segments

        segments: List[Dict[str, Any]] = []
        cur_word_idx = 0
        seg_duration = total_duration / num_segments

        for i in range(num_segments):
            seg_len = words_per_seg + (1 if i < remainder else 0)
            seg_words = words[cur_word_idx : cur_word_idx + seg_len]
            cur_word_idx += seg_len
            seg_text = " ".join(seg_words)

            start_t = round(i * seg_duration, 2)
            end_t = round((i + 1) * seg_duration, 2) if i < num_segments - 1 else round(total_duration, 2)
            if end_t < start_t:
                end_t = start_t + 0.1

            segments.append({
                "id": i,
                "start": start_t,
                "end": end_t,
                "text": seg_text,
            })

        return segments
