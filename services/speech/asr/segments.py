"""
Deprecated timestamp-estimation shim.

Real timestamps must be returned by the ASR/forced-alignment backend. Factory
does not divide audio duration evenly across words or segments.
"""
from factory_contracts.errors import ModelUnavailableError


class SegmentProcessor:
    @staticmethod
    def create_segments(text: str, total_duration: float):
        raise ModelUnavailableError(
            message=(
                "Synthetic timestamp estimation is disabled. "
                "Use native ASR timestamps or the configured forced-alignment backend."
            )
        )
