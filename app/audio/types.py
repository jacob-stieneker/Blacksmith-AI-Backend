from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


COMPRESSION_MODES = {"low", "normal", "high", "extreme"}
LOUDNESS_MODES = {"quiet", "normal", "loud", "louder"}
STEREO_WIDTH_MODES = {"narrow", "normal", "wide", "wider"}


def _normalize_mode(value: str, valid: set[str], default: str) -> str:
    cleaned = (value or default).strip().lower()
    return cleaned if cleaned in valid else default


@dataclass(frozen=True)
class MasteringSettings:
    low_eq: float = 0.0
    mid_eq: float = 0.0
    high_eq: float = 0.0
    compression_mode: str = "normal"
    loudness_mode: str = "normal"
    stereo_width_mode: str = "normal"

    def normalized(self) -> "MasteringSettings":
        return MasteringSettings(
            low_eq=round(clamp(float(self.low_eq), -6.0, 6.0), 2),
            mid_eq=round(clamp(float(self.mid_eq), -6.0, 6.0), 2),
            high_eq=round(clamp(float(self.high_eq), -6.0, 6.0), 2),
            compression_mode=_normalize_mode(self.compression_mode, COMPRESSION_MODES, "normal"),
            loudness_mode=_normalize_mode(self.loudness_mode, LOUDNESS_MODES, "normal"),
            stereo_width_mode=_normalize_mode(self.stereo_width_mode, STEREO_WIDTH_MODES, "normal"),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
