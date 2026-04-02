from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


@dataclass(frozen=True)
class MasteringSettings:
    target_lufs: float = -14.0
    warmth: int = 35
    brightness: int = 35
    punch: int = 45
    low_eq: float = 0.0
    mid_eq: float = 0.0
    high_eq: float = 0.0
    compression: float = 2.0

    def normalized(self) -> "MasteringSettings":
        return MasteringSettings(
            target_lufs=round(clamp(float(self.target_lufs), -16.0, -7.0), 2),
            warmth=int(round(clamp(float(self.warmth), 0.0, 100.0))),
            brightness=int(round(clamp(float(self.brightness), 0.0, 100.0))),
            punch=int(round(clamp(float(self.punch), 0.0, 100.0))),
            low_eq=round(clamp(float(self.low_eq), -6.0, 6.0), 2),
            mid_eq=round(clamp(float(self.mid_eq), -6.0, 6.0), 2),
            high_eq=round(clamp(float(self.high_eq), -6.0, 6.0), 2),
            compression=round(clamp(float(self.compression), 1.0, 4.0), 2),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
