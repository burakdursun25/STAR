"""
One Euro Filter — kemik yön vektörleri için adaptif smoothing.

Neden One Euro?
  • EMA (sabit alpha): yavaş harekette iyi ama hızlı harekette gecikir.
  • One Euro: anlık hıza göre alpha'yı otomatik ayarlar.
    – Yavaş/durağan → daha fazla smooth  (titreşim yok)
    – Hızlı hareket → daha az smooth     (gecikme yok)

Parametreler:
  min_cutoff  : düşük → daha agresif smooth (varsayılan: 1.5 Hz)
  beta        : yüksek → hızlı harekette daha az gecikme (varsayılan: 0.3)
"""
from __future__ import annotations

import math
import time
from typing import Dict, List, Optional

from .types import BoneData, SkeletonFrame


# ─────────────────────────────────────────────────────────────────────────────
# One Euro Filter — tek eksen için
# ─────────────────────────────────────────────────────────────────────────────
class _OneEuro:
    def __init__(self, min_cutoff: float, beta: float, d_cutoff: float = 1.0) -> None:
        self.min_cutoff = min_cutoff
        self.beta       = beta
        self.d_cutoff   = d_cutoff
        self._x: Optional[float] = None
        self._dx: float = 0.0
        self._t: Optional[float] = None

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x: float, t: float) -> float:
        if self._x is None:
            self._x = x
            self._t = t
            return x

        dt = max(t - self._t, 1e-6)
        self._t = t

        # Türev smooth
        dx_raw = (x - self._x) / dt
        a_d    = self._alpha(self.d_cutoff, dt)
        self._dx = a_d * dx_raw + (1.0 - a_d) * self._dx

        # Adaptif kesim frekansı
        cutoff = self.min_cutoff + self.beta * abs(self._dx)
        a      = self._alpha(cutoff, dt)
        self._x = a * x + (1.0 - a) * self._x
        return self._x


# ─────────────────────────────────────────────────────────────────────────────
# Tek kemik için — 3 eksen (x, y, z) ayrı ayrı filtre
# ─────────────────────────────────────────────────────────────────────────────
class _BoneSmoother:
    def __init__(self, min_cutoff: float, beta: float) -> None:
        self._filters = [_OneEuro(min_cutoff, beta) for _ in range(3)]

    def smooth(self, direction: List[float], t: float) -> List[float]:
        raw = [f(v, t) for f, v in zip(self._filters, direction)]
        # Normalize et (EMA sonrası birim uzunluk bozulur)
        n = math.sqrt(sum(v * v for v in raw))
        if n < 1e-6:
            return [0.0, 1.0, 0.0]
        return [v / n for v in raw]


# ─────────────────────────────────────────────────────────────────────────────
# Tüm iskelet için smooth
# ─────────────────────────────────────────────────────────────────────────────
class SkeletonSmoother:
    """
    SkeletonFrame içindeki tüm kemik yön vektörlerine One Euro Filter uygular.

    Kullanım:
        smoother = SkeletonSmoother(min_cutoff=1.5, beta=0.3)
        smoothed_frame = smoother.smooth(raw_frame)

    Ayar kılavuzu:
        min_cutoff  beta   Etki
        0.5         0.1    Çok yumuşak, biraz gecikir
        1.5         0.3    Dengeli  ← varsayılan
        3.0         0.8    Hafif smooth, düşük gecikme
    """

    def __init__(self, min_cutoff: float = 1.5, beta: float = 0.3) -> None:
        self._min_cutoff = min_cutoff
        self._beta       = beta
        self._bone_filters: Dict[str, _BoneSmoother] = {}

    def _get_filter(self, key: str) -> _BoneSmoother:
        if key not in self._bone_filters:
            self._bone_filters[key] = _BoneSmoother(self._min_cutoff, self._beta)
        return self._bone_filters[key]

    def _smooth_dict(
        self, bones: Dict[str, BoneData], t: float
    ) -> Dict[str, BoneData]:
        result: Dict[str, BoneData] = {}
        for name, bone in bones.items():
            smooth_dir = self._get_filter(name).smooth(bone.direction, t)
            result[name] = BoneData(
                name=bone.name,
                position=bone.position,
                direction=smooth_dir,
                confidence=bone.confidence,
            )
        return result

    def smooth(self, frame: SkeletonFrame) -> SkeletonFrame:
        t = frame.timestamp
        return SkeletonFrame(
            timestamp=frame.timestamp,
            frame_number=frame.frame_number,
            camera_id=frame.camera_id,
            bones=self._smooth_dict(frame.bones, t),
            left_hand=self._smooth_dict(frame.left_hand, t),
            right_hand=self._smooth_dict(frame.right_hand, t),
        )
