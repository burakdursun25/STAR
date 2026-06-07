"""İki kameradan gelen iskelet verilerini güven ağırlıklı olarak birleştirir."""
from __future__ import annotations

import time
from typing import Dict, List, Optional

import numpy as np

from .types import BoneData, SkeletonFrame

# Güven eşiği: bu değerin altındaki kemik verileri göz ardı edilir
CONFIDENCE_THRESHOLD = 0.3


class DualCameraFusion:
    """
    İki farklı açıdan bakılan kameradan gelen SkeletonFrame verilerini
    güven skoruna göre ağırlıklı ortalama ile birleştirir.

    Strateji:
      • Yalnızca bir kamera bir kemiği görüyorsa → o kameranın değeri kullanılır.
      • Her iki kamera da görüyorsa → güven ağırlıklı ağırlıklı ortalama alınır.
      • Hiçbiri görmüyorsa → o kemik sonuç veride yer almaz.
    """

    @staticmethod
    def _weighted_avg_bone(a: BoneData, b: BoneData) -> BoneData:
        wa = a.confidence
        wb = b.confidence
        total = wa + wb
        if total < 1e-9:
            return a
        w_a, w_b = wa / total, wb / total
        pos = [a.position[i] * w_a + b.position[i] * w_b for i in range(3)]
        # Yön vektörlerini ağırlıklı ortala, sonra normalize et
        raw = [a.direction[i] * w_a + b.direction[i] * w_b for i in range(3)]
        n = (raw[0]**2 + raw[1]**2 + raw[2]**2) ** 0.5
        direction = [v / n for v in raw] if n > 1e-9 else [0.0, 1.0, 0.0]
        conf = max(a.confidence, b.confidence)
        return BoneData(name=a.name, position=pos, direction=direction, confidence=conf)

    @staticmethod
    def _merge_dicts(
        dict_a: Dict[str, BoneData],
        dict_b: Dict[str, BoneData],
    ) -> Dict[str, BoneData]:
        merged: Dict[str, BoneData] = {}
        all_keys = set(dict_a) | set(dict_b)
        for key in all_keys:
            bone_a = dict_a.get(key)
            bone_b = dict_b.get(key)
            if bone_a is None:
                if bone_b and bone_b.confidence >= CONFIDENCE_THRESHOLD:
                    merged[key] = bone_b
            elif bone_b is None:
                if bone_a.confidence >= CONFIDENCE_THRESHOLD:
                    merged[key] = bone_a
            else:
                if (bone_a.confidence >= CONFIDENCE_THRESHOLD or
                        bone_b.confidence >= CONFIDENCE_THRESHOLD):
                    merged[key] = DualCameraFusion._weighted_avg_bone(bone_a, bone_b)
        return merged

    def fuse(
        self,
        frame_a: Optional[SkeletonFrame],
        frame_b: Optional[SkeletonFrame],
    ) -> Optional[SkeletonFrame]:
        """
        İki frame'i birleştirir.  İkisi de None ise None döner.
        Biri None ise diğeri olduğu gibi döner.
        """
        if frame_a is None and frame_b is None:
            return None
        if frame_a is None:
            return frame_b
        if frame_b is None:
            return frame_a

        return SkeletonFrame(
            timestamp=time.time(),
            frame_number=max(frame_a.frame_number, frame_b.frame_number),
            camera_id=-1,   # -1 = fused
            bones=self._merge_dicts(frame_a.bones, frame_b.bones),
            left_hand=self._merge_dicts(frame_a.left_hand, frame_b.left_hand),
            right_hand=self._merge_dicts(frame_a.right_hand, frame_b.right_hand),
        )
