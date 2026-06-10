"""
Landmark noktalarından Blender iskelet verisi üretir.

Koordinat sistemi dönüşümü
--------------------------
MediaPipe WORLD landmarks (kalça merkezi = origin, metre):
  x → sağ (kameranın sağı = kişinin solu)
  y → aşağı (image y ile aynı yön)
  z → kameraya doğru (pozitif = kameraya yakın)

Blender (Z-up, Y-derinlik):
  x → sağ        → MP.x
  y → ileri/derinlik → -MP.z
  z → yukarı      → -MP.y

Dönüşüm formülü: blender(x, y, z) = mp(lm.x, -lm.z, -lm.y)

El (normalize):  y aşağı, z derinlik → aynı formül
"""
from __future__ import annotations

import time
from typing import Dict, List, Tuple

import numpy as np

from .types import BoneData, FrameResult, LandmarkPoint, SkeletonFrame

# ── Pose bone pairs ───────────────────────────────────────────────────────────
# (kemik_adı, başlangıç_landmark_idx, bitiş_landmark_idx)
POSE_BONE_PAIRS: List[Tuple[str, int, int]] = [
    ("spine",       24, 12),   # sağ kalça → sağ omuz
    ("chest",       12, 11),   # sağ omuz → sol omuz
    ("neck",        11,  0),   # sol omuz → burun
    ("upper_arm.L", 12, 14),   # sağ omuz → sağ dirsek  (kamera sağı = kişi solu)
    ("forearm.L",   14, 16),   # sağ dirsek → sağ bilek
    ("upper_arm.R", 11, 13),   # sol omuz → sol dirsek
    ("forearm.R",   13, 15),   # sol dirsek → sol bilek
    ("thigh.L",     24, 26),   # sağ kalça → sağ diz
    ("shin.L",      26, 28),   # sağ diz → sağ ayak bileği
    ("foot.L",      28, 32),   # sağ ayak bileği → sağ ayak parmağı
    ("thigh.R",     23, 25),   # sol kalça → sol diz
    ("shin.R",      25, 27),   # sol diz → sol ayak bileği
    ("foot.R",      27, 31),   # sol ayak bileği → sol ayak parmağı
]

HAND_BONE_PAIRS: List[Tuple[str, int, int]] = [
    ("palm",      0,  9),
    ("thumb.1",   1,  2), ("thumb.2",  2,  3), ("thumb.3",  3,  4),
    ("index.1",   5,  6), ("index.2",  6,  7), ("index.3",  7,  8),
    ("middle.1",  9, 10), ("middle.2", 10, 11), ("middle.3", 11, 12),
    ("ring.1",   13, 14), ("ring.2",  14, 15), ("ring.3",  15, 16),
    ("pinky.1",  17, 18), ("pinky.2", 18, 19), ("pinky.3", 19, 20),
]


def _to_blender(lm: LandmarkPoint) -> np.ndarray:
    """
    MediaPipe koordinatı → Blender Z-up dünya koordinatı.
    World landmark için (mp.x, -mp.z, -mp.y).
    Normalize landmark için de aynı formül çalışır (z ekseni küçük ama tutarlı).
    """
    return np.array([lm.x, -lm.z, -lm.y], dtype=np.float64)


def _unit(v: np.ndarray) -> List[float]:
    n = np.linalg.norm(v)
    if n < 1e-6:
        return [0.0, 1.0, 0.0]   # fallback: yukarı yön
    return (v / n).tolist()


def _confidence(lm: LandmarkPoint) -> float:
    v = lm.visibility
    return float(v) if v is not None else 1.0


class SkeletonBuilder:
    """
    FrameResult → SkeletonFrame.

    Gövde kemikleri için pose_world_landmarks kullanır (gerçek 3D metre).
    El kemikleri için normalize hand_landmarks kullanır (world yok).
    Her kemik için Blender dünya uzayında yön vektörü (direction) hesaplar.
    Euler açısı hesaplanmaz — rotasyonu Blender'ın kendi mathutils'i yapar.
    """

    def _pose_bones(self, landmarks: List[LandmarkPoint]) -> Dict[str, BoneData]:
        bones: Dict[str, BoneData] = {}
        if not landmarks:
            return bones

        for bone_name, s_idx, e_idx in POSE_BONE_PAIRS:
            if s_idx >= len(landmarks) or e_idx >= len(landmarks):
                continue
            lm_s = landmarks[s_idx]
            lm_e = landmarks[e_idx]
            s_bl = _to_blender(lm_s)
            e_bl = _to_blender(lm_e)
            direction = _unit(e_bl - s_bl)
            conf = min(_confidence(lm_s), _confidence(lm_e))
            bones[bone_name] = BoneData(
                name=bone_name,
                position=s_bl.tolist(),
                direction=direction,
                confidence=conf,
            )

        # Hips: sol + sağ kalça ortası
        if len(landmarks) > 24:
            l_h = landmarks[23]
            r_h = landmarks[24]
            mid = (_to_blender(l_h) + _to_blender(r_h)) * 0.5
            # Hips yön vektörü = yukarı (world +Z)
            bones["hips"] = BoneData(
                name="hips",
                position=mid.tolist(),
                direction=[0.0, 0.0, 1.0],
                confidence=min(_confidence(l_h), _confidence(r_h)),
            )
        return bones

    def _hand_bones(self, landmarks: List[LandmarkPoint]) -> Dict[str, BoneData]:
        bones: Dict[str, BoneData] = {}
        if not landmarks:
            return bones
        for bone_name, s_idx, e_idx in HAND_BONE_PAIRS:
            if s_idx >= len(landmarks) or e_idx >= len(landmarks):
                continue
            lm_s = landmarks[s_idx]
            lm_e = landmarks[e_idx]
            s_bl = _to_blender(lm_s)
            e_bl = _to_blender(lm_e)
            direction = _unit(e_bl - s_bl)
            bones[bone_name] = BoneData(
                name=bone_name,
                position=s_bl.tolist(),
                direction=direction,
                confidence=1.0,
            )
        return bones

    def build(
        self,
        result: FrameResult,
        frame_number: int,
        camera_id: int,
    ) -> SkeletonFrame:
        # World landmarks varsa kullan, yoksa normalize'e geri dön
        pose_src = result.world_pose_landmarks or result.pose_landmarks
        return SkeletonFrame(
            timestamp=time.time(),
            frame_number=frame_number,
            camera_id=camera_id,
            bones=self._pose_bones(pose_src),
            left_hand=self._hand_bones(result.left_hand_landmarks),
            right_hand=self._hand_bones(result.right_hand_landmarks),
        )
