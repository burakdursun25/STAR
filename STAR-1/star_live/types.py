"""Paylaşılan veri sınıfları — tüm modüller buradan import eder."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


@dataclass
class LandmarkPoint:
    x: float
    y: float
    z: float
    visibility: Optional[float] = None


@dataclass
class FrameResult:
    frame_bgr: np.ndarray
    mask: Optional[np.ndarray]
    pose_landmarks: List[LandmarkPoint]        # normalize (ekran) koord.
    world_pose_landmarks: List[LandmarkPoint]  # metre cinsinden 3D dünya koord.
    left_hand_landmarks: List[LandmarkPoint]
    right_hand_landmarks: List[LandmarkPoint]
    face_landmarks: List[LandmarkPoint]


@dataclass
class BoneData:
    name: str
    position: List[float]   # kemiğin başı (Blender dünya koord.)
    direction: List[float]  # normalize yön vektörü (Blender dünya koord.)
    confidence: float = 1.0


@dataclass
class SkeletonFrame:
    timestamp: float
    frame_number: int
    camera_id: int
    bones: Dict[str, BoneData] = field(default_factory=dict)
    left_hand: Dict[str, BoneData] = field(default_factory=dict)
    right_hand: Dict[str, BoneData] = field(default_factory=dict)

    def to_dict(self) -> dict:
        def _bone_dict(bone: BoneData) -> dict:
            return {
                "position":   bone.position,
                "direction":  bone.direction,
                "confidence": bone.confidence,
            }
        return {
            "timestamp":  self.timestamp,
            "frame":      self.frame_number,
            "camera_id":  self.camera_id,
            "skeleton":   {k: _bone_dict(v) for k, v in self.bones.items()},
            "left_hand":  {k: _bone_dict(v) for k, v in self.left_hand.items()},
            "right_hand": {k: _bone_dict(v) for k, v in self.right_hand.items()},
            "bones":      list(self.bones.keys()),
        }
