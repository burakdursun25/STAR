"""MediaPipe Pose / Hand / Face işleme sınıfı."""
from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional
from urllib.request import urlretrieve

import cv2
import mediapipe as mp
import numpy as np

from .types import FrameResult, LandmarkPoint

# ── MediaPipe kısaltmaları ────────────────────────────────────────────────────
BaseOptions         = mp.tasks.BaseOptions
RunningMode         = mp.tasks.vision.RunningMode
PoseLandmarker      = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOpts  = mp.tasks.vision.PoseLandmarkerOptions
HandLandmarker      = mp.tasks.vision.HandLandmarker
HandLandmarkerOpts  = mp.tasks.vision.HandLandmarkerOptions
FaceLandmarker      = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOpts  = mp.tasks.vision.FaceLandmarkerOptions

_MODEL_URLS = {
    "pose_landmarker_lite.task": (
        "https://storage.googleapis.com/mediapipe-models/"
        "pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
    ),
    "hand_landmarker.task": (
        "https://storage.googleapis.com/mediapipe-models/"
        "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
    ),
    "face_landmarker.task": (
        "https://storage.googleapis.com/mediapipe-models/"
        "face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
    ),
}


class ImageProcessor:
    """
    Tek bir BGR kareden pose, el ve yüz landmark'larını çıkarır.
    Her CameraCapture için ayrı bir ImageProcessor oluşturun.
    """

    def __init__(self, model_dir: Optional[Path] = None) -> None:
        self._model_dir = model_dir or Path(__file__).with_name("models")
        self._model_dir.mkdir(parents=True, exist_ok=True)

        pose_path = self._ensure_model("pose_landmarker_lite.task")
        hand_path = self._ensure_model("hand_landmarker.task")
        face_path = self._ensure_model("face_landmarker.task")

        self._pose = PoseLandmarker.create_from_options(
            PoseLandmarkerOpts(
                base_options=BaseOptions(model_asset_path=str(pose_path)),
                running_mode=RunningMode.IMAGE,
                num_poses=1,
                output_segmentation_masks=True,
            )
        )
        self._hands = HandLandmarker.create_from_options(
            HandLandmarkerOpts(
                base_options=BaseOptions(model_asset_path=str(hand_path)),
                running_mode=RunningMode.IMAGE,
                num_hands=2,
            )
        )
        self._face = FaceLandmarker.create_from_options(
            FaceLandmarkerOpts(
                base_options=BaseOptions(model_asset_path=str(face_path)),
                running_mode=RunningMode.IMAGE,
                num_faces=1,
            )
        )

    # ── Model indirme ─────────────────────────────────────────────────────────

    def _ensure_model(self, file_name: str) -> Path:
        path = self._model_dir / file_name
        if not path.exists():
            print(f"[ImageProcessor] Model indiriliyor: {file_name}")
            urlretrieve(_MODEL_URLS[file_name], path)
        return path

    # ── Landmark yardımcıları ─────────────────────────────────────────────────

    @staticmethod
    def _extract(landmarks: Optional[Any]) -> List[LandmarkPoint]:
        if not landmarks:
            return []
        result: List[LandmarkPoint] = []
        for lm in landmarks:
            v = getattr(lm, "visibility", None)
            result.append(
                LandmarkPoint(
                    x=float(lm.x),
                    y=float(lm.y),
                    z=float(lm.z),
                    visibility=float(v) if v is not None else None,
                )
            )
        return result

    # ── Ana işlem metodu ──────────────────────────────────────────────────────

    def process_frame(self, frame_bgr: np.ndarray) -> FrameResult:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        pose_result = self._pose.detect(mp_img)
        hand_result = self._hands.detect(mp_img)
        face_result = self._face.detect(mp_img)

        # Segmentasyon maskesi
        mask: Optional[np.ndarray] = None
        if pose_result.segmentation_masks:
            raw = pose_result.segmentation_masks[0].numpy_view()
            mask = (np.squeeze(raw) * 255).astype(np.uint8)  # her zaman 2D

        # Normalize landmark listeleri (ekran koordinatı)
        pose_lms: List[LandmarkPoint] = []
        if pose_result.pose_landmarks:
            pose_lms = self._extract(pose_result.pose_landmarks[0])

        # World landmark listesi (metre, kalça merkezli 3D dünya koordinatı)
        world_pose_lms: List[LandmarkPoint] = []
        if pose_result.pose_world_landmarks:
            world_pose_lms = self._extract(pose_result.pose_world_landmarks[0])

        left_hand: List[LandmarkPoint] = []
        right_hand: List[LandmarkPoint] = []
        for idx, hand_lms in enumerate(hand_result.hand_landmarks):
            category = hand_result.handedness[idx][0].category_name
            pts = self._extract(hand_lms)
            if category == "Left":
                left_hand = pts
            else:
                right_hand = pts

        face_lms: List[LandmarkPoint] = []
        if face_result.face_landmarks:
            face_lms = self._extract(face_result.face_landmarks[0])

        return FrameResult(
            frame_bgr=frame_bgr,
            mask=mask,
            pose_landmarks=pose_lms,
            world_pose_landmarks=world_pose_lms,
            left_hand_landmarks=left_hand,
            right_hand_landmarks=right_hand,
            face_landmarks=face_lms,
        )

    def close(self) -> None:
        self._pose.close()
        self._hands.close()
        self._face.close()

    # ── Çizim yardımcıları ────────────────────────────────────────────────────

    @staticmethod
    def draw_pose(frame: np.ndarray, landmarks: List[LandmarkPoint]) -> np.ndarray:
        if not landmarks:
            return frame
        h, w = frame.shape[:2]
        connections = [
            (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
            (11, 23), (12, 24), (23, 24), (23, 25), (24, 26),
            (25, 27), (26, 28), (27, 31), (28, 32),
        ]
        for s, e in connections:
            if s < len(landmarks) and e < len(landmarks):
                pt1 = (int(landmarks[s].x * w), int(landmarks[s].y * h))
                pt2 = (int(landmarks[e].x * w), int(landmarks[e].y * h))
                cv2.line(frame, pt1, pt2, (0, 255, 0), 2)
        for lm in landmarks:
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (cx, cy), 4, (255, 0, 0), -1)
        return frame

    @staticmethod
    def overlay_mask(frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
        color = np.zeros_like(frame)
        color[:, :] = (0, 120, 255)
        alpha = (mask[:, :, np.newaxis] / 255.0) * 0.35
        return (frame * (1 - alpha) + color * alpha).astype(np.uint8)
