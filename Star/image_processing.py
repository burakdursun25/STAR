from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.request import urlretrieve

import cv2
import mediapipe as mp
import numpy as np


BaseOptions = mp.tasks.BaseOptions
RunningMode = mp.tasks.vision.RunningMode
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions


POSE_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
HAND_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
FACE_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"


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
    pose_landmarks: List[LandmarkPoint]
    left_hand_landmarks: List[LandmarkPoint]
    right_hand_landmarks: List[LandmarkPoint]
    face_landmarks: List[LandmarkPoint]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pose_landmarks": [point.__dict__ for point in self.pose_landmarks],
            "left_hand_landmarks": [point.__dict__ for point in self.left_hand_landmarks],
            "right_hand_landmarks": [point.__dict__ for point in self.right_hand_landmarks],
            "face_landmarks": [point.__dict__ for point in self.face_landmarks],
            "has_mask": self.mask is not None,
        }


class ImageProcessor:
    def __init__(self, model_dir: Optional[Path] = None) -> None:
        self.model_dir = model_dir or Path(__file__).with_name("models")
        self.model_dir.mkdir(parents=True, exist_ok=True)

        pose_model = self._ensure_model("pose_landmarker_lite.task", POSE_MODEL_URL)
        hand_model = self._ensure_model("hand_landmarker.task", HAND_MODEL_URL)
        face_model = self._ensure_model("face_landmarker.task", FACE_MODEL_URL)

        self.pose = PoseLandmarker.create_from_options(
            PoseLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(pose_model)),
                running_mode=RunningMode.IMAGE,
                num_poses=1,
                output_segmentation_masks=True,
            )
        )
        self.hands = HandLandmarker.create_from_options(
            HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(hand_model)),
                running_mode=RunningMode.IMAGE,
                num_hands=2,
            )
        )
        self.face_landmarker = FaceLandmarker.create_from_options(
            FaceLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(face_model)),
                running_mode=RunningMode.IMAGE,
                num_faces=1,
            )
        )

    def _ensure_model(self, file_name: str, url: str) -> Path:
        destination = self.model_dir / file_name
        if not destination.exists():
            urlretrieve(url, destination)
        return destination

    @staticmethod
    def _extract_landmarks(landmarks: Optional[Any]) -> List[LandmarkPoint]:
        if not landmarks:
            return []
        points: List[LandmarkPoint] = []
        for landmark in landmarks:
            visibility = getattr(landmark, "visibility", None)
            points.append(
                LandmarkPoint(
                    x=float(landmark.x),
                    y=float(landmark.y),
                    z=float(landmark.z),
                    visibility=float(visibility) if visibility is not None else None,
                )
            )
        return points

    @staticmethod
    def _mask_from_segmentation(segmentation_mask: Any, threshold: float = 0.5) -> np.ndarray:
        if hasattr(segmentation_mask, "numpy_view"):
            mask_array = segmentation_mask.numpy_view()
        else:
            mask_array = np.asarray(segmentation_mask)
        return (mask_array > threshold).astype(np.uint8) * 255

    def process_frame(self, frame_bgr: np.ndarray) -> FrameResult:
        if frame_bgr is None or frame_bgr.size == 0:
            raise ValueError("Empty frame received")

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        pose_results = self.pose.detect(mp_image)
        hands_results = self.hands.detect(mp_image)
        face_results = self.face_landmarker.detect(mp_image)

        pose_landmarks = self._extract_landmarks(pose_results.pose_landmarks[0] if pose_results.pose_landmarks else None)
        face_landmarks = self._extract_landmarks(face_results.face_landmarks[0] if face_results.face_landmarks else None)

        left_hand_landmarks: List[LandmarkPoint] = []
        right_hand_landmarks: List[LandmarkPoint] = []
        if hands_results.hand_landmarks and hands_results.handedness:
            for hand_landmarks, handedness in zip(hands_results.hand_landmarks, hands_results.handedness):
                label = handedness[0].category_name.lower()
                extracted = self._extract_landmarks(hand_landmarks)
                if label == "left":
                    left_hand_landmarks = extracted
                elif label == "right":
                    right_hand_landmarks = extracted

        mask = None
        if pose_results.segmentation_masks:
            mask = self._mask_from_segmentation(pose_results.segmentation_masks[0])

        return FrameResult(
            frame_bgr=frame_bgr,
            mask=mask,
            pose_landmarks=pose_landmarks,
            left_hand_landmarks=left_hand_landmarks,
            right_hand_landmarks=right_hand_landmarks,
            face_landmarks=face_landmarks,
        )

    def close(self) -> None:
        self.pose.close()
        self.hands.close()
        self.face_landmarker.close()


def overlay_mask(frame_bgr: np.ndarray, mask: np.ndarray, color: Tuple[int, int, int] = (0, 255, 0), alpha: float = 0.35) -> np.ndarray:
    if mask is None:
        return frame_bgr

    overlay = frame_bgr.copy()
    colored = np.zeros_like(frame_bgr)
    colored[:, :] = color

    binary_mask = np.asarray(mask).astype(bool)
    if binary_mask.ndim == 3:
        binary_mask = np.squeeze(binary_mask, axis=-1)

    blended = cv2.addWeighted(frame_bgr, 1.0 - alpha, colored, alpha, 0)
    overlay[binary_mask] = blended[binary_mask]
    return overlay


def draw_pose(frame_bgr: np.ndarray, pose_landmarks: List[LandmarkPoint]) -> np.ndarray:
    if not pose_landmarks:
        return frame_bgr

    annotated = frame_bgr.copy()
    height, width = annotated.shape[:2]
    for point in pose_landmarks:
        cx = int(point.x * width)
        cy = int(point.y * height)
        cv2.circle(annotated, (cx, cy), 3, (0, 255, 255), -1)
    return annotated


def run_webcam(index: int = 0) -> None:
    processor = ImageProcessor()
    capture = cv2.VideoCapture(index)

    if not capture.isOpened():
        processor.close()
        raise RuntimeError("Webcam could not be opened")

    try:
        while True:
            success, frame = capture.read()
            if not success:
                break

            result = processor.process_frame(frame)
            output = frame
            if result.mask is not None:
                output = overlay_mask(output, result.mask)
            output = draw_pose(output, result.pose_landmarks)

            cv2.imshow("Image Processing Preview", output)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
    finally:
        capture.release()
        processor.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run_webcam()
