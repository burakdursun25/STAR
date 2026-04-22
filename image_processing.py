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


POSE_KEYPOINTS: Dict[str, int] = {
    "nose": 0,
    "left_eye": 2,
    "right_eye": 5,
    "left_ear": 7,
    "right_ear": 8,
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_elbow": 13,
    "right_elbow": 14,
    "left_wrist": 15,
    "right_wrist": 16,
    "left_hip": 23,
    "right_hip": 24,
    "left_knee": 25,
    "right_knee": 26,
    "left_ankle": 27,
    "right_ankle": 28,
}


HAND_KEYPOINTS: Dict[str, int] = {
    "wrist": 0,
    "thumb_tip": 4,
    "index_tip": 8,
    "middle_tip": 12,
    "ring_tip": 16,
    "pinky_tip": 20,
}


FACE_KEYPOINTS: Dict[str, int] = {
    "forehead": 10,
    "left_eye_outer": 33,
    "right_eye_outer": 263,
    "nose_tip": 1,
    "mouth_left": 61,
    "mouth_right": 291,
    "chin": 152,
}


FINGER_TIP_KEYS: Tuple[str, ...] = ("thumb_tip", "index_tip", "middle_tip", "ring_tip", "pinky_tip")


def _format_selected_lines(name: str, landmarks: List[LandmarkPoint], selected_points: Dict[str, int]) -> List[str]:
    lines = [f"{name}:"]
    if not landmarks:
        lines.append("  not detected")
        return lines

    for label, index in selected_points.items():
        if index >= len(landmarks):
            lines.append(f"  {label:<14} missing")
            continue

        point = landmarks[index]
        if point.visibility is None:
            lines.append(f"  {label:<14} x={point.x:.4f} y={point.y:.4f} z={point.z:.4f}")
        else:
            lines.append(
                f"  {label:<14} x={point.x:.4f} y={point.y:.4f} z={point.z:.4f} vis={point.visibility:.4f}"
            )
    return lines


def _movement_axis(delta: float, threshold: float, negative_label: str, positive_label: str) -> str:
    if delta > threshold:
        return positive_label
    if delta < -threshold:
        return negative_label
    return "steady"


def _format_hand_motion_lines(
    hand_name: str,
    current_landmarks: List[LandmarkPoint],
    previous_landmarks: Optional[List[LandmarkPoint]],
) -> List[str]:
    lines = [f"{hand_name} Finger Motions:"]
    if not current_landmarks:
        lines.append("  not detected")
        return lines

    if not previous_landmarks:
        lines.append("  calibrating... keep your hand visible")
        return lines

    movement_threshold = 0.004
    for key in FINGER_TIP_KEYS:
        idx = HAND_KEYPOINTS[key]
        if idx >= len(current_landmarks) or idx >= len(previous_landmarks):
            lines.append(f"  {key:<14} missing")
            continue

        current = current_landmarks[idx]
        previous = previous_landmarks[idx]
        dx = current.x - previous.x
        dy = current.y - previous.y
        dz = current.z - previous.z
        movement = (dx * dx + dy * dy + dz * dz) ** 0.5

        x_dir = _movement_axis(dx, movement_threshold, "left", "right")
        y_dir = _movement_axis(dy, movement_threshold, "up", "down")
        z_dir = _movement_axis(dz, movement_threshold, "forward", "back")
        state = "moving" if movement > movement_threshold else "stable"
        lines.append(
            f"  {key:<14} {state:<6} d={movement:.4f} x={x_dir:<6} y={y_dir:<6} z={z_dir:<7}"
        )

    thumb_idx = HAND_KEYPOINTS["thumb_tip"]
    index_idx = HAND_KEYPOINTS["index_tip"]
    if (
        thumb_idx < len(current_landmarks)
        and index_idx < len(current_landmarks)
        and thumb_idx < len(previous_landmarks)
        and index_idx < len(previous_landmarks)
    ):
        current_pinch = float(
            np.linalg.norm(
                [
                    current_landmarks[thumb_idx].x - current_landmarks[index_idx].x,
                    current_landmarks[thumb_idx].y - current_landmarks[index_idx].y,
                    current_landmarks[thumb_idx].z - current_landmarks[index_idx].z,
                ]
            )
        )
        previous_pinch = float(
            np.linalg.norm(
                [
                    previous_landmarks[thumb_idx].x - previous_landmarks[index_idx].x,
                    previous_landmarks[thumb_idx].y - previous_landmarks[index_idx].y,
                    previous_landmarks[thumb_idx].z - previous_landmarks[index_idx].z,
                ]
            )
        )
        pinch_delta = current_pinch - previous_pinch
        if pinch_delta < -0.003:
            pinch_state = "closing"
        elif pinch_delta > 0.003:
            pinch_state = "opening"
        else:
            pinch_state = "steady"
        lines.append(f"  pinch(thumb-index) {pinch_state} dist={current_pinch:.4f}")

    return lines


def print_landmarks_live(
    result: FrameResult,
    previous_left_hand: Optional[List[LandmarkPoint]] = None,
    previous_right_hand: Optional[List[LandmarkPoint]] = None,
) -> None:
    sections: List[str] = []
    sections.extend(_format_selected_lines("Body Keypoints", result.pose_landmarks, POSE_KEYPOINTS))
    sections.append("")
    sections.extend(_format_selected_lines("Head Keypoints", result.face_landmarks, FACE_KEYPOINTS))
    sections.append("")
    sections.extend(_format_selected_lines("Left Hand Keypoints", result.left_hand_landmarks, HAND_KEYPOINTS))
    sections.append("")
    sections.extend(_format_selected_lines("Right Hand Keypoints", result.right_hand_landmarks, HAND_KEYPOINTS))
    sections.append("")
    sections.extend(_format_hand_motion_lines("Left Hand", result.left_hand_landmarks, previous_left_hand))
    sections.append("")
    sections.extend(_format_hand_motion_lines("Right Hand", result.right_hand_landmarks, previous_right_hand))

    # Clear terminal and rewrite full frame data so values appear as live-updating.
    print("\x1b[2J\x1b[H" + "\n".join(sections), end="", flush=True)


def run_webcam(index: int = 0) -> None:
    processor = ImageProcessor()
    capture = cv2.VideoCapture(index)
    previous_left_hand: Optional[List[LandmarkPoint]] = None
    previous_right_hand: Optional[List[LandmarkPoint]] = None

    if not capture.isOpened():
        processor.close()
        raise RuntimeError("Webcam could not be opened")

    try:
        while True:
            success, frame = capture.read()
            if not success:
                break

            result = processor.process_frame(frame)
            print_landmarks_live(result, previous_left_hand, previous_right_hand)
            previous_left_hand = result.left_hand_landmarks
            previous_right_hand = result.right_hand_landmarks
            output = frame
            if result.mask is not None:
                output = overlay_mask(output, result.mask)
            output = draw_pose(output, result.pose_landmarks)

            cv2.imshow("Image Processing Preview", output)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q")):
                break
    finally:
        capture.release()
        processor.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run_webcam()
