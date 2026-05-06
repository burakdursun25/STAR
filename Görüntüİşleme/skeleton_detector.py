from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.request import urlretrieve
from enum import Enum
import math

import cv2
import mediapipe as mp
import numpy as np
import time

BaseOptions = mp.tasks.BaseOptions
RunningMode = mp.tasks.vision.RunningMode
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions

POSE_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"

CAM_INDEX = 0


# ============================================================================
# DATA MODELS - ANA UZUVLAR VE KEMIKLER
# ============================================================================

@dataclass
class Vector3:
    """3D vektör"""
    x: float
    y: float
    z: float
    
    def to_list(self) -> List[float]:
        return [self.x, self.y, self.z]
    
    def __sub__(self, other: Vector3) -> Vector3:
        return Vector3(self.x - other.x, self.y - other.y, self.z - other.z)
    
    def __add__(self, other: Vector3) -> Vector3:
        return Vector3(self.x + other.x, self.y + other.y, self.z + other.z)
    
    def length(self) -> float:
        return math.sqrt(self.x**2 + self.y**2 + self.z**2)
    
    def normalize(self) -> Vector3:
        length = self.length()
        if length == 0:
            return Vector3(0, 0, 0)
        return Vector3(self.x / length, self.y / length, self.z / length)


@dataclass
class BoneTransform:
    """Kemik transformasyonu (konum ve rotasyon)"""
    name: str
    position: Vector3
    rotation: List[float]  # Euler angles [x, y, z] in radians
    scale: Vector3 = None
    
    def __post_init__(self):
        if self.scale is None:
            self.scale = Vector3(1, 1, 1)


@dataclass
class SkeletonFrame:
    """Tam iskelet frame'i (tüm kemiklerin pozisyonu)"""
    timestamp: float
    bones: List[BoneTransform]
    frame_number: int
    confidence: float


# ============================================================================
# KEYPOINT ve SKELETON KONFIGURASYONU
# ============================================================================

class PoseLandmarks(Enum):
    """MediaPipe Pose keypoint indeksleri"""
    NOSE = 0
    LEFT_EYE_INNER = 1
    LEFT_EYE = 2
    LEFT_EYE_OUTER = 3
    RIGHT_EYE_INNER = 4
    RIGHT_EYE = 5
    RIGHT_EYE_OUTER = 6
    LEFT_EAR = 7
    RIGHT_EAR = 8
    MOUTH_LEFT = 9
    MOUTH_RIGHT = 10
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    LEFT_PINKY = 17
    RIGHT_PINKY = 18
    LEFT_INDEX = 19
    RIGHT_INDEX = 20
    LEFT_THUMB = 21
    RIGHT_THUMB = 22
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_KNEE = 25
    RIGHT_KNEE = 26
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28
    LEFT_HEEL = 29
    RIGHT_HEEL = 30
    LEFT_FOOT_INDEX = 31
    RIGHT_FOOT_INDEX = 32


class SkeletonConfig:
    """İskelet (skeleton) yapısını tanımla - ana uzuvları"""
    
    # Kemik tanımları: (başlangıç keypoint, bitiş keypoint)
    BONE_PAIRS = [
        # Omuz ve göğüs
        (PoseLandmarks.LEFT_SHOULDER.value, PoseLandmarks.RIGHT_SHOULDER.value),
        
        # Sol kol
        (PoseLandmarks.LEFT_SHOULDER.value, PoseLandmarks.LEFT_ELBOW.value),
        (PoseLandmarks.LEFT_ELBOW.value, PoseLandmarks.LEFT_WRIST.value),
        
        # Sağ kol
        (PoseLandmarks.RIGHT_SHOULDER.value, PoseLandmarks.RIGHT_ELBOW.value),
        (PoseLandmarks.RIGHT_ELBOW.value, PoseLandmarks.RIGHT_WRIST.value),
        
        # Gövde
        (PoseLandmarks.LEFT_SHOULDER.value, PoseLandmarks.LEFT_HIP.value),
        (PoseLandmarks.RIGHT_SHOULDER.value, PoseLandmarks.RIGHT_HIP.value),
        (PoseLandmarks.LEFT_HIP.value, PoseLandmarks.RIGHT_HIP.value),
        
        # Sol bacak
        (PoseLandmarks.LEFT_HIP.value, PoseLandmarks.LEFT_KNEE.value),
        (PoseLandmarks.LEFT_KNEE.value, PoseLandmarks.LEFT_ANKLE.value),
        
        # Sağ bacak
        (PoseLandmarks.RIGHT_HIP.value, PoseLandmarks.RIGHT_KNEE.value),
        (PoseLandmarks.RIGHT_KNEE.value, PoseLandmarks.RIGHT_ANKLE.value),
    ]
    
    # Baş - tek nokta olarak (kafa)
    HEAD_POINT = PoseLandmarks.NOSE.value
    
    # Ek: Baş bağlantı kemikleri
    ADDITIONAL_BONES = [
        # Omuzların ortasından başa çizgi
        (PoseLandmarks.LEFT_SHOULDER.value, PoseLandmarks.NOSE.value),
        (PoseLandmarks.RIGHT_SHOULDER.value, PoseLandmarks.NOSE.value),
    ]


# ============================================================================
# ANA VERI SINIFI
# ============================================================================

@dataclass
class LandmarkPoint:
    """Tekil landmark noktası"""
    x: float
    y: float
    z: float
    visibility: Optional[float] = None
    
    def to_vector3(self) -> Vector3:
        return Vector3(self.x, self.y, self.z)


# ============================================================================
# İSKELET İŞLEME
# ============================================================================

class SkeletonProcessor:
    """Pose landmarks'ini iskelet kemiklerine dönüştür"""
    
    def __init__(self):
        self.frame_count = 0
        self.start_time = time.time()
    
    @staticmethod
    def get_bone_euler_angles(start: Vector3, end: Vector3) -> List[float]:
        """Kemik vektöründen Euler açılarını hesapla"""
        direction = end - start
        direction = direction.normalize()
        
        # Z-Y-X rotation order
        x_angle = math.atan2(direction.z, math.sqrt(direction.x**2 + direction.y**2))
        y_angle = math.atan2(-direction.x, direction.y)
        z_angle = 0
        
        return [x_angle, y_angle, z_angle]
    
    @staticmethod
    def get_bone_center(start: Vector3, end: Vector3) -> Vector3:
        """Kemik ortasını hesapla"""
        return Vector3(
            (start.x + end.x) / 2,
            (start.y + end.y) / 2,
            (start.z + end.z) / 2,
        )
    
    def process_landmarks(self, landmarks: List[LandmarkPoint]) -> SkeletonFrame:
        """Landmarks'i iskelet frame'ine dönüştür"""
        bones = []
        current_time = time.time() - self.start_time
        
        # Confidence hesapla (ortalama visibility)
        visibilities = [lm.visibility for lm in landmarks if lm.visibility is not None]
        confidence = np.mean(visibilities) if visibilities else 0.0
        
        for start_idx, end_idx in SkeletonConfig.BONE_PAIRS:
            if start_idx >= len(landmarks) or end_idx >= len(landmarks):
                continue
            
            start_lm = landmarks[start_idx]
            end_lm = landmarks[end_idx]
            
            # Landmark'ların visibility'sini kontrol et
            if start_lm.visibility is None or end_lm.visibility is None:
                continue
            if start_lm.visibility < 0.5 or end_lm.visibility < 0.5:
                continue
            
            start_pos = start_lm.to_vector3()
            end_pos = end_lm.to_vector3()
            
            # Kemik merkezi ve rotasyonunu hesapla
            bone_center = self.get_bone_center(start_pos, end_pos)
            bone_rotation = self.get_bone_euler_angles(start_pos, end_pos)
            
            bone_name = f"Bone_{start_idx}_{end_idx}"
            
            bone = BoneTransform(
                name=bone_name,
                position=bone_center,
                rotation=bone_rotation,
                scale=Vector3(1, 1, 1),
            )
            bones.append(bone)
        
        # Ek kemikleri ekle (kafa çizgileri)
        for start_idx, end_idx in SkeletonConfig.ADDITIONAL_BONES:
            if start_idx >= len(landmarks) or end_idx >= len(landmarks):
                continue
            
            start_lm = landmarks[start_idx]
            end_lm = landmarks[end_idx]
            
            if start_lm.visibility is None or end_lm.visibility is None:
                continue
            if start_lm.visibility < 0.5 or end_lm.visibility < 0.5:
                continue
            
            start_pos = start_lm.to_vector3()
            end_pos = end_lm.to_vector3()
            
            bone_center = self.get_bone_center(start_pos, end_pos)
            bone_rotation = self.get_bone_euler_angles(start_pos, end_pos)
            
            bone_name = f"Head_{start_idx}_{end_idx}"
            
            bone = BoneTransform(
                name=bone_name,
                position=bone_center,
                rotation=bone_rotation,
                scale=Vector3(1, 1, 1),
            )
            bones.append(bone)
        
        self.frame_count += 1
        
        return SkeletonFrame(
            timestamp=current_time,
            bones=bones,
            frame_number=self.frame_count,
            confidence=float(confidence),
        )


# ============================================================================
# IMAGE PROCESSOR
# ============================================================================

class ImageProcessor:
    """MediaPipe pose detection"""
    
    def __init__(self, model_dir: Optional[Path] = None) -> None:
        self.model_dir = model_dir or Path(__file__).with_name("models")
        self.model_dir.mkdir(parents=True, exist_ok=True)

        pose_model = self._ensure_model("pose_landmarker_lite.task", POSE_MODEL_URL)

        self.pose = PoseLandmarker.create_from_options(
            PoseLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(pose_model)),
                running_mode=RunningMode.IMAGE,
                num_poses=1,
                output_segmentation_masks=False,
            )
        )

    def _ensure_model(self, file_name: str, url: str) -> Path:
        destination = self.model_dir / file_name
        if not destination.exists():
            print(f"[Downloading model] {file_name} ...")
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

    def process_frame(self, frame_bgr: np.ndarray) -> List[LandmarkPoint]:
        """Frame'i işle ve landmarks döndür"""
        if frame_bgr is None or frame_bgr.size == 0:
            raise ValueError("Empty frame received")

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        pose_results = self.pose.detect(mp_image)

        pose_landmarks = self._extract_landmarks(
            pose_results.pose_landmarks[0] if pose_results.pose_landmarks else None
        )
        
        return pose_landmarks

    def close(self) -> None:
        self.pose.close()


# ============================================================================
# VISUALIZATION
# ============================================================================

class SkeletonVisualizer:
    """İskeleti görüntüye çiz"""
    
    BONE_COLOR = (0, 255, 0)
    JOINT_COLOR = (0, 0, 255)
    JOINT_RADIUS = 5
    BONE_THICKNESS = 2
    
    @staticmethod
    def draw_skeleton(frame: np.ndarray, landmarks: List[LandmarkPoint]) -> np.ndarray:
        """Skeleton'u görüntüye çiz"""
        if not landmarks:
            return frame
        
        annotated = frame.copy()
        height, width = annotated.shape[:2]
        
        # Kemikleri çiz
        for start_idx, end_idx in SkeletonConfig.BONE_PAIRS:
            if start_idx >= len(landmarks) or end_idx >= len(landmarks):
                continue
            
            start = landmarks[start_idx]
            end = landmarks[end_idx]
            
            if start.visibility is None or end.visibility is None or \
               start.visibility < 0.5 or end.visibility < 0.5:
                continue
            
            start_pos = (int(start.x * width), int(start.y * height))
            end_pos = (int(end.x * width), int(end.y * height))
            
            cv2.line(annotated, start_pos, end_pos, SkeletonVisualizer.BONE_COLOR, 
                    SkeletonVisualizer.BONE_THICKNESS)
        
        # Kafa çizgileri ekle
        for start_idx, end_idx in SkeletonConfig.ADDITIONAL_BONES:
            if start_idx >= len(landmarks) or end_idx >= len(landmarks):
                continue
            
            start = landmarks[start_idx]
            end = landmarks[end_idx]
            
            if start.visibility is None or end.visibility is None or \
               start.visibility < 0.5 or end.visibility < 0.5:
                continue
            
            start_pos = (int(start.x * width), int(start.y * height))
            end_pos = (int(end.x * width), int(end.y * height))
            
            cv2.line(annotated, start_pos, end_pos, SkeletonVisualizer.BONE_COLOR, 
                    SkeletonVisualizer.BONE_THICKNESS)
        
        # Eklemleri çiz (sadece ana eklemler)
        main_joints = [
            PoseLandmarks.NOSE.value,  # Baş - tek nokta
            PoseLandmarks.LEFT_SHOULDER.value,
            PoseLandmarks.RIGHT_SHOULDER.value,
            PoseLandmarks.LEFT_ELBOW.value,
            PoseLandmarks.RIGHT_ELBOW.value,
            PoseLandmarks.LEFT_WRIST.value,
            PoseLandmarks.RIGHT_WRIST.value,
            PoseLandmarks.LEFT_HIP.value,
            PoseLandmarks.RIGHT_HIP.value,
            PoseLandmarks.LEFT_KNEE.value,
            PoseLandmarks.RIGHT_KNEE.value,
            PoseLandmarks.LEFT_ANKLE.value,
            PoseLandmarks.RIGHT_ANKLE.value,
        ]
        
        for joint_idx in main_joints:
            if joint_idx >= len(landmarks):
                continue
            
            landmark = landmarks[joint_idx]
            if landmark.visibility is None or landmark.visibility < 0.5:
                continue
            
            pos = (int(landmark.x * width), int(landmark.y * height))
            cv2.circle(annotated, pos, SkeletonVisualizer.JOINT_RADIUS, 
                      SkeletonVisualizer.JOINT_COLOR, -1)
        
        return annotated


# ============================================================================
# MAIN RUNNER
# ============================================================================

class SkeletonCaptureRunner:
    """Canlı skeleton detection ve görselleştirme"""
    
    def __init__(self, camera_index: int = CAM_INDEX):
        self.camera_index = camera_index
        self.processor = ImageProcessor()
        self.skeleton_processor = SkeletonProcessor()
        self.visualizer = SkeletonVisualizer()
        
        self.capture = cv2.VideoCapture(camera_index)
        if not self.capture.isOpened():
            raise RuntimeError(f"Kamera açılamadı (index={camera_index})")
        
        # Frame rate kontrolü
        self.target_fps = 30
        self.frame_time = 1.0 / self.target_fps
        
        print(f"[SkeletonCapture] Başlatıldı - Kamera {camera_index}, FPS {self.target_fps}")
    
    def run(self) -> None:
        """Ana loop - kamera akışını işle"""
        frame_count = 0
        last_time = time.time()
        
        print("[SkeletonCapture] Çalışıyor... Q tuşuna basarak çık")
        
        try:
            while True:
                success, frame = self.capture.read()
                if not success:
                    break
                
                # Frame'i işle
                frame = cv2.flip(frame, 1)
                landmarks = self.processor.process_frame(frame)
                
                # İskeleti hesapla
                skeleton_frame = self.skeleton_processor.process_landmarks(landmarks)
                
                # Görüntüsü göster
                display_frame = self.visualizer.draw_skeleton(frame, landmarks)
                
                # İstatistikleri ekrana yazı olarak ekle
                cv2.putText(display_frame, f"Frame: {skeleton_frame.frame_number}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(display_frame, f"Confidence: {skeleton_frame.confidence:.2f}", 
                           (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(display_frame, f"Bones: {len(skeleton_frame.bones)}", 
                           (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                cv2.imshow("Skeleton Detection - Live", display_frame)
                
                # FPS kontrolü
                frame_count += 1
                current_time = time.time()
                elapsed = current_time - last_time
                
                if frame_count % 30 == 0:
                    actual_fps = 30 / elapsed
                    print(f"[Frame {skeleton_frame.frame_number}] FPS: {actual_fps:.1f}, "
                          f"Bones: {len(skeleton_frame.bones)}, "
                          f"Confidence: {skeleton_frame.confidence:.3f}")
                    last_time = current_time
                
                # Çık
                if cv2.waitKey(int(self.frame_time * 1000)) & 0xFF in (ord("q"), ord("Q")):
                    break
        
        finally:
            self.cleanup()
    
    def cleanup(self) -> None:
        """Kaynakları temizle"""
        print("[SkeletonCapture] Kapatılıyor...")
        self.capture.release()
        self.processor.close()
        cv2.destroyAllWindows()
        print("[SkeletonCapture] Kapatıldı.")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    runner = SkeletonCaptureRunner(camera_index=CAM_INDEX)
    runner.run()
