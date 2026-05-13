from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.request import urlretrieve
import threading
from abc import ABC, abstractmethod
import tempfile
import shutil
import os

import cv2
import mediapipe as mp
import numpy as np

import json
import socket

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

CAM_FRONT = 0
CAM_SIDE  = 1

UDP_HOST = "127.0.0.1"
UDP_PORT = 5005


# ============================================================================
# DATA MODELS
# ============================================================================

@dataclass
class LandmarkPoint:
    """Tekil landmark noktası"""
    x: float
    y: float
    z: float
    visibility: Optional[float] = None


@dataclass
class FrameResult:
    """Bir frame'den elde edilen deteksyon sonuçları"""
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


@dataclass
class FusedFrameResult:
    """Çift kamera verilerinin birleştirilmiş sonuçları"""
    front: FrameResult
    side:  FrameResult

    def to_dict(self) -> Dict[str, Any]:
        def fuse_landmarks(
            front_list: List[LandmarkPoint],
            side_list:  List[LandmarkPoint],
        ) -> List[Dict[str, Any]]:
            fused = []
            for i, fp in enumerate(front_list):
                real_z = side_list[i].x if i < len(side_list) else fp.z
                fused.append({
                    "x": fp.x,
                    "y": fp.y,
                    "z": real_z,
                    "visibility": fp.visibility,
                })
            return fused

        # Birleştirilmiş pose landmarks'ı Blender skeleton'a dönüştür
        fused_pose_landmarks = fuse_landmarks(
            self.front.pose_landmarks,
            self.side.pose_landmarks,
        )
        fused_points = [LandmarkPoint(
            x=p["x"], y=p["y"], z=p["z"], 
            visibility=p.get("visibility")
        ) for p in fused_pose_landmarks]
        
        return {
            "pose_landmarks": fused_pose_landmarks,
            "left_hand_landmarks": fuse_landmarks(
                self.front.left_hand_landmarks,
                self.side.left_hand_landmarks,
            ),
            "right_hand_landmarks": fuse_landmarks(
                self.front.right_hand_landmarks,
                self.side.right_hand_landmarks,
            ),
            "face_landmarks": fuse_landmarks(
                self.front.face_landmarks,
                self.side.face_landmarks,
            ),
            "has_mask": self.front.mask is not None,
            "skeleton": SkeletalMapper.get_bone_data(fused_points),  # Blender için skeleton data
        }



# ============================================================================
# CONFIGURATION CLASSES
# ============================================================================

class KeypointConfig:
    """Landmark keypoint indekslerini ve mapping'lerini içeren konfigürasyon"""
    
    POSE_KEYPOINTS: Dict[str, int] = {
        "nose": 0, "left_eye": 2, "right_eye": 5,
        "left_ear": 7, "right_ear": 8,
        "left_shoulder": 11, "right_shoulder": 12,
        "left_elbow": 13, "right_elbow": 14,
        "left_wrist": 15, "right_wrist": 16,
        "left_hip": 23, "right_hip": 24,
        "left_knee": 25, "right_knee": 26,
        "left_ankle": 27, "right_ankle": 28,
    }

    HAND_KEYPOINTS: Dict[str, int] = {
        "wrist": 0, "thumb_tip": 4, "index_tip": 8,
        "middle_tip": 12, "ring_tip": 16, "pinky_tip": 20,
    }

    FACE_KEYPOINTS: Dict[str, int] = {
        "forehead": 10, "left_eye_outer": 33, "right_eye_outer": 263,
        "nose_tip": 1, "mouth_left": 61, "mouth_right": 291, "chin": 152,
    }

    FINGER_TIP_KEYS: Tuple[str, ...] = ("thumb_tip", "index_tip", "middle_tip", "ring_tip", "pinky_tip")


class VisualizationConfig:
    """Görselleştirme parametreleri"""
    
    POINT_COLOR = (0, 255, 255)
    POINT_RADIUS = 3
    MASK_COLOR = (0, 255, 0)
    MASK_ALPHA = 0.35


class AnalysisConfig:
    """Analiz parametreleri"""
    
    MOVEMENT_THRESHOLD = 0.004
    PINCH_DELTA_THRESHOLD = 0.003


# ============================================================================
# SKELETAL MAPPER
# ============================================================================

class SkeletalMapper:
    """MediaPipe pose landmarks'ı Rigify Metarig kemik isimlerine dönüştürür.

    Çıktı isimler Rigify Metarig ile birebir aynıdır:
      spine, spine.001 ... spine.003, spine.006, spine.007
      shoulder.L/R, upper_arm.L/R, forearm.L/R, hand.L/R
      thigh.L/R, shin.L/R, foot.L/R, toe.L/R
    """
    
    @staticmethod
    def _pt(lm, i):
        """Landmark'i Blender Y-up koordinat sistemine cevir.
        MediaPipe: +X sola (kamera goruntusu), +Y asagi, +Z kameraya dogru
        Blender:   +X saga, +Y yukari, +Z kameraya dogru
        Donusum:   x=-mp.x (aynalama), y=-mp.y (ters), z=-mp.z
        """
        return np.array([-lm[i].x, -lm[i].y, -lm[i].z], dtype=np.float64)

    @staticmethod
    def _dir(s, e):
        """Baslangic-bitis noktalarindan normalize youn vektoru dondur.
        Euler acisi degil, direkt [dx, dy, dz] gonderilir.
        Addon bu vektoru kemigin rest-Y ile karsilastirir -> dogru rotasyon.
        Varsayilan: [0,1,0] = yukari (spine rest yonu).
        """
        d = e - s
        n = np.linalg.norm(d)
        if n < 1e-6:
            return [0.0, 1.0, 0.0]
        d = d / n
        return [float(d[0]), float(d[1]), float(d[2])]

    @staticmethod
    def get_bone_data(pose_landmarks):
        """Rigify Metarig isimli kemik verisi döndür."""
        if len(pose_landmarks) < 29:
            return {}

        lm = pose_landmarks
        pt = SkeletalMapper._pt
        dr = SkeletalMapper._dir   # normalize yon vektoru

        # Temel noktalar
        l_hip   = pt(lm, 23);  r_hip   = pt(lm, 24)
        l_shldr = pt(lm, 11);  r_shldr = pt(lm, 12)
        l_elbow = pt(lm, 13);  r_elbow = pt(lm, 14)
        l_wrist = pt(lm, 15);  r_wrist = pt(lm, 16)
        l_knee  = pt(lm, 25);  r_knee  = pt(lm, 26)
        l_ankle = pt(lm, 27);  r_ankle = pt(lm, 28)
        l_ear   = pt(lm,  7);  r_ear   = pt(lm,  8)
        nose    = pt(lm,  0)

        hip_c   = (l_hip   + r_hip)   / 2
        shldr_c = (l_shldr + r_shldr) / 2
        ear_c   = (l_ear   + r_ear)   / 2
        head_top = nose + (nose - ear_c) * 0.35

        has_feet = len(lm) >= 33
        l_foot = pt(lm, 31) if has_feet else l_ankle + np.array([0, 0.05, 0.1])
        r_foot = pt(lm, 32) if has_feet else r_ankle + np.array([0, 0.05, 0.1])
        l_toe  = l_foot + (l_foot - l_ankle) * 0.3
        r_toe  = r_foot + (r_foot - r_ankle) * 0.3

        bones = {}

        def add(name, pos, s, e, vis=1.0):
            """rotation = normalize yon vektoru [dx,dy,dz], euler degil."""
            bones[name] = {
                "position":   pos.tolist(),
                "rotation":   dr(s, e),
                "confidence": float(vis),
            }

        # Spine: sadece kok kemik doner, aralar rest-yon
        sv = shldr_c - hip_c
        bones["spine"] = {
            "position":   hip_c.tolist(),
            "rotation":   dr(hip_c, hip_c + sv),
            "confidence": 1.0,
        }
        # Ara kemikler: [0,1,0] = Blender +Y = yukari = rest yonu
        for bname, t in [("spine.001", 0.25), ("spine.002", 0.50), ("spine.003", 0.75)]:
            bones[bname] = {
                "position":   (hip_c + sv * t).tolist(),
                "rotation":   [0.0, 1.0, 0.0],
                "confidence": 1.0,
            }

        add("spine.006", shldr_c, shldr_c, ear_c)   # Boyun
        add("spine.007", ear_c,   ear_c,   head_top) # Kafa

        # Klavikula
        add("shoulder.L", shldr_c, shldr_c, l_shldr)
        add("shoulder.R", shldr_c, shldr_c, r_shldr)

        # Kollar
        add("upper_arm.L", l_shldr, l_shldr, l_elbow, lm[11].visibility or 1.0)
        add("forearm.L",   l_elbow, l_elbow, l_wrist, lm[13].visibility or 1.0)
        add("hand.L",      l_wrist, l_wrist, l_wrist + (l_wrist - l_elbow) * 0.3)
        add("upper_arm.R", r_shldr, r_shldr, r_elbow, lm[12].visibility or 1.0)
        add("forearm.R",   r_elbow, r_elbow, r_wrist, lm[14].visibility or 1.0)
        add("hand.R",      r_wrist, r_wrist, r_wrist + (r_wrist - r_elbow) * 0.3)

        # Bacaklar
        add("thigh.L", l_hip,   l_hip,   l_knee,  lm[23].visibility or 1.0)
        add("shin.L",  l_knee,  l_knee,  l_ankle, lm[25].visibility or 1.0)
        add("foot.L",  l_ankle, l_ankle, l_foot,  lm[27].visibility or 1.0)
        add("toe.L",   l_foot,  l_foot,  l_toe)
        add("thigh.R", r_hip,   r_hip,   r_knee,  lm[24].visibility or 1.0)
        add("shin.R",  r_knee,  r_knee,  r_ankle, lm[26].visibility or 1.0)
        add("foot.R",  r_ankle, r_ankle, r_foot,  lm[28].visibility or 1.0)
        add("toe.R",   r_foot,  r_foot,  r_toe)

        return bones


# ============================================================================
# PROCESSOR CLASSES
# ============================================================================

class ImageProcessor:
    """MediaPipe modelleriyle pose, hand ve face detection yapan sınıf"""
    
    def __init__(self, model_dir: Optional[Path] = None) -> None:
        self.model_dir = model_dir or Path(__file__).with_name("models")
        self.model_dir.mkdir(parents=True, exist_ok=True)

        # MediaPipe, Unicode yolları desteklemediği için ASCII-only path kullan
        safe_model_dir = self._get_safe_model_dir()
        
        pose_model = self._ensure_model("pose_landmarker_lite.task", POSE_MODEL_URL, safe_model_dir)
        hand_model = self._ensure_model("hand_landmarker.task", HAND_MODEL_URL, safe_model_dir)
        face_model = self._ensure_model("face_landmarker.task", FACE_MODEL_URL, safe_model_dir)

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
    
    def _get_safe_model_dir(self) -> Path:
        """Unicode karakterleri olan yolları ASCII-only path'e dönüştür"""
        try:
            # Yolda ASCII olmayan karakterler var mı kontrol et
            str(self.model_dir).encode('ascii')
            return self.model_dir
        except UnicodeEncodeError:
            # ASCII olmayan karakterler varsa, temp directory'ye kopyala
            temp_dir = Path(tempfile.gettempdir()) / "mediapipe_models"
            temp_dir.mkdir(parents=True, exist_ok=True)
            
            # Mevcut modelleri temp directory'ye kopyala
            for model_file in self.model_dir.glob("*.task"):
                dest = temp_dir / model_file.name
                if not dest.exists():
                    shutil.copy2(model_file, dest)
            
            return temp_dir

    def _ensure_model(self, file_name: str, url: str, model_dir: Optional[Path] = None) -> Path:
        """Model dosyasını indir veya var olan'ı kullan"""
        if model_dir is None:
            model_dir = self.model_dir
        destination = model_dir / file_name
        if not destination.exists():
            # Dosya orijinal konumda varsa, kopyala
            original = self.model_dir / file_name
            if original.exists():
                shutil.copy2(original, destination)
            else:
                print(f"[Downloading model] {file_name} ...")
                urlretrieve(url, destination)
        return destination

    @staticmethod
    def _extract_landmarks(landmarks: Optional[Any]) -> List[LandmarkPoint]:
        """MediaPipe landmarks'i LandmarkPoint listesine dönüştür"""
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
        """Segmentasyon maskesini numpy array'e dönüştür"""
        if hasattr(segmentation_mask, "numpy_view"):
            mask_array = segmentation_mask.numpy_view()
        else:
            mask_array = np.asarray(segmentation_mask)
        return (mask_array > threshold).astype(np.uint8) * 255

    def process_frame(self, frame_bgr: np.ndarray) -> FrameResult:
        """Frame'i işle ve deteksiyon sonuçlarını döndür"""
        if frame_bgr is None or frame_bgr.size == 0:
            raise ValueError("Empty frame received")

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        pose_results  = self.pose.detect(mp_image)
        hands_results = self.hands.detect(mp_image)
        face_results  = self.face_landmarker.detect(mp_image)

        pose_landmarks = self._extract_landmarks(
            pose_results.pose_landmarks[0] if pose_results.pose_landmarks else None
        )
        face_landmarks = self._extract_landmarks(
            face_results.face_landmarks[0] if face_results.face_landmarks else None
        )

        left_hand_landmarks: List[LandmarkPoint] = []
        right_hand_landmarks: List[LandmarkPoint] = []
        if hands_results.hand_landmarks and hands_results.handedness:
            for hand_landmarks, handedness in zip(hands_results.hand_landmarks, hands_results.handedness):
                label = handedness[0].category_name.lower()
                extracted = self._extract_landmarks(hand_landmarks)
                if label == "left":
                    right_hand_landmarks = extracted
                elif label == "right":
                    left_hand_landmarks = extracted

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
        """Kaynakları temizle"""
        self.pose.close()
        self.hands.close()
        self.face_landmarker.close()


# ============================================================================
# VISUALIZATION CLASSES
# ============================================================================

class LandmarkVisualizer:
    """Landmarks'i görüntüye çizmek ve görselleştirmek için sınıf"""
    
    def __init__(self, config: Optional[VisualizationConfig] = None):
        self.config = config or VisualizationConfig()
    
    def overlay_mask(self, frame_bgr: np.ndarray, mask: Optional[np.ndarray]) -> np.ndarray:
        """Maske transparanslı olarak görüntüye ekle"""
        if mask is None:
            return frame_bgr
        overlay = frame_bgr.copy()
        colored = np.zeros_like(frame_bgr)
        colored[:, :] = self.config.MASK_COLOR
        binary_mask = np.asarray(mask).astype(bool)
        if binary_mask.ndim == 3:
            binary_mask = np.squeeze(binary_mask, axis=-1)
        blended = cv2.addWeighted(frame_bgr, 1.0 - self.config.MASK_ALPHA, 
                                 colored, self.config.MASK_ALPHA, 0)
        overlay[binary_mask] = blended[binary_mask]
        return overlay

    def draw_pose(self, frame_bgr: np.ndarray, pose_landmarks: List[LandmarkPoint]) -> np.ndarray:
        """Pose landmarks'ini goruntuye ciz.
        Sari (0,255,255) : govde / eklenen kalca+bel noktalar
        Mavi (255,140,0) : sol kol
        Kirmizi (255,80,80): sag kol
        Yesil (0,220,80)  : sol bacak
        Koyu yesil(0,160,60): sag bacak
        """
        if not pose_landmarks:
            return frame_bgr
        annotated = frame_bgr.copy()
        h, w = annotated.shape[:2]
        lm = pose_landmarks

        def pxi(i): return (int(lm[i].x * w), int(lm[i].y * h))
        def ok(i):  return i < len(lm) and (lm[i].visibility or 0) > 0.25

        # Iskelet baglanti cizgileri
        CONN = [
            (11,12,(0,200,200)), (11,23,(0,200,200)), (12,24,(0,200,200)), (23,24,(0,200,200)),
            (11,13,(255,140,0)), (13,15,(255,140,0)),
            (12,14,(255,80,80)), (14,16,(255,80,80)),
            (23,25,(0,220,80)),  (25,27,(0,220,80)),  (27,31,(0,220,80)),
            (24,26,(0,160,60)),  (26,28,(0,160,60)),  (28,32,(0,160,60)),
            (0, 7,(180,180,180)),(0, 8,(180,180,180)),
        ]
        for a,b,col in CONN:
            if ok(a) and ok(b):
                cv2.line(annotated, pxi(a), pxi(b), col, 2, cv2.LINE_AA)

        # Grup renkleri
        GC = {**{i:(180,180,180) for i in range(11)},
              **{i:(255,140,0) for i in [11,13,15,17,19,21]},
              **{i:(255,80,80)  for i in [12,14,16,18,20,22]},
              **{i:(0,220,80)   for i in [23,25,27,29,31]},
              **{i:(0,160,60)   for i in [24,26,28,30,32]}}
        for i,pt in enumerate(lm):
            if (pt.visibility or 0) < 0.2: continue
            cx,cy = int(pt.x*w), int(pt.y*h)
            col = GC.get(i,(0,255,255))
            cv2.circle(annotated,(cx,cy),self.config.POINT_RADIUS,col,-1)
            cv2.circle(annotated,(cx,cy),self.config.POINT_RADIUS+1,(0,0,0),1)

        # Eklenen kalca/bel noktalar (MediaPipe'da ara omurga yok)
        EC=(0,255,255); ER=7  # sari, buyuk
        if len(lm)>=25 and ok(23) and ok(24):
            lh=lm[23]; rh=lm[24]
            hx=int((lh.x+rh.x)/2*w); hy=int((lh.y+rh.y)/2*h)
            cv2.circle(annotated,(hx,hy),ER,EC,-1)
            cv2.circle(annotated,(hx,hy),ER+1,(0,0,0),1)
            cv2.putText(annotated,"kalca",(hx+8,hy+4),cv2.FONT_HERSHEY_SIMPLEX,.38,EC,1,cv2.LINE_AA)
            if ok(11) and ok(12):
                ls=lm[11]; rs=lm[12]
                sx=int((ls.x+rs.x)/2*w); sy=int((ls.y+rs.y)/2*h)
                # Bel = kalca ile omuz arasinin %35'i yukari
                wx=int(hx*.65+sx*.35); wy=int(hy*.65+sy*.35)
                cv2.circle(annotated,(wx,wy),ER,EC,-1)
                cv2.circle(annotated,(wx,wy),ER+1,(0,0,0),1)
                cv2.putText(annotated,"bel",(wx+8,wy+4),cv2.FONT_HERSHEY_SIMPLEX,.38,EC,1,cv2.LINE_AA)
                cv2.circle(annotated,(sx,sy),ER-1,EC,-1)
                cv2.circle(annotated,(sx,sy),ER,(0,0,0),1)
                cv2.putText(annotated,"omuz_m",(sx+8,sy+4),cv2.FONT_HERSHEY_SIMPLEX,.38,EC,1,cv2.LINE_AA)
                cv2.line(annotated,(hx,hy),(sx,sy),EC,2,cv2.LINE_AA)  # spine cizgisi
        return annotated



# ============================================================================
# ANALYSIS CLASSES
# ============================================================================

class LandmarkAnalyzer:
    """Landmark verilerini analiz etmek ve formatlamak için sınıf"""
    
    def __init__(self, config: Optional[AnalysisConfig] = None):
        self.config = config or AnalysisConfig()
    
    def _get_movement_direction(self, delta: float, threshold: float, 
                               negative_label: str, positive_label: str) -> str:
        """Belirli bir eksende hareket yönünü belirle"""
        if delta > threshold:
            return positive_label
        if delta < -threshold:
            return negative_label
        return "steady"
    
    def format_landmark_section(self, name: str, landmarks: List[LandmarkPoint], 
                               keypoints: Dict[str, int]) -> List[str]:
        """Landmark bölümünü formatla"""
        lines = [f"{name}:"]
        if not landmarks:
            lines.append("  not detected")
            return lines
        
        for label, index in keypoints.items():
            if index >= len(landmarks):
                lines.append(f"  {label:<14} missing")
                continue
            point = landmarks[index]
            if point.visibility is None:
                lines.append(f"  {label:<14} x={point.x:.4f} y={point.y:.4f} z={point.z:.4f}")
            else:
                lines.append(f"  {label:<14} x={point.x:.4f} y={point.y:.4f} z={point.z:.4f} vis={point.visibility:.4f}")
        return lines
    
    def format_hand_motion(self, hand_name: str, current_landmarks: List[LandmarkPoint],
                          previous_landmarks: Optional[List[LandmarkPoint]]) -> List[str]:
        """Parmak hareketlerini ve pinch durumunu formatla"""
        lines = [f"{hand_name} Finger Motions:"]
        if not current_landmarks:
            lines.append("  not detected")
            return lines
        if not previous_landmarks:
            lines.append("  calibrating... keep your hand visible")
            return lines
        
        # Parmak hareketlerini analiz et
        for key in KeypointConfig.FINGER_TIP_KEYS:
            idx = KeypointConfig.HAND_KEYPOINTS[key]
            if idx >= len(current_landmarks) or idx >= len(previous_landmarks):
                lines.append(f"  {key:<14} missing")
                continue
            
            current = current_landmarks[idx]
            previous = previous_landmarks[idx]
            dx = current.x - previous.x
            dy = current.y - previous.y
            dz = current.z - previous.z
            movement = (dx * dx + dy * dy + dz * dz) ** 0.5
            
            x_dir = self._get_movement_direction(dx, self.config.MOVEMENT_THRESHOLD, 
                                               "left", "right")
            y_dir = self._get_movement_direction(dy, self.config.MOVEMENT_THRESHOLD, 
                                               "up", "down")
            z_dir = self._get_movement_direction(dz, self.config.MOVEMENT_THRESHOLD, 
                                               "forward", "back")
            state = "moving" if movement > self.config.MOVEMENT_THRESHOLD else "stable"
            lines.append(f"  {key:<14} {state:<6} d={movement:.4f} x={x_dir:<6} y={y_dir:<6} z={z_dir:<7}")
        
        # Pinch hareketi analiz et
        thumb_idx = KeypointConfig.HAND_KEYPOINTS["thumb_tip"]
        index_idx = KeypointConfig.HAND_KEYPOINTS["index_tip"]
        if (thumb_idx < len(current_landmarks) and index_idx < len(current_landmarks) and
            thumb_idx < len(previous_landmarks) and index_idx < len(previous_landmarks)):
            
            current_pinch = float(np.linalg.norm([
                current_landmarks[thumb_idx].x - current_landmarks[index_idx].x,
                current_landmarks[thumb_idx].y - current_landmarks[index_idx].y,
                current_landmarks[thumb_idx].z - current_landmarks[index_idx].z,
            ]))
            previous_pinch = float(np.linalg.norm([
                previous_landmarks[thumb_idx].x - previous_landmarks[index_idx].x,
                previous_landmarks[thumb_idx].y - previous_landmarks[index_idx].y,
                previous_landmarks[thumb_idx].z - previous_landmarks[index_idx].z,
            ]))
            pinch_delta = current_pinch - previous_pinch
            pinch_state = ("closing" if pinch_delta < -self.config.PINCH_DELTA_THRESHOLD 
                          else "opening" if pinch_delta > self.config.PINCH_DELTA_THRESHOLD 
                          else "steady")
            lines.append(f"  pinch(thumb-index) {pinch_state} dist={current_pinch:.4f}")
        
        return lines


class LandmarkPrinter:
    """Landmark verilerini terminal'e yazdırmak için sınıf"""
    
    def __init__(self, analyzer: Optional[LandmarkAnalyzer] = None):
        self.analyzer = analyzer or LandmarkAnalyzer()
    
    def print_live(self, result: FrameResult,
                   previous_left_hand: Optional[List[LandmarkPoint]] = None,
                   previous_right_hand: Optional[List[LandmarkPoint]] = None) -> None:
        """Tüm landmark bilgilerini terminal'e yazdır"""
        sections: List[str] = []
        
        # Pose keypoints
        sections.extend(self.analyzer.format_landmark_section(
            "Body Keypoints", result.pose_landmarks, KeypointConfig.POSE_KEYPOINTS))
        sections.append("")
        
        # Face keypoints
        sections.extend(self.analyzer.format_landmark_section(
            "Head Keypoints", result.face_landmarks, KeypointConfig.FACE_KEYPOINTS))
        sections.append("")
        
        # Hand keypoints
        sections.extend(self.analyzer.format_landmark_section(
            "Left Hand Keypoints", result.left_hand_landmarks, KeypointConfig.HAND_KEYPOINTS))
        sections.append("")
        sections.extend(self.analyzer.format_landmark_section(
            "Right Hand Keypoints", result.right_hand_landmarks, KeypointConfig.HAND_KEYPOINTS))
        sections.append("")
        
        # Hand motions
        sections.extend(self.analyzer.format_hand_motion(
            "Left Hand", result.left_hand_landmarks, previous_left_hand))
        sections.append("")
        sections.extend(self.analyzer.format_hand_motion(
            "Right Hand", result.right_hand_landmarks, previous_right_hand))
        
        print("\x1b[2J\x1b[H" + "\n".join(sections), end="", flush=True)


# ============================================================================
# COMMUNICATION CLASSES
# ============================================================================

class UDPPublisher:
    """UDP socket üzerinden veri gönderme"""
    
    def __init__(self, host: str = UDP_HOST, port: int = UDP_PORT):
        self.host = host
        self.port = port
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.target = (host, port)
    
    def send(self, data: Dict[str, Any]) -> None:
        """Dictionary verilerini JSON olarak UDP'ye gönder"""
        try:
            payload = json.dumps(data).encode("utf-8")
            self.socket.sendto(payload, self.target)
        except Exception as e:
            print(f"[UDP Error] {e}")
    
    def close(self) -> None:
        """Socket'i kapat"""
        self.socket.close()


# ============================================================================
# RUNNER BASE CLASSES
# ============================================================================

class CameraRunner(ABC):
    """Kamera ve processing logic'ini yöneten abstract sınıf"""
    
    def __init__(self):
        self.processor = ImageProcessor()
        self.visualizer = LandmarkVisualizer()
        self.analyzer = LandmarkAnalyzer()
        self.printer = LandmarkPrinter(self.analyzer)
        self.publisher = UDPPublisher()
        self.previous_left_hand: Optional[List[LandmarkPoint]] = None
        self.previous_right_hand: Optional[List[LandmarkPoint]] = None
    
    @abstractmethod
    def run(self) -> None:
        """Başlıca işletme loop'u"""
        pass
    
    def cleanup(self) -> None:
        """Kaynakları temizle"""
        self.processor.close()
        self.publisher.close()
        cv2.destroyAllWindows()


class SingleCameraRunner(CameraRunner):
    """Tek kamera çalıştıran sınıf"""
    
    def __init__(self, camera_index: int = 0):
        super().__init__()
        self.camera_index = camera_index
        self.capture = cv2.VideoCapture(camera_index)
        
        if not self.capture.isOpened():
            self.cleanup()
            raise RuntimeError(f"Webcam (index={camera_index}) could not be opened")
    
    def run(self) -> None:
        """Webcam akışını işle"""
        try:
            while True:
                success, frame = self.capture.read()
                if not success:
                    break
                
                frame = cv2.flip(frame, 1)
                result = self.processor.process_frame(frame)
                
                # Terminal çıktısı
                self.printer.print_live(result, self.previous_left_hand, self.previous_right_hand)
                
                # El hareketi takibi
                self.previous_left_hand = result.left_hand_landmarks
                self.previous_right_hand = result.right_hand_landmarks
                
                # UDP gonder: ham landmarks + hesaplanmis skeleton
                data_to_send = result.to_dict()
                if result.pose_landmarks:
                    data_to_send["skeleton"] = SkeletalMapper.get_bone_data(result.pose_landmarks)
                self.publisher.send(data_to_send)
                
                # Görüntüsü göster
                output = self.visualizer.draw_pose(frame, result.pose_landmarks)
                cv2.imshow("Camera (Single)", output)
                
                if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
                    break
        finally:
            self.capture.release()
            self.cleanup()


class DualCameraRunner(CameraRunner):
    """Dual kamera (ön + yan) çalıştıran sınıf"""
    
    def __init__(self, front_index: int = CAM_FRONT, side_index: int = CAM_SIDE):
        super().__init__()
        self.front_index = front_index
        self.side_index = side_index
        
        self.processor_front = ImageProcessor()
        self.processor_side = ImageProcessor()
        
        self.cap_front = cv2.VideoCapture(front_index)
        self.cap_side = cv2.VideoCapture(side_index)
        
        self._result_front: Optional[FrameResult] = None
        self._result_side: Optional[FrameResult] = None
        self._lock = threading.Lock()
        self._running = False
        
        self._check_cameras()
    
    def _check_cameras(self) -> None:
        """Kameraların açılabilir durumda olup olmadığını kontrol et"""
        if not self.cap_front.isOpened():
            raise RuntimeError(f"Front camera (index={self.front_index}) could not be opened")
        if not self.cap_side.isOpened():
            raise RuntimeError(
                f"Side camera (index={self.side_index}) could not be opened. "
                "Use SingleCameraRunner to test with a single camera."
            )
    
    def _front_loop(self) -> None:
        """Ön kamera işleme loop'u"""
        while self._running:
            success, frame = self.cap_front.read()
            if not success:
                continue
            frame = cv2.flip(frame, 1)
            result = self.processor_front.process_frame(frame)
            with self._lock:
                self._result_front = result
    
    def _side_loop(self) -> None:
        """Yan kamera işleme loop'u"""
        while self._running:
            success, frame = self.cap_side.read()
            if not success:
                continue
            result = self.processor_side.process_frame(frame)
            with self._lock:
                self._result_side = result
    
    def run(self) -> None:
        """Dual kamera akışını işle"""
        self._running = True
        
        # İş parçacıklarını başlat
        t_front = threading.Thread(target=self._front_loop, daemon=True)
        t_side = threading.Thread(target=self._side_loop, daemon=True)
        t_front.start()
        t_side.start()
        
        print(f"[LiveSync] Dual camera started → UDP {UDP_HOST}:{UDP_PORT}")
        print("[LiveSync] Press 'Q' to quit")
        
        try:
            while True:
                with self._lock:
                    front = self._result_front
                    side = self._result_side
                
                if front is not None and side is not None:
                    fused = FusedFrameResult(front=front, side=side)
                    
                    # Terminal çıktısı
                    self.printer.print_live(front, self.previous_left_hand, self.previous_right_hand)
                    
                    # El hareketi takibi
                    self.previous_left_hand = front.left_hand_landmarks
                    self.previous_right_hand = front.right_hand_landmarks
                    
                    # UDP gönder
                    self.publisher.send(fused.to_dict())
                    
                    # Görüntüsü göster
                    out_front = self.visualizer.draw_pose(front.frame_bgr, front.pose_landmarks)
                    out_side = self.visualizer.draw_pose(side.frame_bgr, side.pose_landmarks)
                    
                    # Ekranları aynı boyuta getir ve yanyana göster
                    h = min(out_front.shape[0], out_side.shape[0])
                    out_front = cv2.resize(out_front, (int(out_front.shape[1] * h / out_front.shape[0]), h))
                    out_side = cv2.resize(out_side, (int(out_side.shape[1] * h / out_side.shape[0]), h))
                    cv2.imshow("LiveSync — Front | Side", np.hstack([out_front, out_side]))
                
                if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
                    break
        
        finally:
            self._running = False
            self.cap_front.release()
            self.cap_side.release()
            self.processor_front.close()
            self.processor_side.close()
            self.cleanup()
            print("[LiveSync] Closed.")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    # ── Dual kamera (2 kamera gerektirir) ──────────────────────────────────
    # runner = DualCameraRunner()
    # ───────────────────────────────────────────────────────────────────────

    # Tek kamera — şu an aktif
    runner = SingleCameraRunner(camera_index=0)
    runner.run()
