"""
STAR-1 – Blender Live Stream Sender
=====================================
Webcam'den MediaPipe ile pose/hand/face verisi alır,
Blender'ın anlayacağı iskelet formatına çevirir
ve UDP üzerinden Blender addon'una canlı olarak gönderir.

# STAR-1 → Blender Canlı İskelet Akışı

## Sistem Mimarisi

```
Webcam
  │
  ▼
image_processing.py  (MediaPipe: Pose + Hand + Face)
  │
  ▼
blender_stream_sender.py  (Landmark → Euler dönüşümü)
  │  UDP  localhost:7777
  ▼
blender_live_addon.py  (Blender'da kemikleri günceller)
  │
  ▼
Armature (canlı hareket)
```

---

## Kurulum

### 1. Python Tarafı

```bash
pip install opencv-python mediapipe numpy
```

### 2. Blender Addon Kurulumu

1. Blender'ı aç
2. Edit → Preferences → Add-ons → Install…
3. `blender_live_addon.py` dosyasını seç
4. STAR Live Pose Receiver addonunu etkinleştir (checkbox)

### 3. Armature Hazırlığı

Blender'daki kemik isimlerini `blender_stream_sender.py` içindeki
`POSE_BONE_PAIRS` listesiyle eşleştir:

| Python tarafı | Blender kemik ismi (örnek) |
|--------------|---------------------------|
| `spine`      | `spine`                   |
| `upper_arm.L`| `upper_arm.L`             |
| `upper_arm.R`| `upper_arm.R`             |
| `forearm.L`  | `forearm.L`               |
| `forearm.R`  | `forearm.R`               |
| `thigh.L`    | `thigh.L`                 |
| `shin.L`     | `shin.L`                  |
| `thigh.R`    | `thigh.R`                 |
| `shin.R`     | `shin.R`                  |
| `hips`       | `hips`                    |

> Not: Blender'da Pose Mode'a geç, kemik isimlerini N-Panel → Item bölümünden gör.
> İsimler farklıysa `POSE_BONE_PAIRS` listesini güncelle.

---

## Çalıştırma

### Adım 1: Blender'da addon'u başlat

1. 3D Viewport → N tuşu → "STAR Live" sekmesi
2. Armature alanına nesne adını yaz (örn. `Armature`)
3. "Dinlemeyi Başlat" butonuna bas

### Adım 2: Python'u çalıştır

```bash
# Basit kullanım (kamera 0, localhost:7777)
python blender_stream_sender.py

# Özel kamera / port
python blender_stream_sender.py --camera 1 --port 7777

# Önizleme olmadan (hafif)
python blender_stream_sender.py --no-preview
```
"""

from __future__ import annotations

import argparse
import json
import math
import socket
import struct
import time
from typing import Dict, List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np

from image_processing import (
    FrameResult,
    ImageProcessor,
    LandmarkPoint,
    draw_pose,
    overlay_mask,
)

# ─────────────────────────────────────────────────────────────────────────────
# MediaPipe Pose landmark index → Blender bone name mapping
# Blender'daki iskelet için standart Rigify/Human Meta rig isimleri kullanılır.
# Kendi armature isimlerinizi buraya yazın.
# ─────────────────────────────────────────────────────────────────────────────
POSE_BONE_PAIRS: List[Tuple[str, int, int]] = [
    # (bone_name, landmark_start_idx, landmark_end_idx)
    ("spine",          23, 11),   # hips → left_shoulder (gövde)
    ("chest",          11, 12),   # left_shoulder → right_shoulder
    ("neck",           12,  0),   # right_shoulder → nose (boyun yaklaşımı)
    ("upper_arm.L",    11, 13),   # left_shoulder → left_elbow
    ("forearm.L",      13, 15),   # left_elbow → left_wrist
    ("upper_arm.R",    12, 14),   # right_shoulder → right_elbow
    ("forearm.R",      14, 16),   # right_elbow → right_wrist
    ("thigh.L",        23, 25),   # left_hip → left_knee
    ("shin.L",         25, 27),   # left_knee → left_ankle
    ("thigh.R",        24, 26),   # right_hip → right_knee
    ("shin.R",         26, 28),   # right_knee → right_ankle
    ("foot.L",         27, 31),   # left_ankle → left_foot_index
    ("foot.R",         28, 32),   # right_ankle → right_foot_index
]

# ─────────────────────────────────────────────────────────────────────────────
# Yardımcı matematik fonksiyonları
# ─────────────────────────────────────────────────────────────────────────────

def _lm_to_vec3(lm: LandmarkPoint) -> np.ndarray:
    """MediaPipe normalize koordinat → 3D vektör (z ekseni derinlik)."""
    return np.array([lm.x, -lm.y, -lm.z], dtype=np.float32)  # Y-up dönüşümü


def _vec3_to_euler(direction: np.ndarray) -> Tuple[float, float, float]:
    """
    Bir yön vektörünü Euler açılarına (XYZ, radyan) dönüştürür.
    Blender'ın varsayılan eksen yönlendirmesiyle uyumludur.
    """
    norm = np.linalg.norm(direction)
    if norm < 1e-6:
        return (0.0, 0.0, 0.0)
    d = direction / norm

    pitch = math.asin(max(-1.0, min(1.0, float(-d[1]))))
    yaw   = math.atan2(float(d[0]), float(d[2]))
    roll  = 0.0  # roll bilgisi tek vektörden çıkarılamaz
    return (pitch, yaw, roll)


def _landmark_confidence(lm: LandmarkPoint) -> float:
    return float(lm.visibility) if lm.visibility is not None else 1.0


# ─────────────────────────────────────────────────────────────────────────────
# Pose → Skeleton frame dönüşümü
# ─────────────────────────────────────────────────────────────────────────────

def build_skeleton_frame(result: FrameResult, frame_number: int) -> Dict:
    """
    FrameResult'tan Blender addon'unun beklediği formatta bir dict üretir:
    {
        "timestamp": float,
        "frame": int,
        "skeleton": {
            "bone_name": {
                "position": [x, y, z],       # head landmark pozisyonu (normalize)
                "rotation": [rx, ry, rz],     # Euler açıları (radyan)
                "confidence": float
            }, ...
        },
        "left_hand":  {...},   # el kemikleri (opsiyonel)
        "right_hand": {...},
    }
    """
    skeleton: Dict = {}

    pose = result.pose_landmarks
    if pose:
        for bone_name, start_idx, end_idx in POSE_BONE_PAIRS:
            if start_idx >= len(pose) or end_idx >= len(pose):
                continue
            lm_start = pose[start_idx]
            lm_end   = pose[end_idx]
            direction = _lm_to_vec3(lm_end) - _lm_to_vec3(lm_start)
            rotation  = _vec3_to_euler(direction)
            position  = [lm_start.x, lm_start.y, lm_start.z]
            confidence = min(
                _landmark_confidence(lm_start),
                _landmark_confidence(lm_end),
            )
            skeleton[bone_name] = {
                "position":   position,
                "rotation":   list(rotation),
                "confidence": confidence,
            }

        # Hips (pelvis) ayrıca: sol + sağ kalça ortalaması
        if len(pose) > 24:
            l_hip = pose[23]
            r_hip = pose[24]
            skeleton["hips"] = {
                "position": [
                    (l_hip.x + r_hip.x) / 2,
                    (l_hip.y + r_hip.y) / 2,
                    (l_hip.z + r_hip.z) / 2,
                ],
                "rotation":   [0.0, 0.0, 0.0],
                "confidence": min(_landmark_confidence(l_hip), _landmark_confidence(r_hip)),
            }

    # ── El kemikleri ─────────────────────────────────────────────────────────
    def _hand_dict(landmarks: List[LandmarkPoint]) -> Dict:
        hand: Dict = {}
        HAND_BONES = [
            ("palm",       0, 9),
            ("thumb.1",    1, 2),
            ("thumb.2",    2, 3),
            ("thumb.3",    3, 4),
            ("index.1",    5, 6),
            ("index.2",    6, 7),
            ("index.3",    7, 8),
            ("middle.1",   9, 10),
            ("middle.2",  10, 11),
            ("middle.3",  11, 12),
            ("ring.1",    13, 14),
            ("ring.2",    14, 15),
            ("ring.3",    15, 16),
            ("pinky.1",   17, 18),
            ("pinky.2",   18, 19),
            ("pinky.3",   19, 20),
        ]
        if not landmarks:
            return hand
        for bone_name, s, e in HAND_BONES:
            if s >= len(landmarks) or e >= len(landmarks):
                continue
            direction = _lm_to_vec3(landmarks[e]) - _lm_to_vec3(landmarks[s])
            rotation  = _vec3_to_euler(direction)
            hand[bone_name] = {
                "position":   [landmarks[s].x, landmarks[s].y, landmarks[s].z],
                "rotation":   list(rotation),
                "confidence": 1.0,
            }
        return hand

    return {
        "timestamp":   time.time(),
        "frame":       frame_number,
        "skeleton":    skeleton,
        "left_hand":   _hand_dict(result.left_hand_landmarks),
        "right_hand":  _hand_dict(result.right_hand_landmarks),
        "bones":       list(skeleton.keys()),   # panel'de gösterim için
    }


# ─────────────────────────────────────────────────────────────────────────────
# UDP Gönderici
# ─────────────────────────────────────────────────────────────────────────────

class UDPSender:
    """
    Büyük JSON paketlerini (>65507 byte) chunk'lara bölerek UDP ile gönderir.
    Blender addon'u aynı protokolü kullanarak birleştirir.
    """
    MAX_CHUNK = 60000  # UDP güvenli sınır

    def __init__(self, host: str = "127.0.0.1", port: int = 7777) -> None:
        self.addr = (host, port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._seq = 0

    def send(self, data: Dict) -> None:
        payload = json.dumps(data, separators=(",", ":")).encode("utf-8")
        chunks  = [payload[i:i + self.MAX_CHUNK] for i in range(0, len(payload), self.MAX_CHUNK)]
        n       = len(chunks)
        seq     = self._seq & 0xFFFF
        self._seq += 1

        for idx, chunk in enumerate(chunks):
            # Header: [seq:2B][total:2B][idx:2B]
            header = struct.pack("!HHH", seq, n, idx)
            self.sock.sendto(header + chunk, self.addr)

    def close(self) -> None:
        self.sock.close()


# ─────────────────────────────────────────────────────────────────────────────
# Ana döngü
# ─────────────────────────────────────────────────────────────────────────────

def run(camera: int = 0, host: str = "127.0.0.1", port: int = 7777, preview: bool = True) -> None:
    print(f"[STAR-1] Başlatılıyor → kamera={camera}  hedef={host}:{port}")
    processor = ImageProcessor()
    sender    = UDPSender(host, port)
    cap       = cv2.VideoCapture(camera)

    if not cap.isOpened():
        processor.close()
        sender.close()
        raise RuntimeError(f"Kamera {camera} açılamadı.")

    frame_number = 0
    fps_timer    = time.time()
    fps_count    = 0
    fps_display  = 0.0

    print("[STAR-1] Çalışıyor — 'q' ile çıkın.")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            result = processor.process_frame(frame)
            skeleton_data = build_skeleton_frame(result, frame_number)
            sender.send(skeleton_data)

            frame_number += 1
            fps_count    += 1
            now = time.time()
            if now - fps_timer >= 1.0:
                fps_display = fps_count / (now - fps_timer)
                fps_count   = 0
                fps_timer   = now

            if preview:
                out = frame.copy()
                if result.mask is not None:
                    out = overlay_mask(out, result.mask)
                out = draw_pose(out, result.pose_landmarks)

                bone_count = len(skeleton_data["skeleton"])
                cv2.putText(out, f"FPS: {fps_display:.1f}  Bones: {bone_count}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(out, f"UDP → {host}:{port}",
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 220, 255), 2)
                cv2.imshow("STAR-1  |  Blender Stream", out)

                if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q"), 27):
                    break
            else:
                # Preview kapalıysa klavye kontrolü için kısa bekleme
                time.sleep(0.001)

    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        processor.close()
        sender.close()
        if preview:
            cv2.destroyAllWindows()
        print("[STAR-1] Durduruldu.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="STAR-1 Blender Live Stream Sender")
    parser.add_argument("--camera",  type=int,   default=0,           help="Kamera indeksi (varsayılan: 0)")
    parser.add_argument("--host",    type=str,   default="127.0.0.1", help="Blender addon UDP host")
    parser.add_argument("--port",    type=int,   default=7777,        help="Blender addon UDP port")
    parser.add_argument("--preview", action="store_true", default=True, help="Canlı kamera önizlemesi")
    parser.add_argument("--no-preview", dest="preview", action="store_false")
    args = parser.parse_args()

    run(camera=args.camera, host=args.host, port=args.port, preview=args.preview)
