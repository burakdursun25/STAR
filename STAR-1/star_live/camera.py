"""Tek bir fiziksel kamerayı yöneten sınıf."""
from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np


class CameraCapture:
    """
    OpenCV VideoCapture sarmalayıcısı.
    Context manager olarak kullanılabilir:
        with CameraCapture(0) as cam:
            frame = cam.read()
    """

    def __init__(
        self,
        camera_id: int,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
    ) -> None:
        self.camera_id = camera_id
        self._width = width
        self._height = height
        self._fps = fps
        self._cap: Optional[cv2.VideoCapture] = None

    # ── Yaşam döngüsü ────────────────────────────────────────────────────────

    def open(self) -> "CameraCapture":
        self._cap = cv2.VideoCapture(self.camera_id)
        if not self._cap.isOpened():
            raise RuntimeError(f"Kamera {self.camera_id} açılamadı.")
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        self._cap.set(cv2.CAP_PROP_FPS, self._fps)
        return self

    def release(self) -> None:
        if self._cap and self._cap.isOpened():
            self._cap.release()
            self._cap = None

    def __enter__(self) -> "CameraCapture":
        return self.open()

    def __exit__(self, *_) -> None:
        self.release()

    # ── Kare okuma ───────────────────────────────────────────────────────────

    def read(self) -> Optional[np.ndarray]:
        """Başarılı olursa BGR kareyi, başarısız olursa None döner."""
        if self._cap is None or not self._cap.isOpened():
            return None
        ok, frame = self._cap.read()
        return frame if ok else None

    @property
    def is_open(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

    @property
    def resolution(self) -> Tuple[int, int]:
        """(genişlik, yükseklik)"""
        if self._cap is None:
            return (self._width, self._height)
        return (
            int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
