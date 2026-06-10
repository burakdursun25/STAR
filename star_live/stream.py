"""Ana orkestratör: 2 kamera → MediaPipe → Füzyon → Blender UDP."""
from __future__ import annotations

import threading
import time
from typing import List, Optional

import cv2
import numpy as np

from .camera import CameraCapture
from .fusion import DualCameraFusion
from .processor import ImageProcessor
from .sender import UDPSender
from .skeleton import SkeletonBuilder
from .smoother import SkeletonSmoother
from .types import FrameResult, SkeletonFrame


class _CameraWorker(threading.Thread):
    """
    Tek bir kamerayı ayrı thread'de okur ve son SkeletonFrame'i saklar.
    Ana thread ile lock üzerinden haberleşir.
    """

    def __init__(
        self,
        camera: CameraCapture,
        processor: ImageProcessor,
        builder: SkeletonBuilder,
    ) -> None:
        super().__init__(daemon=True)
        self._camera    = camera
        self._processor = processor
        self._builder   = builder
        self._lock      = threading.Lock()
        self._frame_result: Optional[FrameResult]  = None
        self._skeleton:     Optional[SkeletonFrame] = None
        self._raw_frame:    Optional[np.ndarray]    = None
        self._frame_no      = 0
        self._stop_event    = threading.Event()

    def run(self) -> None:
        while not self._stop_event.is_set():
            raw = self._camera.read()
            if raw is None:
                time.sleep(0.005)
                continue
            result   = self._processor.process_frame(raw)
            skeleton = self._builder.build(result, self._frame_no, self._camera.camera_id)
            self._frame_no += 1
            with self._lock:
                self._raw_frame    = raw
                self._frame_result = result
                self._skeleton     = skeleton

    def stop(self) -> None:
        self._stop_event.set()

    @property
    def latest_skeleton(self) -> Optional[SkeletonFrame]:
        with self._lock:
            return self._skeleton

    @property
    def latest_frame(self) -> Optional[np.ndarray]:
        with self._lock:
            return self._raw_frame

    @property
    def latest_result(self) -> Optional[FrameResult]:
        with self._lock:
            return self._frame_result


class StarLiveStream:
    """
    2 kamera ile canlı iskelet akışı.

    Kullanım:
        stream = StarLiveStream(camera_ids=[0, 1])
        stream.run()          # q tuşuyla çıkın
    """

    def __init__(
        self,
        camera_ids: List[int] = [0, 1],
        host: str = "127.0.0.1",
        port: int = 7777,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        smooth_cutoff: float = 1.5,
        smooth_beta: float = 0.3,
    ) -> None:
        self._camera_ids    = camera_ids
        self._host          = host
        self._port          = port
        self._width         = width
        self._height        = height
        self._fps           = fps
        self._smooth_cutoff = smooth_cutoff
        self._smooth_beta   = smooth_beta

    # ── Ana döngü ─────────────────────────────────────────────────────────────

    def run(self, preview: bool = True) -> None:
        n = len(self._camera_ids)
        print(f"[STAR-1] {n} kamera başlatılıyor: {self._camera_ids}")
        print(f"[STAR-1] Blender hedefi: {self._host}:{self._port}")

        cameras   = [
            CameraCapture(cid, self._width, self._height, self._fps).open()
            for cid in self._camera_ids
        ]
        processors = [ImageProcessor() for _ in cameras]
        builder    = SkeletonBuilder()
        fusion     = DualCameraFusion()
        smoother   = SkeletonSmoother(self._smooth_cutoff, self._smooth_beta)
        sender     = UDPSender(self._host, self._port)

        workers = [
            _CameraWorker(cam, proc, builder)
            for cam, proc in zip(cameras, processors)
        ]
        for w in workers:
            w.start()

        frame_number = 0
        fps_timer    = time.time()
        fps_count    = 0
        fps_display  = 0.0

        print("[STAR-1] Çalışıyor — 'q' ile çıkın.")
        try:
            while True:
                # ── Tüm kameralardan skeleton al ve birleştir ──────────────
                skeletons = [w.latest_skeleton for w in workers]
                fused = skeletons[0]
                for sk in skeletons[1:]:
                    fused = fusion.fuse(fused, sk)

                if fused is not None:
                    fused.frame_number = frame_number
                    fused = smoother.smooth(fused)
                    sender.send(fused.to_dict())

                frame_number += 1
                fps_count    += 1
                now = time.time()
                if now - fps_timer >= 1.0:
                    fps_display = fps_count / (now - fps_timer)
                    fps_count   = 0
                    fps_timer   = now

                # ── Önizleme ──────────────────────────────────────────────
                if preview:
                    frames = []
                    for idx, w in enumerate(workers):
                        raw    = w.latest_frame
                        result = w.latest_result
                        if raw is None:
                            continue
                        out = raw.copy()
                        if result is not None:
                            if result.mask is not None:
                                out = ImageProcessor.overlay_mask(out, result.mask)
                            out = ImageProcessor.draw_pose(out, result.pose_landmarks)
                        cam_label = f"Cam {self._camera_ids[idx]}"
                        cv2.putText(out, cam_label, (10, 28),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                        frames.append(out)

                    if frames:
                        # Kare boyutlarını eşitle ve yan yana göster
                        h = max(f.shape[0] for f in frames)
                        w = max(f.shape[1] for f in frames)
                        padded = []
                        for f in frames:
                            canvas = np.zeros((h, w, 3), dtype=np.uint8)
                            canvas[: f.shape[0], : f.shape[1]] = f
                            padded.append(canvas)
                        combined = np.hstack(padded)

                        bone_count = len(fused.bones) if fused else 0
                        cv2.putText(
                            combined,
                            f"FPS: {fps_display:.1f}  Bones: {bone_count}  "
                            f"UDP→{self._host}:{self._port}",
                            (10, combined.shape[0] - 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1,
                        )
                        cv2.imshow("STAR-1 | Dual Camera → Blender", combined)

                    if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q"), 27):
                        break
                else:
                    time.sleep(1 / 60)

        except KeyboardInterrupt:
            pass
        finally:
            for w in workers:
                w.stop()
            for w in workers:
                w.join(timeout=2.0)
            for cam in cameras:
                cam.release()
            for proc in processors:
                proc.close()
            sender.close()
            if preview:
                cv2.destroyAllWindows()
            print("[STAR-1] Durduruldu.")
