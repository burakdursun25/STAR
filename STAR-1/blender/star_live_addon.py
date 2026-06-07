"""
STAR-1 Live Pose Receiver — Blender 5.0 Addon
==============================================
UDP üzerinden gelen yön vektörlerini alır ve pbone.matrix üzerinden
Blender armature'ına doğrudan uygular.

Rotasyon hesabı Python tarafında yapılmaz.
Blender'ın kendi mathutils.Vector.to_track_quat('Y','Z') kullanılır.
Bu sayede rest pose farkı otomatik olarak düzeltilir.
"""

bl_info = {
    "name":        "STAR-1 Live Pose",
    "author":      "STAR-1 Project",
    "version":     (2, 1, 0),
    "blender":     (5, 0, 0),
    "location":    "3D Viewport › N-Panel › STAR Live",
    "description": "2 kamera kaynaklı canlı iskelet akışı (UDP) → Armature",
    "category":    "Animation",
}

import json
import socket
import struct
import threading
from typing import Dict, List, Optional

import bpy
import mathutils

# ─────────────────────────────────────────────────────────────────────────────
# UDP protokol sabitleri
# ─────────────────────────────────────────────────────────────────────────────
HEADER_SIZE    = 6        # 3 × uint16 big-endian: [seq][total][idx]
MAX_CHUNK      = 60_000
BUFFER_SIZE    = MAX_CHUNK + HEADER_SIZE + 64

# Kemikler kök→yaprak sırasına göre sıralı uygulanmalı.
# pbone.matrix parent'a bağlı olduğu için önce parent'ı set etmeliyiz.
BONE_ORDER = [
    "hips",
    "spine", "chest", "neck",
    "upper_arm.L", "forearm.L",
    "upper_arm.R", "forearm.R",
    "thigh.L", "shin.L", "foot.L",
    "thigh.R", "shin.R", "foot.R",
    "palm",
    "thumb.1", "thumb.2", "thumb.3",
    "index.1", "index.2", "index.3",
    "middle.1", "middle.2", "middle.3",
    "ring.1", "ring.2", "ring.3",
    "pinky.1", "pinky.2", "pinky.3",
]


# ─────────────────────────────────────────────────────────────────────────────
# Thread-safe alıcı durumu
# ─────────────────────────────────────────────────────────────────────────────
class _State:
    def __init__(self):
        self.lock     = threading.Lock()
        self.latest   : Optional[dict] = None
        self.running  : bool = False
        self.sock     : Optional[socket.socket] = None
        self._chunks  : Dict[int, Dict[int, bytes]] = {}
        self._totals  : Dict[int, int] = {}

    def store(self, seq: int, total: int, idx: int, data: bytes) -> Optional[bytes]:
        if seq not in self._chunks:
            self._chunks[seq] = {}
            self._totals[seq] = total
        self._chunks[seq][idx] = data
        if len(self._chunks[seq]) == self._totals[seq]:
            payload = b"".join(self._chunks[seq][i] for i in range(self._totals[seq]))
            del self._chunks[seq]
            del self._totals[seq]
            # Eski eksik paketleri temizle
            for old in list(self._chunks)[:8]:
                self._chunks.pop(old, None)
                self._totals.pop(old, None)
            return payload
        return None


_state = _State()


# ─────────────────────────────────────────────────────────────────────────────
# UDP dinleyici thread
# ─────────────────────────────────────────────────────────────────────────────
def _receiver(port: int) -> None:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 20)
    sock.settimeout(0.5)
    sock.bind(("0.0.0.0", port))
    with _state.lock:
        _state.sock = sock

    while True:
        with _state.lock:
            if not _state.running:
                break
        try:
            raw, _ = sock.recvfrom(BUFFER_SIZE)
        except socket.timeout:
            continue
        except Exception:
            break
        if len(raw) < HEADER_SIZE:
            continue
        seq, total, idx = struct.unpack("!HHH", raw[:HEADER_SIZE])
        with _state.lock:
            payload = _state.store(seq, total, idx, raw[HEADER_SIZE:])
            if payload is not None:
                try:
                    _state.latest = json.loads(payload.decode("utf-8"))
                except Exception:
                    pass

    sock.close()
    with _state.lock:
        _state.sock = None


# ─────────────────────────────────────────────────────────────────────────────
# Kemik rotasyon uygulayıcı — ana algoritma
# ─────────────────────────────────────────────────────────────────────────────
def _set_bone(arm_obj: bpy.types.Object, pbone, direction: list, confidence: float) -> None:
    """
    Pose bone'u verilen yön vektörüne göre döndürür.

    Algoritma:
    1. Yön vektörünü Blender dünya uzayından armature yerel uzayına çevir.
    2. to_track_quat('Y','Z') ile yerel +Y ekseni yön vektörüne hizalansın.
       'Z' referansı kemik etrafındaki dönmeyi (roll) minimize eder.
    3. Hedef matriksi pbone.bone.head_local konumuyla birleştir.
    4. pbone.matrix'e ata — Blender parent zincirini otomatik hesaplar.
    """
    if confidence < 0.25:
        return

    d_world = mathutils.Vector(direction)
    if d_world.length_squared < 1e-6:
        return
    d_world.normalize()

    # Dünya → armature yerel uzayı
    arm_inv = arm_obj.matrix_world.inverted()
    d_local = (arm_inv.to_3x3() @ d_world).normalized()

    # Yerel +Y → d_local yönlendir, +Z yukarı referans (roll = 0)
    quat = d_local.to_track_quat('Y', 'Z')

    # Armature uzayında hedef matris: sadece rotasyon + kemiğin rest konumu
    mat = quat.to_matrix().to_4x4()
    mat.translation = pbone.bone.head_local

    pbone.matrix = mat


def _apply_bones(arm_obj: bpy.types.Object, bone_dict: dict) -> None:
    """Kök→yaprak sırasına göre kemikleri uygula."""
    # Önce sıralı listeden geçerek mevcut kemikleri uygula
    applied = set()
    for name in BONE_ORDER:
        info = bone_dict.get(name)
        if info is None:
            continue
        pbone = arm_obj.pose.bones.get(name)
        if pbone is None:
            continue
        _set_bone(arm_obj, pbone, info["direction"], float(info.get("confidence", 1.0)))
        applied.add(name)

    # Sıralamada olmayan ama gelen kemikler (el parmakları vb.)
    for name, info in bone_dict.items():
        if name in applied:
            continue
        pbone = arm_obj.pose.bones.get(name)
        if pbone is None:
            continue
        _set_bone(arm_obj, pbone, info["direction"], float(info.get("confidence", 1.0)))


# ─────────────────────────────────────────────────────────────────────────────
# Blender timer callback (~60 Hz)
# ─────────────────────────────────────────────────────────────────────────────
def _timer() -> Optional[float]:
    props = bpy.context.scene.star_live_props
    if not props.is_running:
        return None  # timer'ı durdurur

    with _state.lock:
        data = _state.latest
        _state.latest = None

    if data is None:
        return 1 / 60

    arm_obj = bpy.data.objects.get(props.armature_name)
    if arm_obj is None or arm_obj.type != "ARMATURE":
        return 1 / 60

    _apply_bones(arm_obj, data.get("skeleton", {}))
    _apply_bones(arm_obj, data.get("left_hand", {}))
    _apply_bones(arm_obj, data.get("right_hand", {}))

    # Parent zincirini güncelleyerek child kemiğinin matrisini doğru hesaplat
    bpy.context.view_layer.update()

    # Viewport'u yenile
    for area in bpy.context.screen.areas:
        if area.type == "VIEW_3D":
            area.tag_redraw()
            break

    return 1 / 60


# ─────────────────────────────────────────────────────────────────────────────
# Scene özellikleri
# ─────────────────────────────────────────────────────────────────────────────
class StarLiveProperties(bpy.types.PropertyGroup):
    armature_name: bpy.props.StringProperty(  # type: ignore[valid-type]
        name="Armature",
        description="Sahnedeki armature objesinin adı",
        default="Armature",
    )
    port: bpy.props.IntProperty(  # type: ignore[valid-type]
        name="Port",
        default=7777, min=1024, max=65535,
    )
    is_running: bpy.props.BoolProperty(default=False)  # type: ignore[valid-type]


# ─────────────────────────────────────────────────────────────────────────────
# Operatörler
# ─────────────────────────────────────────────────────────────────────────────
class STAR_OT_Start(bpy.types.Operator):
    bl_idname      = "star.start_receiver"
    bl_label       = "Dinlemeyi Başlat"
    bl_description = "UDP'den iskelet verisi almaya başla"

    def execute(self, context):
        props = context.scene.star_live_props
        if props.is_running:
            self.report({"WARNING"}, "Zaten çalışıyor.")
            return {"CANCELLED"}

        arm_obj = bpy.data.objects.get(props.armature_name)
        if arm_obj is None or arm_obj.type != "ARMATURE":
            self.report({"ERROR"}, f'"{props.armature_name}" adında bir Armature bulunamadı.')
            return {"CANCELLED"}

        with _state.lock:
            _state.running = True

        t = threading.Thread(target=_receiver, args=(props.port,), daemon=True)
        t.start()

        props.is_running = True
        bpy.app.timers.register(_timer, persistent=True)
        self.report({"INFO"}, f"STAR-1: UDP:{props.port} dinleniyor")
        return {"FINISHED"}


class STAR_OT_Stop(bpy.types.Operator):
    bl_idname      = "star.stop_receiver"
    bl_label       = "Dinlemeyi Durdur"

    def execute(self, context):
        props = context.scene.star_live_props
        with _state.lock:
            _state.running = False
            if _state.sock:
                _state.sock.close()
        props.is_running = False
        if bpy.app.timers.is_registered(_timer):
            bpy.app.timers.unregister(_timer)
        self.report({"INFO"}, "STAR-1: Durduruldu.")
        return {"FINISHED"}


# ─────────────────────────────────────────────────────────────────────────────
# N-Panel
# ─────────────────────────────────────────────────────────────────────────────
class STAR_PT_Panel(bpy.types.Panel):
    bl_label       = "STAR-1 Live"
    bl_idname      = "STAR_PT_panel"
    bl_space_type  = "VIEW_3D"
    bl_region_type = "UI"
    bl_category    = "STAR Live"

    def draw(self, context):
        layout = self.layout
        props  = context.scene.star_live_props

        layout.label(text="2-Kamera Canlı İskelet Akışı", icon="ARMATURE_DATA")
        layout.separator()
        col = layout.column(align=True)
        col.prop(props, "armature_name", icon="OUTLINER_OB_ARMATURE")
        col.prop(props, "port",          icon="NETWORK_DRIVE")
        layout.separator()

        if props.is_running:
            layout.operator("star.stop_receiver", icon="PAUSE")
            box = layout.box()
            box.label(text="Dinleniyor...", icon="REC")
        else:
            layout.operator("star.start_receiver", icon="PLAY")

        layout.separator()
        layout.label(text="python -m star_live", icon="CONSOLE")


# ─────────────────────────────────────────────────────────────────────────────
# Kayıt / silme
# ─────────────────────────────────────────────────────────────────────────────
_CLASSES = [StarLiveProperties, STAR_OT_Start, STAR_OT_Stop, STAR_PT_Panel]


def register():
    for cls in _CLASSES:
        bpy.utils.register_class(cls)
    bpy.types.Scene.star_live_props = bpy.props.PointerProperty(type=StarLiveProperties)


def unregister():
    with _state.lock:
        _state.running = False
        if _state.sock:
            _state.sock.close()
    if bpy.app.timers.is_registered(_timer):
        bpy.app.timers.unregister(_timer)
    for cls in reversed(_CLASSES):
        bpy.utils.unregister_class(cls)
    del bpy.types.Scene.star_live_props


if __name__ == "__main__":
    register()
