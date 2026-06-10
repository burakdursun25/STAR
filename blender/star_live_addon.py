"""
STAR-1 Live Pose Receiver — Blender 5.0 Addon
==============================================
UDP üzerinden gelen yön vektörlerini alır ve herhangi bir
Blender armature'ına uygular.

YENİ: Bone Mapping desteği — kendi karakterinizin kemik isimlerini
      STAR-1'in kemik isimleriyle eşleştirebilirsiniz.
"""

bl_info = {
    "name":        "STAR-1 Live Pose",
    "author":      "STAR-1 Project",
    "version":     (2, 2, 0),
    "blender":     (5, 0, 0),
    "location":    "3D Viewport > N-Panel > STAR Live",
    "description": "Canlı iskelet akışı (UDP) → herhangi bir Armature (bone mapping destekli)",
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
# Sabitler
# ─────────────────────────────────────────────────────────────────────────────
HEADER_SIZE = 6
MAX_CHUNK   = 60_000
BUFFER_SIZE = MAX_CHUNK + HEADER_SIZE + 64

# STAR-1'in ürettiği kemik isimleri (kök→yaprak sırası)
STAR_BONE_ORDER = [
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

# Rigify metarig için otomatik mapping (STAR-1 adı → Rigify kemiği)
RIGIFY_MAPPING = {
    "hips":       "spine",
    "spine":      "spine.001",
    "chest":      "spine.003",
    "neck":       "spine.004",
    "upper_arm.L":"upper_arm.L",
    "forearm.L":  "forearm.L",
    "upper_arm.R":"upper_arm.R",
    "forearm.R":  "forearm.R",
    "thigh.L":    "thigh.L",
    "shin.L":     "shin.L",
    "foot.L":     "foot.L",
    "thigh.R":    "thigh.R",
    "shin.R":     "shin.R",
    "foot.R":     "foot.R",
}

# Mixamo için otomatik mapping
MIXAMO_MAPPING = {
    "hips":       "mixamorig:Hips",
    "spine":      "mixamorig:Spine",
    "chest":      "mixamorig:Spine1",
    "neck":       "mixamorig:Neck",
    "upper_arm.L":"mixamorig:RightArm",
    "forearm.L":  "mixamorig:RightForeArm",
    "upper_arm.R":"mixamorig:LeftArm",
    "forearm.R":  "mixamorig:LeftForeArm",
    "thigh.L":    "mixamorig:RightUpLeg",
    "shin.L":     "mixamorig:RightLeg",
    "foot.L":     "mixamorig:RightFoot",
    "thigh.R":    "mixamorig:LeftUpLeg",
    "shin.R":     "mixamorig:LeftLeg",
    "foot.R":     "mixamorig:LeftFoot",
}


# ─────────────────────────────────────────────────────────────────────────────
# Yardımcı: Armature nesnesini bul (linked library dahil)
# ─────────────────────────────────────────────────────────────────────────────
def _find_armature(name: str) -> Optional[bpy.types.Object]:
    """
    bpy.data.objects içinde tam eşleşme arar.
    Bulamazsa bpy.context.scene içindeki tüm nesneleri tarar (linked dahil).
    """
    # 1. Doğrudan isimle bul
    obj = bpy.data.objects.get(name)
    if obj and obj.type == "ARMATURE":
        return obj

    # 2. Tüm sahne nesnelerini tara (linked library nesneleri dahil)
    for obj in bpy.context.scene.objects:
        if obj.type == "ARMATURE" and (obj.name == name or obj.name.split(".")[0] == name):
            return obj

    # 3. Tüm bpy.data.objects tarama (başka sahnelerdekiler dahil)
    for obj in bpy.data.objects:
        if obj.type == "ARMATURE" and obj.name == name:
            return obj

    return None


def _get_all_armatures() -> List[str]:
    """Sahnedeki tüm armature isimlerini döner."""
    names = []
    for obj in bpy.context.scene.objects:
        if obj.type == "ARMATURE":
            names.append(obj.name)
    return names


def _parse_mapping(mapping_json: str) -> Dict[str, str]:
    """JSON mapping string'ini dict'e çevirir. Hata durumunda boş dict döner."""
    if not mapping_json.strip():
        return {}
    try:
        result = json.loads(mapping_json)
        if isinstance(result, dict):
            return {str(k): str(v) for k, v in result.items()}
    except Exception:
        pass
    return {}


# ─────────────────────────────────────────────────────────────────────────────
# Thread-safe alıcı durumu
# ─────────────────────────────────────────────────────────────────────────────
class _State:
    def __init__(self):
        self.lock    = threading.Lock()
        self.latest  : Optional[dict] = None
        self.running : bool = False
        self.sock    : Optional[socket.socket] = None
        self._chunks : Dict[int, Dict[int, bytes]] = {}
        self._totals : Dict[int, int] = {}

    def store(self, seq: int, total: int, idx: int, data: bytes) -> Optional[bytes]:
        if seq not in self._chunks:
            self._chunks[seq] = {}
            self._totals[seq] = total
        self._chunks[seq][idx] = data
        if len(self._chunks[seq]) == self._totals[seq]:
            payload = b"".join(self._chunks[seq][i] for i in range(self._totals[seq]))
            del self._chunks[seq]
            del self._totals[seq]
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
# Kemik rotasyon uygulayıcı
# ─────────────────────────────────────────────────────────────────────────────
def _set_bone(arm_obj: bpy.types.Object, pbone, direction: list, confidence: float) -> None:
    if confidence < 0.25:
        return
    d_world = mathutils.Vector(direction)
    if d_world.length_squared < 1e-6:
        return
    d_world.normalize()
    arm_inv = arm_obj.matrix_world.inverted()
    d_local = (arm_inv.to_3x3() @ d_world).normalized()
    quat = d_local.to_track_quat('Y', 'Z')
    mat = quat.to_matrix().to_4x4()
    mat.translation = pbone.bone.head_local
    pbone.matrix = mat


def _apply_bones(arm_obj: bpy.types.Object, bone_dict: dict, mapping: Dict[str, str]) -> None:
    """
    STAR-1 kemik verilerini armature'a uygular.
    mapping: {star_bone_name: rig_bone_name}
    Mapping boşsa STAR-1 isimlerini doğrudan kullanır.
    """
    applied = set()

    def _apply_one(star_name: str, info: dict):
        # Mapping varsa rig'deki ismi bul, yoksa star_name kullan
        rig_name = mapping.get(star_name, star_name)
        pbone = arm_obj.pose.bones.get(rig_name)
        if pbone is None:
            return
        _set_bone(arm_obj, pbone, info["direction"], float(info.get("confidence", 1.0)))
        applied.add(star_name)

    # Önce sıralı (kök→yaprak) uygula
    for star_name in STAR_BONE_ORDER:
        info = bone_dict.get(star_name)
        if info is not None:
            _apply_one(star_name, info)

    # Sıralamada olmayan kemikler (parmaklar vb.)
    for star_name, info in bone_dict.items():
        if star_name not in applied:
            _apply_one(star_name, info)


# ─────────────────────────────────────────────────────────────────────────────
# Blender timer callback (~60 Hz)
# ─────────────────────────────────────────────────────────────────────────────
def _timer() -> Optional[float]:
    props = bpy.context.scene.star_live_props
    if not props.is_running:
        return None

    with _state.lock:
        data = _state.latest
        _state.latest = None

    if data is None:
        return 1 / 60

    arm_obj = _find_armature(props.armature_name)
    if arm_obj is None:
        return 1 / 60

    mapping = _parse_mapping(props.bone_mapping)

    _apply_bones(arm_obj, data.get("skeleton",   {}), mapping)
    _apply_bones(arm_obj, data.get("left_hand",  {}), mapping)
    _apply_bones(arm_obj, data.get("right_hand", {}), mapping)

    bpy.context.view_layer.update()
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
    bone_mapping: bpy.props.StringProperty(  # type: ignore[valid-type]
        name="Bone Mapping",
        description=(
            'JSON: {"star_kemik": "rig_kemik"}. '
            'Boş bırakılırsa STAR-1 kemik isimlerini doğrudan kullanır.'
        ),
        default="",
    )
    show_mapping: bpy.props.BoolProperty(  # type: ignore[valid-type]
        name="Bone Mapping Göster",
        default=False,
    )
    is_running: bpy.props.BoolProperty(default=False)  # type: ignore[valid-type]


# ─────────────────────────────────────────────────────────────────────────────
# Operatörler
# ─────────────────────────────────────────────────────────────────────────────
class STAR_OT_Start(bpy.types.Operator):
    bl_idname      = "star.start_receiver"
    bl_label       = "Dinlemeyi Baslat"
    bl_description = "UDP'den iskelet verisi almaya basla"

    def execute(self, context):
        props = context.scene.star_live_props
        if props.is_running:
            self.report({"WARNING"}, "Zaten calisiyor.")
            return {"CANCELLED"}

        arm_obj = _find_armature(props.armature_name)
        if arm_obj is None:
            # Tüm armature isimlerini listele
            available = _get_all_armatures()
            msg = f'"{props.armature_name}" bulunamadi.'
            if available:
                msg += f' Mevcut: {", ".join(available)}'
            else:
                msg += " Sahnede hic armature yok."
            self.report({"ERROR"}, msg)
            return {"CANCELLED"}

        with _state.lock:
            _state.running = True

        t = threading.Thread(target=_receiver, args=(props.port,), daemon=True)
        t.start()

        props.is_running = True
        bpy.app.timers.register(_timer, persistent=True)
        self.report({"INFO"}, f"STAR-1: '{arm_obj.name}' → UDP:{props.port} dinleniyor")
        return {"FINISHED"}


class STAR_OT_Stop(bpy.types.Operator):
    bl_idname = "star.stop_receiver"
    bl_label  = "Durdur"

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


class STAR_OT_DetectArmature(bpy.types.Operator):
    """Sahnedeki ilk armature'u otomatik seç"""
    bl_idname = "star.detect_armature"
    bl_label  = "Armature'u Otomatik Bul"

    def execute(self, context):
        props = context.scene.star_live_props
        names = _get_all_armatures()
        if not names:
            self.report({"ERROR"}, "Sahnede armature bulunamadi!")
            return {"CANCELLED"}
        # Aktif nesne armature ise onu kullan, yoksa ilkini al
        active = context.active_object
        if active and active.type == "ARMATURE":
            props.armature_name = active.name
            chosen = active.name
        else:
            props.armature_name = names[0]
            chosen = names[0]

        # Kemik isimlerini konsola yaz (mapping için referans)
        arm = _find_armature(props.armature_name)
        if arm:
            bones = [b.name for b in arm.data.bones]
            print(f"\n[STAR-1] Secilen armature: '{chosen}'")
            print(f"[STAR-1] Kemikler ({len(bones)} adet):")
            print("  " + ", ".join(bones))
            print(f"\n[STAR-1] STAR-1 kemik isimleri: {STAR_BONE_ORDER}\n")

        self.report({"INFO"}, f"Secildi: '{chosen}' | Kemikler konsola yazildi.")
        return {"FINISHED"}


class STAR_OT_ApplyRigifyMapping(bpy.types.Operator):
    """Rigify metarig icin hazir bone mapping yukle"""
    bl_idname = "star.apply_rigify_mapping"
    bl_label  = "Rigify Mapping Yukle"

    def execute(self, context):
        props = context.scene.star_live_props
        props.bone_mapping = json.dumps(RIGIFY_MAPPING, ensure_ascii=False)
        self.report({"INFO"}, "Rigify mapping yuklendi. Gerekirse duzenleyin.")
        return {"FINISHED"}


class STAR_OT_ApplyMixamoMapping(bpy.types.Operator):
    """Mixamo rigi icin hazir bone mapping yukle"""
    bl_idname = "star.apply_mixamo_mapping"
    bl_label  = "Mixamo Mapping Yukle"

    def execute(self, context):
        props = context.scene.star_live_props
        props.bone_mapping = json.dumps(MIXAMO_MAPPING, ensure_ascii=False)
        self.report({"INFO"}, "Mixamo mapping yuklendi. Gerekirse duzenleyin.")
        return {"FINISHED"}


class STAR_OT_ClearMapping(bpy.types.Operator):
    """Bone mapping'i temizle (STAR-1 isimlerini dogrudan kullan)"""
    bl_idname = "star.clear_mapping"
    bl_label  = "Mapping Temizle"

    def execute(self, context):
        context.scene.star_live_props.bone_mapping = ""
        self.report({"INFO"}, "Mapping temizlendi. STAR-1 isimleri dogrudan kullanilacak.")
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

        layout.label(text="2-Kamera Canli Iskelet Akisi", icon="ARMATURE_DATA")
        layout.separator()

        # ── Armature ──────────────────────────────────────────────────────────
        box = layout.box()
        box.label(text="Armature", icon="OUTLINER_OB_ARMATURE")
        row = box.row(align=True)
        row.prop_search(props, "armature_name", bpy.data, "objects", text="")
        row.operator("star.detect_armature", text="", icon="VIEWZOOM")
        box.prop(props, "port", icon="NETWORK_DRIVE")

        # ── Bone Mapping ──────────────────────────────────────────────────────
        box2 = layout.box()
        row2 = box2.row()
        row2.prop(props, "show_mapping",
                  icon="TRIA_DOWN" if props.show_mapping else "TRIA_RIGHT",
                  emboss=False)
        row2.label(text="Bone Mapping")

        if props.show_mapping:
            # Hızlı mapping butonları
            row3 = box2.row(align=True)
            row3.operator("star.apply_rigify_mapping", text="Rigify")
            row3.operator("star.apply_mixamo_mapping", text="Mixamo")
            row3.operator("star.clear_mapping",        text="", icon="X")

            # Durum
            mapping = _parse_mapping(props.bone_mapping)
            if mapping:
                box2.label(text=f"{len(mapping)} kemik eslestirmesi aktif", icon="CHECKMARK")
            else:
                box2.label(text="Mapping yok — STAR-1 isimleri kullaniliyor", icon="INFO")

            # Düzenlenebilir alan
            box2.label(text='JSON: {"star_kemik": "rig_kemik"}')
            box2.prop(props, "bone_mapping", text="")

        # ── Kontrol butonları ─────────────────────────────────────────────────
        layout.separator()
        if props.is_running:
            layout.operator("star.stop_receiver", icon="PAUSE")
            row_s = layout.row()
            row_s.alert = True
            row_s.label(text="Dinleniyor...", icon="REC")
        else:
            layout.operator("star.start_receiver", icon="PLAY")

        layout.separator()
        layout.label(text="python -m star_live", icon="CONSOLE")


# ─────────────────────────────────────────────────────────────────────────────
# Kayıt / silme
# ─────────────────────────────────────────────────────────────────────────────
_CLASSES = [
    StarLiveProperties,
    STAR_OT_Start,
    STAR_OT_Stop,
    STAR_OT_DetectArmature,
    STAR_OT_ApplyRigifyMapping,
    STAR_OT_ApplyMixamoMapping,
    STAR_OT_ClearMapping,
    STAR_PT_Panel,
]


def register():
    for cls in _CLASSES:
        bpy.utils.register_class(cls)
    bpy.types.Scene.star_live_props = bpy.props.PointerProperty(type=StarLiveProperties)
    print("[STAR-1] Live Pose Addon v2.2 kayit edildi.")


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
    print("[STAR-1] Live Pose Addon kaldiruldi.")


if __name__ == "__main__":
    register()
