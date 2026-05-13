"""
Blender Live Pose Capture Addon
UDP port 5005'ten MediaPipe verilerini alır ve iskeleti hareket ettirir.
"""

import bpy
import json
import socket
import threading
import mathutils
import math
from bpy.props import BoolProperty, PointerProperty
from bpy.types import Panel, Operator, PropertyGroup

bl_info = {
    "name": "Live Pose Capture",
    "author": "Custom",
    "version": (1, 0, 0),
    "blender": (4, 0, 0),
    "location": "3D View > Sidebar > Live Pose",
    "description": "UDP üzerinden MediaPipe pose verisi alır ve iskeleti hareket ettirir",
    "category": "Animation",
}

UDP_HOST = "127.0.0.1"
UDP_PORT = 5005

# ============================================================
# Global state
# ============================================================
_receiver_thread = None
_running = False
_latest_data = None
_data_lock = threading.Lock()


# ============================================================
# UDP Receiver Thread
# ============================================================
def udp_receiver():
    global _running, _latest_data

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.settimeout(1.0)

    try:
        sock.bind((UDP_HOST, UDP_PORT))
        print(f"[LivePose] UDP dinleniyor: {UDP_HOST}:{UDP_PORT}")
    except Exception as e:
        print(f"[LivePose] UDP bind hatası: {e}")
        return

    while _running:
        try:
            data, _ = sock.recvfrom(65535)
            parsed = json.loads(data.decode("utf-8"))
            with _data_lock:
                _latest_data = parsed
        except socket.timeout:
            continue
        except Exception as e:
            if _running:
                print(f"[LivePose] Veri alma hatası: {e}")

    sock.close()
    print("[LivePose] UDP kapatıldı.")


# ============================================================
# Bone Updater (Blender timer callback)
# ============================================================
def update_bones():
    global _latest_data

    if not _running:
        return None  # Timer'ı durdur

    with _data_lock:
        data = _latest_data
        _latest_data = None

    if data is None:
        return 0.016  # ~60fps, veri yoksa bekle

    # Skeleton verisini al (FusedFrameResult veya FrameResult)
    skeleton = data.get("skeleton")
    if not skeleton:
        # Skeleton yoksa pose_landmarks'tan hesapla
        pose_landmarks = data.get("pose_landmarks", [])
        if not pose_landmarks:
            return 0.016
        skeleton = build_skeleton_from_landmarks(pose_landmarks)

    # Armature'ı bul
    scene = bpy.context.scene
    armature_obj = scene.get("posecapture_armature_name")
    if armature_obj:
        arm_obj = bpy.data.objects.get(armature_obj)
    else:
        # İlk armature'ı bul
        arm_obj = next((o for o in bpy.data.objects if o.type == 'ARMATURE'), None)

    if arm_obj is None:
        return 0.016

    # Pose mode'da kemikleri güncelle
    try:
        apply_skeleton_to_armature(arm_obj, skeleton)
    except Exception as e:
        print(f"[LivePose] Kemik güncelleme hatası: {e}")

    return 0.016  # 60fps


def build_skeleton_from_landmarks(landmarks):
    """Pose landmarks listesinden skeleton dict oluştur"""
    if len(landmarks) < 29:
        return {}

    def lm(i):
        p = landmarks[i]
        if isinstance(p, dict):
            return [p.get("x", 0), p.get("y", 0), p.get("z", 0)]
        return [0, 0, 0]

    def vec(a, b):
        return [b[0]-a[0], b[1]-a[1], b[2]-a[2]]

    def euler_from_vec(v):
        n = math.sqrt(v[0]**2 + v[1]**2 + v[2]**2)
        if n < 1e-6:
            return [0, 0, 0]
        v = [v[0]/n, v[1]/n, v[2]/n]
        pitch = math.asin(-v[1])
        yaw = math.atan2(v[0], v[2])
        return [pitch, yaw, 0]

    left_hip  = lm(23)
    right_hip = lm(24)
    hip_center = [(left_hip[0]+right_hip[0])/2,
                  (left_hip[1]+right_hip[1])/2,
                  (left_hip[2]+right_hip[2])/2]

    bones = {
        "Hips":        {"position": hip_center,  "rotation": [0, 0, 0]},
        "LeftUpLeg":   {"position": lm(23), "rotation": euler_from_vec(vec(lm(23), lm(25)))},
        "LeftLeg":     {"position": lm(25), "rotation": euler_from_vec(vec(lm(25), lm(27)))},
        "LeftFoot":    {"position": lm(27), "rotation": [0, 0, 0]},
        "RightUpLeg":  {"position": lm(24), "rotation": euler_from_vec(vec(lm(24), lm(26)))},
        "RightLeg":    {"position": lm(26), "rotation": euler_from_vec(vec(lm(26), lm(28)))},
        "RightFoot":   {"position": lm(28), "rotation": [0, 0, 0]},
        "LeftArm":     {"position": lm(11), "rotation": euler_from_vec(vec(lm(11), lm(13)))},
        "LeftForeArm": {"position": lm(13), "rotation": euler_from_vec(vec(lm(13), lm(15)))},
        "LeftHand":    {"position": lm(15), "rotation": [0, 0, 0]},
        "RightArm":    {"position": lm(12), "rotation": euler_from_vec(vec(lm(12), lm(14)))},
        "RightForeArm":{"position": lm(14), "rotation": euler_from_vec(vec(lm(14), lm(16)))},
        "RightHand":   {"position": lm(16), "rotation": [0, 0, 0]},
        "Head":        {"position": lm(0),  "rotation": [0, 0, 0]},
    }
    return bones


def apply_skeleton_to_armature(arm_obj, skeleton):
    """Skeleton verilerini armature'a uygula"""
    if arm_obj.mode != 'POSE':
        # Pose mode'a geçmeye zorla
        prev_active = bpy.context.view_layer.objects.active
        bpy.context.view_layer.objects.active = arm_obj
        bpy.ops.object.mode_set(mode='POSE')
        bpy.context.view_layer.objects.active = prev_active

    pose_bones = arm_obj.pose.bones

    SCALE = 2.0  # Koordinat ölçeği

    for bone_name, transform in skeleton.items():
        if bone_name not in pose_bones:
            continue

        pbone = pose_bones[bone_name]
        rot = transform.get("rotation", [0, 0, 0])

        # Rotasyonu uygula
        pbone.rotation_mode = 'XYZ'
        pbone.rotation_euler = mathutils.Euler((
            rot[0],
            rot[1],
            rot[2]
        ), 'XYZ')

    # Viewport'u güncelle
    arm_obj.update_tag()
    bpy.context.view_layer.update()


# ============================================================
# Scene Property
# ============================================================
class PoseCaptureProperties(PropertyGroup):
    is_running: BoolProperty(
        name="Çalışıyor",
        default=False
    )


# ============================================================
# Operators
# ============================================================
class POSEBOT_OT_Start(Operator):
    bl_idname = "posebot.start"
    bl_label = "BAŞLAT"
    bl_description = "UDP dinlemeyi başlat ve iskeleti canlı güncelle"

    def execute(self, context):
        global _receiver_thread, _running

        if _running:
            self.report({'WARNING'}, "Zaten çalışıyor!")
            return {'CANCELLED'}

        _running = True
        context.scene.posecapture_props.is_running = True

        # UDP thread başlat
        _receiver_thread = threading.Thread(target=udp_receiver, daemon=True)
        _receiver_thread.start()

        # Blender timer başlat
        bpy.app.timers.register(update_bones, first_interval=0.016)

        self.report({'INFO'}, f"LivePose başlatıldı! UDP:{UDP_PORT}")
        print(f"[LivePose] Başlatıldı. Şimdi Python scriptini çalıştır:")
        print(f"  python image_processor_oop.py")
        return {'FINISHED'}


class POSEBOT_OT_Stop(Operator):
    bl_idname = "posebot.stop"
    bl_label = "DURDUR"
    bl_description = "UDP dinlemeyi durdur"

    def execute(self, context):
        global _running

        _running = False
        context.scene.posecapture_props.is_running = False

        self.report({'INFO'}, "LivePose durduruldu.")
        return {'FINISHED'}


# ============================================================
# Panel
# ============================================================
class POSEBOT_PT_Panel(Panel):
    bl_label = "Live Pose Capture"
    bl_idname = "POSEBOT_PT_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Live Pose"

    def draw(self, context):
        layout = self.layout
        props = context.scene.posecapture_props

        # Durum göstergesi
        if props.is_running:
            layout.label(text="● Canlı - Veri Bekleniyor", icon='REC')
        else:
            layout.label(text="○ Durduruldu", icon='PAUSE')

        layout.separator()

        # Başlat / Durdur butonları
        if not props.is_running:
            layout.operator("posebot.start", icon='PLAY')
        else:
            layout.operator("posebot.stop", icon='PAUSE')

        layout.separator()
        layout.label(text=f"UDP Port: {UDP_PORT}", icon='INFO')

        # Armature bilgisi
        arm = next((o for o in bpy.data.objects if o.type == 'ARMATURE'), None)
        if arm:
            layout.label(text=f"Armature: {arm.name}", icon='ARMATURE_DATA')
        else:
            layout.label(text="Armature bulunamadı!", icon='ERROR')


# ============================================================
# Register / Unregister
# ============================================================
classes = [
    PoseCaptureProperties,
    POSEBOT_OT_Start,
    POSEBOT_OT_Stop,
    POSEBOT_PT_Panel,
]


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.posecapture_props = PointerProperty(type=PoseCaptureProperties)
    # Eski property adıyla uyumluluk
    bpy.types.Scene.posecapture_armature = PointerProperty(
        type=bpy.types.Object,
        name="Armature",
        poll=lambda self, obj: obj.type == 'ARMATURE'
    )
    print("[LivePose] Addon register edildi.")


def unregister():
    global _running
    _running = False

    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)

    if hasattr(bpy.types.Scene, "posecapture_props"):
        del bpy.types.Scene.posecapture_props
    if hasattr(bpy.types.Scene, "posecapture_armature"):
        del bpy.types.Scene.posecapture_armature


if __name__ == "__main__":
    register()