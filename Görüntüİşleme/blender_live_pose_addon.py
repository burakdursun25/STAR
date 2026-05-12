bl_info = {
    "name": "Live Pose Capture",
    "blender": (3, 0, 0),
    "category": "Animation",
    "version": (1, 0, 0),
    "author": "Pose Capture",
    "description": "Canlı pose capture - UDP üzerinden skeletal data alır",
}

import bpy
import json
import socket
import threading
import math
from mathutils import Quaternion, Euler, Vector

# ============================================================================
# GLOBAL STATE
# ============================================================================

g_socket = None
g_receiver_thread = None
g_running = False
g_last_frame_data = None
g_lock = threading.Lock()


# ============================================================================
# UDP RECEIVER
# ============================================================================

class UDPReceiver:
    """UDP socket'den iskelet verisi al"""
    
    def __init__(self, host="127.0.0.1", port=5006):
        self.host = host
        self.port = port
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.bind((host, port))
        self.socket.settimeout(0.5)
        print(f"[UDPReceiver] Listening on {host}:{port}")
    
    def receive_frame(self) -> dict:
        """Bir frame veri almayı dene"""
        try:
            data, addr = self.socket.recvfrom(65536)
            frame_data = json.loads(data.decode("utf-8"))
            return frame_data
        except socket.timeout:
            return None
        except Exception as e:
            print(f"[UDPReceiver Error] {e}")
            return None
    
    def close(self):
        self.socket.close()


class ReceiverThread(threading.Thread):
    """UDP verisi alan background thread"""
    
    def __init__(self, receiver: UDPReceiver):
        super().__init__(daemon=True)
        self.receiver = receiver
        self.running = True
    
    def run(self):
        """Thread loop - UDP'den veri al"""
        while self.running:
            frame_data = self.receiver.receive_frame()
            if frame_data:
                with g_lock:
                    global g_last_frame_data
                    g_last_frame_data = frame_data
    
    def stop(self):
        self.running = False


# ============================================================================
# BLENDER VERI İŞLEME
# ============================================================================

def euler_to_quaternion(euler_angles):
    """Euler açılarını (radians) Quaternion'a dönüştür"""
    e = Euler(euler_angles, 'XYZ')
    return e.to_quaternion()


def apply_bone_transform(armature_obj, bone_name, position, rotation):
    """Kemik transformasyonunu Blender armature'ına uygula"""
    if not armature_obj or armature_obj.type != 'ARMATURE':
        return False
    
    pose = armature_obj.pose
    
    if bone_name not in pose.bones:
        return False
    
    bone = pose.bones[bone_name]
    
    # Rotasyon uygula (Euler -> Quaternion)
    try:
        quat = euler_to_quaternion(rotation)
        bone.rotation_quaternion = quat
    except Exception as e:
        print(f"[Armature Update Error] {bone_name}: {e}")
        return False
    
    return True


def update_armature_from_skeleton(armature_obj, skeleton_data):
    """Tüm armature'ı skeleton data'dan güncelle"""
    if not armature_obj or armature_obj.type != 'ARMATURE':
        return 0
    
    updated_count = 0
    
    # skeleton_data bir dict: {"LeftArm": {...}, "RightArm": {...}, ...}
    for bone_name, bone_transform in skeleton_data.items():
        position = bone_transform.get("position", [0, 0, 0])
        rotation = bone_transform.get("rotation", [0, 0, 0])
        
        if apply_bone_transform(armature_obj, bone_name, position, rotation):
            updated_count += 1
    
    # Constraints'leri update et
    bpy.context.view_layer.update()
    
    return updated_count



# ============================================================================
# BLENDER PANEL VE OPERATORS
# ============================================================================

class POSECAPTURE_OT_start(bpy.types.Operator):
    """UDP receiver'ı başlat"""
    bl_idname = "posecapture.start"
    bl_label = "Başlat (Live Pose)"
    
    def execute(self, context):
        global g_socket, g_receiver_thread, g_running, g_last_frame_data
        
        if g_running:
            self.report({'WARNING'}, "Zaten çalışıyor")
            return {'FINISHED'}
        
        try:
            g_socket = UDPReceiver(
                host=context.scene.posecapture_host,
                port=context.scene.posecapture_port
            )
            
            g_receiver_thread = ReceiverThread(g_socket)
            g_receiver_thread.start()
            
            g_running = True
            g_last_frame_data = None
            
            # Timer işlev'ini kaydet
            wm = context.window_manager
            wm.modal_timer_remove(getattr(wm, 'posecapture_timer', None))
            wm.posecapture_timer = wm.event_timer_add(0.016, window=context.window)  # ~60 FPS
            
            self.report({'INFO'}, "Live Pose Capture başlatıldı")
            return {'FINISHED'}
        
        except Exception as e:
            self.report({'ERROR'}, f"Başlangıç hatası: {e}")
            return {'FINISHED'}


class POSECAPTURE_OT_stop(bpy.types.Operator):
    """UDP receiver'ı durdur"""
    bl_idname = "posecapture.stop"
    bl_label = "Durdur"
    
    def execute(self, context):
        global g_socket, g_receiver_thread, g_running
        
        if not g_running:
            self.report({'WARNING'}, "Zaten durmuş")
            return {'FINISHED'}
        
        try:
            g_running = False
            
            if g_receiver_thread:
                g_receiver_thread.stop()
                g_receiver_thread.join(timeout=1)
            
            if g_socket:
                g_socket.close()
            
            g_socket = None
            g_receiver_thread = None
            
            self.report({'INFO'}, "Live Pose Capture durduruldu")
            return {'FINISHED'}
        
        except Exception as e:
            self.report({'ERROR'}, f"Durdurma hatası: {e}")
            return {'FINISHED'}


class POSECAPTURE_OT_update(bpy.types.Operator):
    """Armature'ı skeleton frame'den güncelle"""
    bl_idname = "posecapture.update"
    bl_label = "Güncelle"
    
    # Timer event
    _timer = None
    
    def modal(self, context, event):
        if event.type == 'TIMER':
            if not g_running:
                wm = context.window_manager
                wm.event_timer_remove(self._timer)
                return {'FINISHED'}
            
            with g_lock:
                if g_last_frame_data:
                    # Custom property'den armature adını al
                    armature_name = context.scene.get("posecapture_armature")
                    if armature_name:
                        armature = bpy.data.objects.get(armature_name)
                        if armature and armature.type == 'ARMATURE':
                            # skeleton data'sını frame_data'dan al
                            skeleton_data = g_last_frame_data.get("skeleton", {})
                            if skeleton_data:
                                updated = update_armature_from_skeleton(armature, skeleton_data)
            
            return {'RUNNING_MODAL'}
        
        elif event.type in {'ESC'}:
            wm = context.window_manager
            wm.event_timer_remove(self._timer)
            return {'FINISHED'}
        
        return {'RUNNING_MODAL'}
    
    def execute(self, context):
        wm = context.window_manager
        self._timer = wm.event_timer_add(0.016, window=context.window)  # ~60 FPS
        wm.modal_handler_add(self)
        return {'RUNNING_MODAL'}


class POSECAPTURE_PT_panel(bpy.types.Panel):
    """Live Pose Capture kontrol paneli"""
    bl_label = "Live Pose Capture"
    bl_idname = "POSECAPTURE_PT_panel"
    bl_space_type = 'PROPERTIES'
    bl_region_type = 'WINDOW'
    bl_context = "object"
    
    def draw(self, context):
        layout = self.layout
        scene = context.scene
        
        # Başlık
        layout.label(text="Blender Canlı Pose Alma", icon='ARMATURE_DATA')
        
        layout.separator()
        
        # Ayarlar
        layout.label(text="Ayarlar:", icon='PREFERENCES')
        layout.prop(scene, "posecapture_host", text="Host")
        layout.prop(scene, "posecapture_port", text="Port")
        
        layout.separator()
        
        # Armature seçimi
        layout.label(text="Armature Seç:", icon='BONE_DATA')
        layout.prop(scene, "posecapture_armature", text="")
        
        layout.separator()
        
        # Kontrol butonları
        row = layout.row(align=True)
        row.operator("posecapture.start", text="▶ BAŞLAT", icon='PLAY')
        row.operator("posecapture.stop", text="⏹ DURDUR", icon='PAUSE')
        
        row = layout.row()
        row.operator("posecapture.update", text="Canlı Güncelle (Modal)", icon='ANIM')
        
        layout.separator()
        
        # Durum
        layout.label(text=f"Durum: {'🔴 AKTIF' if g_running else '🔵 DURMUŞ'}", icon='INFO')
        
        if g_last_frame_data:
            layout.label(text=f"Frame #: {g_last_frame_data.get('frame_number', '-')}")
            layout.label(text=f"Bones: {len(g_last_frame_data.get('bones', []))}")
            layout.label(text=f"Confidence: {g_last_frame_data.get('confidence', 0):.2%}")


# ============================================================================
# BLENDER HOOKS
# ============================================================================

def scene_update_handler(scene, depsgraph):
    """Scene update hook - armature'ı canlı güncelle"""
    global g_last_frame_data
    
    if not g_running or not g_last_frame_data:
        return
    
    with g_lock:
        frame_data = g_last_frame_data
    
    try:
        # PointerProperty'den armature'ı al
        armature = scene.posecapture_armature
        if armature and armature.type == 'ARMATURE':
            skeleton_data = frame_data.get("skeleton", {})
            if skeleton_data:
                update_armature_from_skeleton(armature, skeleton_data)
    except Exception as e:
        pass


# ============================================================================
# ADDON REGISTRATION
# ============================================================================

def register():
    """Addon register - sınıfları ve özellikleri kaydet"""
    
    # Sınıfları kaydet
    bpy.utils.register_class(POSECAPTURE_OT_start)
    bpy.utils.register_class(POSECAPTURE_OT_stop)
    bpy.utils.register_class(POSECAPTURE_OT_update)
    bpy.utils.register_class(POSECAPTURE_PT_panel)
    
    # Scene properties
    bpy.types.Scene.posecapture_host = bpy.props.StringProperty(
        name="Host",
        description="UDP receiver host",
        default="127.0.0.1"
    )
    bpy.types.Scene.posecapture_port = bpy.props.IntProperty(
        name="Port",
        description="UDP receiver port",
        default=5006,
        min=1024,
        max=65535
    )
    bpy.types.Scene.posecapture_armature = bpy.props.PointerProperty(
        name="Armature",
        description="Güncellenecek Armature",
        type=bpy.types.Object,
        poll=lambda self, obj: obj.type == 'ARMATURE'
    )
    
    # Scene update handler'ı kaydet
    bpy.app.handlers.depsgraph_update_post.append(scene_update_handler)
    
    print("[Live Pose Capture] Addon yüklendi")


def unregister():
    """Addon unregister - temizle"""
    global g_running, g_socket, g_receiver_thread
    
    # Çalışan thread'i durdur
    if g_running:
        g_running = False
        if g_receiver_thread:
            g_receiver_thread.stop()
            g_receiver_thread.join(timeout=1)
        if g_socket:
            g_socket.close()
    
    # Handler'ı kaldır
    bpy.app.handlers.depsgraph_update_post.remove(scene_update_handler)
    
    # Sınıfları kaldır
    bpy.utils.unregister_class(POSECAPTURE_OT_start)
    bpy.utils.unregister_class(POSECAPTURE_OT_stop)
    bpy.utils.unregister_class(POSECAPTURE_OT_update)
    bpy.utils.unregister_class(POSECAPTURE_PT_panel)
    
    print("[Live Pose Capture] Addon kaldırıldı")


if __name__ == "__main__":
    register()
