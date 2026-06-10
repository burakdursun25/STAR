"""
STAR-1 Armature Oluşturucu
===========================
Blender 5.0  →  Scripting sekmesi  →  bu kodu yapıştır  →  Run Script

Oluşturur:
  • Object adı : "Armature"   ← addon'daki varsayılan isimle eşleşiyor
  • 14 gövde kemiği + 2×16 el kemiği
  • Tüm kemik isimleri skeleton.py'daki BONE_PAIRS ile birebir aynı
"""

import bpy
import math
import mathutils

# ─── Varsa eski objeyi temizle ───────────────────────────────────────────────
for obj in list(bpy.data.objects):
    if obj.name == "Armature":
        bpy.data.objects.remove(obj, do_unlink=True)
for arm in list(bpy.data.armatures):
    if arm.name == "STAR_Rig":
        bpy.data.armatures.remove(arm)

# ─── Armature oluştur ────────────────────────────────────────────────────────
arm_data = bpy.data.armatures.new("STAR_Rig")
arm_obj  = bpy.data.objects.new("Armature", arm_data)
bpy.context.collection.objects.link(arm_obj)
bpy.context.view_layer.objects.active = arm_obj
arm_obj.select_set(True)

bpy.ops.object.mode_set(mode="EDIT")
eb = arm_data.edit_bones


def bone(name, head, tail, parent=None, connect=False):
    b = eb.new(name)
    b.head = head
    b.tail = tail
    if parent:
        b.parent = eb[parent]
        b.use_connect = connect
    return b


# ─── Gövde kemikleri (Z-up, T-pose) ─────────────────────────────────────────
#  Ölçek: yaklaşık 1.75 m boyunda insan, zemin = Z 0
bone("hips",         ( 0.00, 0, 0.95), ( 0.00, 0, 1.05))
bone("spine",        ( 0.00, 0, 1.05), ( 0.00, 0, 1.35), "hips",       True)
bone("chest",        ( 0.00, 0, 1.35), ( 0.00, 0, 1.50), "spine",      True)
bone("neck",         ( 0.00, 0, 1.50), ( 0.00, 0, 1.65), "chest",      True)

bone("upper_arm.L",  (-0.20, 0, 1.45), (-0.50, 0, 1.45), "chest",     False)
bone("forearm.L",    (-0.50, 0, 1.45), (-0.80, 0, 1.45), "upper_arm.L", True)

bone("upper_arm.R",  ( 0.20, 0, 1.45), ( 0.50, 0, 1.45), "chest",     False)
bone("forearm.R",    ( 0.50, 0, 1.45), ( 0.80, 0, 1.45), "upper_arm.R", True)

bone("thigh.L",      (-0.10, 0, 0.95), (-0.11, 0, 0.50), "hips",      False)
bone("shin.L",       (-0.11, 0, 0.50), (-0.11, 0, 0.08), "thigh.L",   True)
bone("foot.L",       (-0.11, 0, 0.08), (-0.11, 0.18, 0.00), "shin.L", True)

bone("thigh.R",      ( 0.10, 0, 0.95), ( 0.11, 0, 0.50), "hips",      False)
bone("shin.R",       ( 0.11, 0, 0.50), ( 0.11, 0, 0.08), "thigh.R",   True)
bone("foot.R",       ( 0.11, 0, 0.08), ( 0.11, 0.18, 0.00), "shin.R", True)

# ─── Sol el kemikleri ────────────────────────────────────────────────────────
# Palm: ek bir bağlantı kemiği (bilek → orta parmak tabanı)
bone("palm",      (-0.80, 0, 1.45), (-0.87, 0, 1.45), "forearm.L", True)

bone("thumb.1",   (-0.81, 0, 1.44), (-0.84, 0, 1.42), "palm",    False)
bone("thumb.2",   (-0.84, 0, 1.42), (-0.87, 0, 1.40), "thumb.1", True)
bone("thumb.3",   (-0.87, 0, 1.40), (-0.90, 0, 1.38), "thumb.2", True)

bone("index.1",   (-0.87, 0, 1.46), (-0.90, 0, 1.46), "palm",    False)
bone("index.2",   (-0.90, 0, 1.46), (-0.93, 0, 1.46), "index.1", True)
bone("index.3",   (-0.93, 0, 1.46), (-0.96, 0, 1.46), "index.2", True)

bone("middle.1",  (-0.87, 0, 1.45), (-0.90, 0, 1.45), "palm",     False)
bone("middle.2",  (-0.90, 0, 1.45), (-0.93, 0, 1.45), "middle.1", True)
bone("middle.3",  (-0.93, 0, 1.45), (-0.96, 0, 1.45), "middle.2", True)

bone("ring.1",    (-0.87, 0, 1.44), (-0.90, 0, 1.44), "palm",   False)
bone("ring.2",    (-0.90, 0, 1.44), (-0.93, 0, 1.44), "ring.1", True)
bone("ring.3",    (-0.93, 0, 1.44), (-0.96, 0, 1.44), "ring.2", True)

bone("pinky.1",   (-0.87, 0, 1.43), (-0.89, 0, 1.43), "palm",    False)
bone("pinky.2",   (-0.89, 0, 1.43), (-0.91, 0, 1.43), "pinky.1", True)
bone("pinky.3",   (-0.91, 0, 1.43), (-0.93, 0, 1.43), "pinky.2", True)

# ─── Roll hizalama: her kemiğin yerel +Z'si dünya +Z'sine baksın ────────────
# Bu adım olmadan to_track_quat('Y','Z') yanlış roll üretir.
up = mathutils.Vector((0, 0, 1))
for b in eb.values():
    b.align_roll(up)

# ─── Object moduna dön ───────────────────────────────────────────────────────
bpy.ops.object.mode_set(mode="OBJECT")

# Görünürlük: X-ray modu açık (kemikler üstte görünsün)
arm_obj.show_in_front = True

print("=" * 50)
print("STAR-1 Armature başarıyla oluşturuldu!")
print(f"  Obje adı  : {arm_obj.name}")
print(f"  Kemik sayısı: {len(arm_data.bones)}")
print("=" * 50)
