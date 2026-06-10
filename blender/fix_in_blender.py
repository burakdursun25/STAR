"""
Bu scripti Blender'in Scripting sekmesinde calistirin:
  1. Blender'i StarScene.blend ile acin
  2. Ust menuden "Scripting" sekmesine gecin
  3. "New" butonuna basin
  4. Bu kodu yapistirin
  5. "Run Script" butonuna basin (veya Alt+P)

Bu script:
  - Eksik addon kayitlarini kaldirir
  - star_live_addon'u etkinlestirir
  - Armature ayarlarini yapar
"""
import bpy
import os

print("\n=== STAR-1 FIX SCRIPT BASLIYOR ===\n")

# 1. Eksik addon'lari kaldir
stale_addons = ["blender_live_addon", "blender_live_pose_addon", "blender_addon_oop"]
for name in stale_addons:
    addon = bpy.context.preferences.addons.get(name)
    if addon:
        bpy.context.preferences.addons.remove(addon)
        print(f"[OK] Kaldirildi: {name}")
    else:
        print(f"[--] Zaten yok: {name}")

# 2. star_live_addon etkin mi kontrol et
addon_name = "star_live_addon"
if addon_name in bpy.context.preferences.addons:
    print(f"[OK] {addon_name} zaten etkin")
else:
    print(f"[..] {addon_name} etkinlestiriliyor...")
    try:
        bpy.ops.preferences.addon_enable(module=addon_name)
        print(f"[OK] {addon_name} etkinlestirildi")
    except Exception as e:
        print(f"[!!] Hata: {e}")
        # Dosyadan yukle
        addon_paths = [
            r"C:\Users\burak\AppData\Roaming\Blender Foundation\Blender\5.0\scripts\addons\star_live_addon.py",
            r"c:\Users\burak\Desktop\STAR-1\blender\star_live_addon.py",
        ]
        for path in addon_paths:
            if os.path.exists(path):
                try:
                    bpy.ops.preferences.addon_install(filepath=path, overwrite=True)
                    bpy.ops.preferences.addon_enable(module=addon_name)
                    print(f"[OK] Dosyadan yuklendi: {path}")
                    break
                except Exception as e2:
                    print(f"[!!] {path}: {e2}")

# 3. Sahnedeki armature'u bul
armatures = [obj for obj in bpy.data.objects if obj.type == "ARMATURE"]
if armatures:
    arm = armatures[0]
    print(f"\n[OK] Armature bulundu: '{arm.name}' ({len(arm.data.bones)} kemik)")
    print(f"     Kemikler: {[b.name for b in arm.data.bones]}")
else:
    print("\n[!!] Sahnede ARMATURE bulunamadi!")
    print("     create_armature.py'yi calistirin.")

# 4. Addon props ayarla
try:
    props = bpy.context.scene.star_live_props
    if armatures:
        props.armature_name = armatures[0].name
    props.port = 7777
    print(f"\n[OK] Props: armature='{props.armature_name}', port={props.port}")
except Exception as e:
    print(f"\n[!!] Props hatasi: {e}")
    print("     star_live_addon etkin degil olabilir - Preferences'tan etkinlestirin")

# 5. Kaydet
try:
    bpy.ops.wm.save_userpref()
    bpy.ops.wm.save_mainfile()
    print("\n[OK] Kaydedildi!")
except Exception as e:
    print(f"\n[?] Kayit bilgisi: {e}")

print("\n=== BITTI ===")
print("Simdi N paneli > STAR Live > 'Dinlemeyi Baslat' tusuna basin")
