"""
STAR-1 Blender Otomatik Kurulum Scripti
========================================
Blender komut satırından çağrılır:
  blender StarScene.blend --python blender/auto_setup.py

Ne yapar:
  1. star_live_addon.py'yi Blender'a addon olarak kurar ve etkinleştirir.
  2. Sahnede 'Armature' yoksa create_armature.py çalıştırır.
  3. addon portunu 7777 olarak ayarlar.
"""
import bpy
import sys
import os

# ── Dosya yollarını belirle ───────────────────────────────────────────────────
THIS_DIR   = os.path.dirname(os.path.abspath(__file__))
ADDON_FILE = os.path.join(THIS_DIR, "star_live_addon.py")
ARM_FILE   = os.path.join(THIS_DIR, "create_armature.py")

print("\n" + "=" * 60)
print("  STAR-1 Otomatik Kurulum Başladı")
print("=" * 60)

# ── 1) Addon'u kur ve etkinleştir ─────────────────────────────────────────────
print(f"\n[1/3] Addon kuruluyor: {ADDON_FILE}")
try:
    bpy.ops.preferences.addon_install(filepath=ADDON_FILE, overwrite=True)
    bpy.ops.preferences.addon_enable(module="star_live_addon")
    bpy.ops.wm.save_userpref()
    print("      ✓ Addon kuruldu ve etkinleştirildi.")
except Exception as e:
    print(f"      ✗ Addon kurulum hatası: {e}")
    sys.exit(1)

# ── 2) Armature kontrolü ──────────────────────────────────────────────────────
print("\n[2/3] Armature kontrol ediliyor...")
arm_obj = bpy.data.objects.get("Armature")
if arm_obj and arm_obj.type == "ARMATURE":
    print(f"      ✓ '{arm_obj.name}' mevcut ({len(arm_obj.data.bones)} kemik).")
else:
    print("      ! 'Armature' bulunamadı — create_armature.py çalıştırılıyor...")
    try:
        exec(open(ARM_FILE, encoding="utf-8").read())
        print("      ✓ Armature oluşturuldu.")
    except Exception as e:
        print(f"      ✗ Armature oluşturma hatası: {e}")

# ── 3) Addon ayarlarını yapılandır ────────────────────────────────────────────
print("\n[3/3] Addon ayarları yapılandırılıyor...")
try:
    props = bpy.context.scene.star_live_props
    props.armature_name = "Armature"
    props.port = 7777
    print("      ✓ armature_name = 'Armature'")
    print("      ✓ port          = 7777")
except Exception as e:
    print(f"      ✗ Ayar hatası: {e}")

# ── Sahneyi kaydet ────────────────────────────────────────────────────────────
try:
    bpy.ops.wm.save_mainfile()
    print("\n      ✓ StarScene.blend kaydedildi.")
except Exception as e:
    print(f"\n      ! Kayıt hatası (normal olabilir): {e}")

print("\n" + "=" * 60)
print("  STAR-1 Kurulum Tamamlandı!")
print("  Blender'ı açtıktan sonra N-Panel → STAR Live → Dinlemeyi Başlat")
print("=" * 60 + "\n")
