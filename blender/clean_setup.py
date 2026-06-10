"""
STAR-1 Temizleme + Kurulum Scripti
====================================
- Eksik (Missing) eski addon kayıtlarını siler
- star_live_addon'u doğru şekilde kurar ve etkinleştirir
- Sahneyi kaydeder
"""
import bpy
import sys
import os

THIS_DIR   = os.path.dirname(os.path.abspath(__file__))
ADDON_FILE = os.path.join(THIS_DIR, "star_live_addon.py")
ARM_FILE   = os.path.join(THIS_DIR, "create_armature.py")

print("\n" + "=" * 60)
print("  STAR-1 Temizleme + Kurulum")
print("=" * 60)

# ── 0) Eski/eksik addon'ları kaldır ──────────────────────────────────────────
print("\n[0/4] Eski addon kayitlari temizleniyor...")
stale = ["blender_live_addon", "blender_live_pose_addon", "blender_addon_oop"]
for mod_name in stale:
    try:
        if mod_name in bpy.context.preferences.addons:
            bpy.ops.preferences.addon_disable(module=mod_name)
            print(f"      - {mod_name} devre disi birakildi")
        # Preferences'tan da sil
        addon = bpy.context.preferences.addons.get(mod_name)
        if addon:
            bpy.context.preferences.addons.remove(addon)
            print(f"      - {mod_name} kaldirildi")
    except Exception as e:
        print(f"      (bilgi) {mod_name}: {e}")

# ── 1) star_live_addon kur ve etkinleştir ─────────────────────────────────────
print(f"\n[1/4] Addon kuruluyor: {ADDON_FILE}")
try:
    bpy.ops.preferences.addon_install(filepath=ADDON_FILE, overwrite=True)
    bpy.ops.preferences.addon_enable(module="star_live_addon")
    print("      OK - Addon kuruldu ve etkinlestirildi.")
except Exception as e:
    print(f"      HATA: {e}")
    sys.exit(1)

# ── 2) Armature kontrolü ──────────────────────────────────────────────────────
print("\n[2/4] Armature kontrol ediliyor...")
arm_obj = bpy.data.objects.get("Armature")
if arm_obj and arm_obj.type == "ARMATURE":
    print(f"      OK - '{arm_obj.name}' mevcut ({len(arm_obj.data.bones)} kemik).")
else:
    print("      ! 'Armature' bulunamadi - create_armature.py calistiriliyor...")
    try:
        exec(open(ARM_FILE, encoding="utf-8").read())
        print("      OK - Armature olusturuldu.")
    except Exception as e:
        print(f"      HATA: {e}")

# ── 3) Addon ayarları ─────────────────────────────────────────────────────────
print("\n[3/4] Addon ayarlari yapilandiriliyor...")
try:
    props = bpy.context.scene.star_live_props
    props.armature_name = "Armature"
    props.port = 7777
    print("      OK - armature_name = 'Armature', port = 7777")
except Exception as e:
    print(f"      HATA: {e}")

# ── 4) Kaydet ─────────────────────────────────────────────────────────────────
print("\n[4/4] Kaydediliyor...")
try:
    bpy.ops.wm.save_userpref()
    bpy.ops.wm.save_mainfile()
    print("      OK - StarScene.blend ve kullanici tercihleri kaydedildi.")
except Exception as e:
    print(f"      (bilgi) Kayit: {e}")

print("\n" + "=" * 60)
print("  STAR-1 Kurulum Tamamlandi!")
print("  Blender acikken: N-Panel > STAR Live > Dinlemeyi Baslat")
print("=" * 60 + "\n")
