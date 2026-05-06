"""
Blender'da iskelet (armature) oluştur ve addon'u configure et
"""
import bpy
import os

def create_basic_armature():
    """Basit bir insan iskeletini oluştur"""
    
    # Yeni armature oluştur
    armature_data = bpy.data.armatures.new("HumanArmature")
    armature_obj = bpy.data.objects.new("Armature", armature_data)
    bpy.context.collection.objects.link(armature_obj)
    bpy.context.view_layer.objects.active = armature_obj
    armature_obj.select_set(True)
    
    # Edit mode'a gir
    bpy.ops.object.mode_set(mode='EDIT')
    
    # Bones ekle
    bones_to_create = [
        # Main
        ("Hips", None),
        
        # Left leg
        ("LeftUpLeg", "Hips"),
        ("LeftLeg", "LeftUpLeg"),
        ("LeftFoot", "LeftLeg"),
        
        # Right leg
        ("RightUpLeg", "Hips"),
        ("RightLeg", "RightUpLeg"),
        ("RightFoot", "RightLeg"),
        
        # Spine
        ("Spine", "Hips"),
        ("Chest", "Spine"),
        ("Neck", "Chest"),
        ("Head", "Neck"),
        
        # Left arm
        ("LeftArm", "Chest"),
        ("LeftForeArm", "LeftArm"),
        ("LeftHand", "LeftForeArm"),
        
        # Right arm
        ("RightArm", "Chest"),
        ("RightForeArm", "RightArm"),
        ("RightHand", "RightForeArm"),
    ]
    
    bone_positions = {
        "Hips": (0, 1, 0),
        "Spine": (0, 1.2, 0),
        "Chest": (0, 1.4, 0),
        "Neck": (0, 1.6, 0),
        "Head": (0, 1.8, 0),
        
        "LeftUpLeg": (-0.1, 0.8, 0),
        "LeftLeg": (-0.1, 0.4, 0),
        "LeftFoot": (-0.1, 0, 0),
        
        "RightUpLeg": (0.1, 0.8, 0),
        "RightLeg": (0.1, 0.4, 0),
        "RightFoot": (0.1, 0, 0),
        
        "LeftArm": (-0.4, 1.3, 0),
        "LeftForeArm": (-0.6, 1.0, 0),
        "LeftHand": (-0.7, 0.8, 0),
        
        "RightArm": (0.4, 1.3, 0),
        "RightForeArm": (0.6, 1.0, 0),
        "RightHand": (0.7, 0.8, 0),
    }
    
    created_bones = {}
    
    for bone_name, parent_name in bones_to_create:
        bone = armature_data.edit_bones.new(bone_name)
        pos = bone_positions.get(bone_name, (0, 1, 0))
        bone.head = pos
        bone.tail = (pos[0], pos[1] - 0.2, pos[2])
        
        if parent_name and parent_name in created_bones:
            bone.parent = created_bones[parent_name]
        
        created_bones[bone_name] = bone
    
    # Object mode'a dön
    bpy.ops.object.mode_set(mode='OBJECT')
    
    return armature_obj


def setup_addon():
    """Addon'u ekle ve configure et"""
    addon_path = r"c:\Users\burak\Desktop\Görüntüİşleme\blender_live_pose_addon.py"
    
    try:
        # Addon dosyasından module'ü import et
        import importlib.util
        spec = importlib.util.spec_from_file_location("live_pose_addon", addon_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # Addon'ı register et
        if hasattr(module, 'register'):
            module.register()
            print(f"✓ Addon başarıyla yüklendi ve register edildi")
            return True
        
    except Exception as e:
        print(f"✗ Addon yükleme hatası: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Ana setup fonksiyonu"""
    print("=" * 50)
    print("Blender Live Pose Capture Setup")
    print("=" * 50)
    
    # Mevcut tüm objeleri temizle
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()
    
    # Armature oluştur
    print("\n[1] Armature oluşturuluyor...")
    armature = create_basic_armature()
    print(f"✓ Armature oluşturuldu: {armature.name}")
    
    # Addon'u yükle (property registration'u yapacak)
    print("\n[2] Addon yükleniyor...")
    if not setup_addon():
        print("✗ Addon yükleme başarısız, script iptal ediliyor")
        return
    
    # Addon load edildikten sonra, scene'de property olur
    # Armature'ı scene'e ata
    try:
        bpy.context.scene.posecapture_armature = armature
        print(f"✓ Armature scene'e atandı")
    except Exception as e:
        print(f"✗ Armature ataması hatası: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n" + "=" * 50)
    print("Setup tamamlandı!")
    print("=" * 50)
    print("\nSonraki adımlar:")
    print("1. 3D View'de 'Live Pose Capture' panelini aç")
    print("2. 'BAŞLAT' butonuna tıkla")
    print("3. Terminalde Python script'ini çalıştır:")
    print("   python image_processor_oop.py")



if __name__ == "__main__":
    main()
