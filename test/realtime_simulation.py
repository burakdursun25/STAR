import bpy
import random
import math
import sys
import os

# Script'in bulunduğu klasörü doğrudan sisteme ekliyoruz
script_dir = r"d:\StarProject\tools\test"
if script_dir not in sys.path:
    sys.path.insert(0, script_dir) # En başa ekliyoruz ki kesinlikle bulsun

# Modülü içe aktar, Blender önbelleğinde kalmışsa diye reload (yenile) işlemi yap
import importlib
import animate_location
importlib.reload(animate_location)
from animate_location import animate_location

class RealtimeSimulator:
    def __init__(self, obj, max_frames=250):
        self.obj = obj
        self.max_frames = max_frames
        
        # Simülasyonun başladığı anki kareyi kaydet
        self.start_frame = bpy.context.scene.frame_current
        
        # Animasyon verilerini temizle ve bulunduğumuz kareye sıfır noktasını ekle
        self.obj.animation_data_clear()
        animate_location(self.obj, (0, 0, 0), self.start_frame)
        # Daha yumuşak ve doğal bir hareket için açı değişkeni başlat
        self.angle = 0.0
        
    def step(self):
        # Timeline'daki o anki canlı kareyi al
        current_frame = bpy.context.scene.frame_current
        
        # Obje yoksa veya belirlediğimiz maksimum süreyi geçtiysek durdur
        if not self.obj or current_frame >= self.start_frame + self.max_frames:
            # Çalan animasyonu durdur
            if bpy.context.screen.is_animation_playing:
                bpy.ops.screen.animation_cancel(restore_frame=False)
            print("Simülasyon tamamlandı.")
            return None # Timer'ı bitir
            
        # --- KAVİSLİ VE ORGANİK HAREKET (Gerçek Zamanlı) ---
        # Açıyı hafifçe değiştir, böylece obje kavis çizerek (yumuşak dönerek) ilerler
        self.angle += random.uniform(-0.3, 0.3)
        speed = random.uniform(0.1, 0.4) # Hız
        
        dx = math.cos(self.angle) * speed
        dy = math.sin(self.angle) * speed
        dz = random.uniform(-0.1, 0.1) # Hafif dikey sekme
        
        # Mevcut konumun üzerine ekle
        new_x = self.obj.location[0] + dx
        new_y = self.obj.location[1] + dy
        new_z = max(0.0, self.obj.location[2] + dz)
        
        # O anki canlı kareye yeni kavisli değerleri keyframe olarak yaz
        animate_location(self.obj, (new_x, new_y, new_z), current_frame)
        
        # Sahnede ayarlı olan FPS değerine göre bir sonraki timer adımı için bekle
        fps = bpy.context.scene.render.fps / bpy.context.scene.render.fps_base
        return 1.0 / fps 

if __name__ == "__main__":
    active_obj = bpy.context.active_object
    if active_obj:
        print("Gerçek zamanlı rastgele animasyon simülasyonu başlatılıyor...")
        
        # Timeline oynatımını başlat (gerçek zamanlı izleyebilmek için)
        if not bpy.context.screen.is_animation_playing:
            bpy.ops.screen.animation_play()
            
        simulator = RealtimeSimulator(active_obj, max_frames=250)
        bpy.app.timers.register(simulator.step)
    else:
        print("Lütfen önce bir obje seçin!")
