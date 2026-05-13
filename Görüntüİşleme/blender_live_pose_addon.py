bl_info = {
    "name": "STAR Live Pose Capture",
    "blender": (3, 0, 0),
    "category": "Animation",
    "version": (2, 4, 0),
    "author": "STAR Project",
    "description": "Webcam pose verisiyle Rigify iskeletini canli hareket ettirir",
}

import bpy, json, socket, threading, time, math
from mathutils import Euler, Vector, Matrix

# ── Global state ──────────────────────────────────────────────────────────────
g_udp_socket    = None
g_thread        = None
g_running       = False
g_last_data     = None
g_lock          = threading.Lock()
g_fps_counter   = 0
g_fps_display   = 0.0
g_fps_time      = 0.0
g_updated_bones = 0

# ── Bone groups ───────────────────────────────────────────────────────────────
BONE_GROUPS = [
    ("Govde",     ["spine","spine.001","spine.002","spine.003","spine.006","spine.007"]),
    ("Sol Kol",   ["shoulder.L","upper_arm.L","forearm.L","hand.L"]),
    ("Sag Kol",   ["shoulder.R","upper_arm.R","forearm.R","hand.R"]),
    ("Sol Bacak", ["thigh.L","shin.L","foot.L","toe.L"]),
    ("Sag Bacak", ["thigh.R","shin.R","foot.R","toe.R"]),
]
ALL_SOURCE_BONES = [b for _, bones in BONE_GROUPS for b in bones]

RIGIFY_METARIG_MAP   = {b: b for b in ALL_SOURCE_BONES}
RIGIFY_GENERATED_MAP = {
    "spine":"torso","spine.001":"spine_fk","spine.002":"spine_fk.001",
    "spine.003":"chest","spine.006":"neck","spine.007":"head",
    "shoulder.L":"shoulder.L","shoulder.R":"shoulder.R",
    "upper_arm.L":"upper_arm_fk.L","forearm.L":"forearm_fk.L","hand.L":"hand_fk.L",
    "upper_arm.R":"upper_arm_fk.R","forearm.R":"forearm_fk.R","hand.R":"hand_fk.R",
    "thigh.L":"thigh_fk.L","shin.L":"shin_fk.L","foot.L":"foot_ik.L","toe.L":"toe_ik.L",
    "thigh.R":"thigh_fk.R","shin.R":"shin_fk.R","foot.R":"foot_ik.R","toe.R":"toe_ik.R",
}

def _detect_rigify_type(arm):
    names = {b.name for b in arm.data.bones}
    if "torso" in names or "DEF-spine" in names: return "generated"
    if "spine" in names and "upper_arm.L" in names: return "metarig"
    return "unknown"

# ── Direction vector helper ───────────────────────────────────────────────────
def _dir(s, e):
    """Normalize yon vektoru. Hata durumunda +Y (rest yonu) donder."""
    d = e - s
    if d.length < 1e-6: return [0.0, 1.0, 0.0]
    d = d.normalized()
    return [d.x, d.y, d.z]

# ── Skeleton from landmarks (addon side, fallback) ────────────────────────────
def compute_skeleton_from_landmarks(lm):
    """pose_landmarks JSON listesinden Rigify skeleton hesapla.
    rotation = normalize yon vektoru [dx,dy,dz], Euler degil.
    """
    if len(lm) < 29: return {}
    # -x: kamera aynalamasi duzeltmesi
    def pt(i): return Vector((-lm[i]["x"], -lm[i]["y"], -lm[i]["z"]))
    def vis(i): return float(lm[i].get("visibility", 1.0) or 1.0)

    l_hip=pt(23); r_hip=pt(24); l_sh=pt(11); r_sh=pt(12)
    l_el=pt(13);  r_el=pt(14);  l_wr=pt(15); r_wr=pt(16)
    l_kn=pt(25);  r_kn=pt(26);  l_an=pt(27); r_an=pt(28)
    l_ear=pt(7);  r_ear=pt(8);  nose=pt(0)

    hip_c  = (l_hip+r_hip)/2; sh_c=(l_sh+r_sh)/2
    ear_c  = (l_ear+r_ear)/2; head_top=nose+(nose-ear_c)*0.35
    hf=len(lm)>=33
    l_ft=pt(31) if hf else l_an+Vector((0,.05,.1))
    r_ft=pt(32) if hf else r_an+Vector((0,.05,.1))

    sk={}
    def add(n,pos,s,e,c=1.0): sk[n]={"position":[pos.x,pos.y,pos.z],"rotation":_dir(s,e),"confidence":c}

    sv=sh_c-hip_c
    sk["spine"]={"position":[hip_c.x,hip_c.y,hip_c.z],"rotation":_dir(hip_c,hip_c+sv),"confidence":1.0}
    for n,t in [("spine.001",.25),("spine.002",.5),("spine.003",.75)]:
        p=hip_c+sv*t; sk[n]={"position":[p.x,p.y,p.z],"rotation":[0.,1.,0.],"confidence":1.0}

    add("spine.006",sh_c,sh_c,ear_c); add("spine.007",ear_c,ear_c,head_top)
    add("shoulder.L",sh_c,sh_c,l_sh); add("shoulder.R",sh_c,sh_c,r_sh)
    add("upper_arm.L",l_sh,l_sh,l_el,vis(11)); add("forearm.L",l_el,l_el,l_wr,vis(13))
    add("hand.L",l_wr,l_wr,l_wr+(l_wr-l_el)*.3)
    add("upper_arm.R",r_sh,r_sh,r_el,vis(12)); add("forearm.R",r_el,r_el,r_wr,vis(14))
    add("hand.R",r_wr,r_wr,r_wr+(r_wr-r_el)*.3)
    add("thigh.L",l_hip,l_hip,l_kn,vis(23)); add("shin.L",l_kn,l_kn,l_an,vis(25))
    add("foot.L",l_an,l_an,l_ft,vis(27)); add("toe.L",l_ft,l_ft,l_ft+(l_ft-l_an)*.3)
    add("thigh.R",r_hip,r_hip,r_kn,vis(24)); add("shin.R",r_kn,r_kn,r_an,vis(26))
    add("foot.R",r_an,r_an,r_ft,vis(28)); add("toe.R",r_ft,r_ft,r_ft+(r_ft-r_an)*.3)
    return sk

# ── Bone Smoother ─────────────────────────────────────────────────────────────
class BoneSmoother:
    """EMA + confidence esigi + max-delta klamp"""
    def __init__(self): self._s = {}
    def reset(self): self._s.clear()
    def smooth(self, name, rot, conf, alpha, max_delta):
        prev = self._s.get(name)
        if prev is None: self._s[name]=list(rot); return rot
        cl=[]
        for p,r in zip(prev,rot):
            d=r-p
            if abs(d)>max_delta: r=p+math.copysign(max_delta,d)
            cl.append(r)
        sm=[prev[i]+alpha*(cl[i]-prev[i]) for i in range(len(rot))]
        self._s[name]=sm; return sm

g_smoother=BoneSmoother()

# ── UDP receiver ──────────────────────────────────────────────────────────────
class UDPReceiver:
    def __init__(self,host,port):
        self.sock=socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
        self.sock.bind((host,port)); self.sock.settimeout(0.5)
        print(f"[STAR] UDP dinleniyor -> {host}:{port}")
    def receive(self):
        try: d,_=self.sock.recvfrom(65536); return json.loads(d.decode())
        except socket.timeout: return None
        except Exception as e: print(f"[STAR UDP] {e}"); return None
    def close(self):
        try: self.sock.close()
        except: pass

def _receiver_loop():
    global g_last_data,g_fps_counter,g_fps_display,g_fps_time
    g_fps_time=time.time()
    while g_running:
        data=g_udp_socket.receive()
        if data:
            if "skeleton" not in data and "pose_landmarks" in data:
                data["skeleton"]=compute_skeleton_from_landmarks(data["pose_landmarks"])
            with g_lock:
                g_last_data=data; g_fps_counter+=1
                now=time.time()
                if now-g_fps_time>=1.0:
                    g_fps_display=g_fps_counter/(now-g_fps_time)
                    g_fps_counter=0; g_fps_time=now

# ── Apply skeleton ────────────────────────────────────────────────────────────
def apply_skeleton(armature_obj, skeleton_data, bone_mappings, scene):
    """rotation field = normalize yon vektoru [dx,dy,dz].
    
    Her kemik icin:
      1. Kemigin REST-POSE world-space +Y yonu alinir  (bone_y_rest)
      2. skeleton 'rotation' vektoru target_dir olarak alinir
      3. bone_y_rest -> target_dir arasi DELTA QUATERNION hesaplanir
      4. Parent world matrisinin inversiyle LOCAL rotasyona cevrilir

    [0,1,0] gelirse = rest yonu = no rotation = kemik oynamaz. DOGRU.
    Spine yukariya bakiyorsa bone_y=(0,1,0), target=(0,1,0) -> identity. DOGRU.
    """
    if not armature_obj or armature_obj.type!='ARMATURE': return 0

    pose       = armature_obj.pose
    arm_mat    = armature_obj.matrix_world
    arm_rot3   = arm_mat.to_3x3()
    alpha      = scene.star_smooth_alpha
    conf_thresh= scene.star_conf_threshold
    max_delta  = scene.star_max_delta
    count=0

    for m in bone_mappings:
        if not m.enabled: continue
        target=m.target_bone
        if not target or target not in pose.bones: continue
        bd=skeleton_data.get(m.source_name)
        if not bd: continue

        rot_raw = bd.get("rotation",[0.,1.,0.])  # [dx,dy,dz] normalize yon
        conf    = float(bd.get("confidence",1.0))
        if conf<conf_thresh: continue

        # EMA filtresi (3 elemanli yon vektoru)
        sm = g_smoother.smooth(m.source_name, rot_raw, conf, alpha, max_delta)
        target_dir = Vector(sm).normalized()

        try:
            pb   = pose.bones[target]
            bone = armature_obj.data.bones[target]

            # Kemigin rest-pose world-space +Y yonu
            bone_y = (arm_rot3 @ bone.matrix_local.to_3x3() @ Vector((0,1,0))).normalized()

            # bone_y -> target_dir arasi delta
            dq = bone_y.rotation_difference(target_dir)

            # Local space'e cevir (parent world matrisinin inversi)
            if pb.parent:
                pw3 = (arm_mat @ pb.parent.matrix).to_3x3()
            else:
                pw3 = arm_rot3
            lq = (pw3.inverted() @ dq.to_matrix()).to_quaternion()
            lq.normalize()

            pb.rotation_mode       = 'QUATERNION'
            pb.rotation_quaternion = lq
            count+=1
        except Exception as e:
            try:
                pb.rotation_mode = 'XYZ'
                pb.rotation_euler= Euler((0,0,0),'XYZ')
            except: pass
            print(f"[STAR Kemik] {target}: {e}")

    return count

# ── Modal operator ────────────────────────────────────────────────────────────
class STAR_OT_live_updater(bpy.types.Operator):
    bl_idname="star.live_updater"; bl_label="STAR Live Pose Driver"
    _timer=None
    def modal(self,context,event):
        global g_last_data,g_updated_bones
        if not g_running: self.cancel(context); return {'FINISHED'}
        if event.type=='TIMER':
            with g_lock: data=g_last_data
            if data:
                scene=context.scene; arm=scene.star_armature
                sk=data.get("skeleton",{})
                if arm and sk:
                    g_updated_bones=apply_skeleton(arm,sk,scene.star_bone_mappings,scene)
                    try: context.view_layer.update()
                    except: pass
            for area in context.screen.areas:
                if area.type=='VIEW_3D': area.tag_redraw()
        return {'PASS_THROUGH'}
    def execute(self,context):
        wm=context.window_manager
        self._timer=wm.event_timer_add(0.016,window=context.window)
        wm.modal_handler_add(self)
        return {'RUNNING_MODAL'}
    def cancel(self,context):
        wm=context.window_manager
        if self._timer: wm.event_timer_remove(self._timer); self._timer=None

# ── Property group ────────────────────────────────────────────────────────────
class STAR_BoneMapping(bpy.types.PropertyGroup):
    source_name: bpy.props.StringProperty(default="")
    target_bone: bpy.props.StringProperty(default="")
    enabled:     bpy.props.BoolProperty(default=True)

# ── Operators ─────────────────────────────────────────────────────────────────
class STAR_OT_start(bpy.types.Operator):
    bl_idname="star.start"; bl_label="BASLAT"
    def execute(self,context):
        global g_udp_socket,g_thread,g_running
        if g_running: self.report({'WARNING'},"Zaten calisiyor!"); return {'FINISHED'}
        try:
            sc=context.scene; g_udp_socket=UDPReceiver(sc.star_host,sc.star_port)
            g_running=True; g_smoother.reset()
            g_thread=threading.Thread(target=_receiver_loop,daemon=True); g_thread.start()
            bpy.ops.star.live_updater('INVOKE_DEFAULT')
            self.report({'INFO'},f"Basladi -> {sc.star_host}:{sc.star_port}")
        except Exception as e: g_running=False; self.report({'ERROR'},f"Hata: {e}")
        return {'FINISHED'}

class STAR_OT_stop(bpy.types.Operator):
    bl_idname="star.stop"; bl_label="DURDUR"
    def execute(self,context):
        global g_udp_socket,g_thread,g_running,g_last_data
        g_running=False
        if g_thread: g_thread.join(timeout=1.)
        if g_udp_socket: g_udp_socket.close()
        g_udp_socket=g_thread=g_last_data=None; g_smoother.reset()
        self.report({'INFO'},"Durduruldu."); return {'FINISHED'}

class STAR_OT_reset_mappings(bpy.types.Operator):
    bl_idname="star.reset_mappings"; bl_label="Sifirla"
    def execute(self,context):
        sc=context.scene; sc.star_bone_mappings.clear()
        for s in ALL_SOURCE_BONES:
            it=sc.star_bone_mappings.add(); it.source_name=s; it.target_bone=s; it.enabled=True
        self.report({'INFO'},"Sifirlanadi."); return {'FINISHED'}

class STAR_OT_fill_from_armature(bpy.types.Operator):
    bl_idname="star.fill_from_armature"; bl_label="Isme Gore Doldur"
    def execute(self,context):
        sc=context.scene; arm=sc.star_armature
        if not arm or arm.type!='ARMATURE': self.report({'WARNING'},"Armature secin!"); return {'FINISHED'}
        ab={b.name.lower():b.name for b in arm.data.bones}; n=0
        for m in sc.star_bone_mappings:
            c=ab.get(m.source_name.lower())
            if c: m.target_bone=c; n+=1
        self.report({'INFO'},f"{n} kemik eslesti."); return {'FINISHED'}

class STAR_OT_detect_rigify(bpy.types.Operator):
    bl_idname="star.detect_rigify"; bl_label="Rigify Otomatik Tani"
    def execute(self,context):
        sc=context.scene; arm=sc.star_armature
        if not arm or arm.type!='ARMATURE': self.report({'WARNING'},"Armature secin!"); return {'FINISHED'}
        rt=_detect_rigify_type(arm)
        if rt=="generated": mt=RIGIFY_GENERATED_MAP; lbl="Generated Rig"
        elif rt=="metarig": mt=RIGIFY_METARIG_MAP; lbl="Metarig"
        else:
            ab={b.name for b in arm.data.bones}
            gs=sum(1 for v in RIGIFY_GENERATED_MAP.values() if v in ab)
            ms=sum(1 for v in RIGIFY_METARIG_MAP.values() if v in ab)
            if gs>=ms and gs>0: mt=RIGIFY_GENERATED_MAP; lbl="Generated Rig (tahmini)"
            elif ms>0: mt=RIGIFY_METARIG_MAP; lbl="Metarig (tahmini)"
            else: self.report({'WARNING'},"Rigify taninamadi!"); return {'FINISHED'}
        ab={b.name for b in arm.data.bones}; ok=0; miss=[]
        for m in sc.star_bone_mappings:
            tgt=mt.get(m.source_name)
            if tgt and tgt in ab: m.target_bone=tgt; m.enabled=True; ok+=1
            else: miss.append(m.source_name); m.enabled=False
        msg=f"{lbl} -> {ok} kemik eslesti."
        if miss: msg+=f" ({len(miss)} bulunamadi: {', '.join(miss[:3])})"
        self.report({'INFO'},msg); return {'FINISHED'}

class STAR_OT_toggle_group(bpy.types.Operator):
    bl_idname="star.toggle_group"; bl_label="Grubu Ac/Kapat"
    group_name: bpy.props.StringProperty()
    def execute(self,context):
        sc=context.scene; gb=next((b for n,b in BONE_GROUPS if n==self.group_name),[])
        md={m.source_name:m for m in sc.star_bone_mappings}
        items=[md[b] for b in gb if b in md]
        if not items: return {'FINISHED'}
        ao=all(m.enabled for m in items)
        for m in items: m.enabled=not ao
        return {'FINISHED'}

# ── Panels ────────────────────────────────────────────────────────────────────
class STAR_PT_main(bpy.types.Panel):
    bl_label="STAR Live Pose v2.4"; bl_idname="STAR_PT_main"
    bl_space_type='VIEW_3D'; bl_region_type='UI'; bl_category="STAR Live"
    def draw(self,context):
        layout=self.layout; sc=context.scene
        box=layout.box(); box.label(text="Baglanti",icon='PLUGIN')
        col=box.column(align=True)
        col.prop(sc,"star_host",text="Host"); col.prop(sc,"star_port",text="Port")
        box.separator(factor=.3); row=box.row(); row.scale_y=1.8
        if g_running: row.operator("star.stop",icon='PAUSE',text="  DURDUR")
        else:         row.operator("star.start",icon='PLAY', text="  BASLAT")
        box2=layout.box(); sr=box2.row()
        if g_running: sr.label(text="CANLI",icon='RADIOBUT_ON')
        else:         sr.label(text="DURMUS",icon='RADIOBUT_OFF')
        if g_last_data:
            c2=box2.column(align=True)
            c2.label(text=f"FPS:         {g_fps_display:.1f}")
            c2.label(text=f"Guncellenen: {g_updated_bones} kemik")
            c2.label(text=f"Gelen kemik: {len(g_last_data.get('skeleton',{}))}")
        else: box2.label(text="Veri bekleniyor...",icon='TIME')
        box3=layout.box(); box3.label(text="Hareket Filtresi",icon='MOD_SMOOTH')
        col3=box3.column(align=True)
        col3.prop(sc,"star_smooth_alpha",   text="Yumusatma")
        col3.prop(sc,"star_conf_threshold", text="Min. Gorunurluk")
        col3.prop(sc,"star_max_delta",      text="Max Sicrama")

class STAR_PT_armature(bpy.types.Panel):
    bl_label="Armature"; bl_idname="STAR_PT_armature"
    bl_space_type='VIEW_3D'; bl_region_type='UI'; bl_category="STAR Live"
    bl_parent_id="STAR_PT_main"
    def draw(self,context):
        layout=self.layout; sc=context.scene
        layout.prop(sc,"star_armature",text="")
        row=layout.row(); row.scale_y=1.4
        row.operator("star.detect_rigify",icon='ARMATURE_DATA',text="Rigify Otomatik Tani")
        row2=layout.row(align=True)
        row2.operator("star.fill_from_armature",icon='BONE_DATA',text="Isme Gore Doldur")
        row2.operator("star.reset_mappings",icon='LOOP_BACK',text="Sifirla")

class STAR_PT_bones(bpy.types.Panel):
    bl_label="Kemik Eslestirme"; bl_idname="STAR_PT_bones"
    bl_space_type='VIEW_3D'; bl_region_type='UI'; bl_category="STAR Live"
    bl_parent_id="STAR_PT_main"; bl_options={'DEFAULT_CLOSED'}
    def draw(self,context):
        layout=self.layout; sc=context.scene; mappings=sc.star_bone_mappings
        if not mappings:
            layout.label(text="Eslestirme yok.",icon='ERROR')
            layout.operator("star.reset_mappings",icon='LOOP_BACK'); return
        md={m.source_name:m for m in mappings}
        for gn,bl in BONE_GROUPS:
            col=layout.column(align=True); hdr=col.row(align=True)
            hdr.label(text=gn,icon='GROUP_BONE')
            op=hdr.operator("star.toggle_group",text="",icon='CHECKBOX_HLT'); op.group_name=gn
            box=col.box()
            for sb in bl:
                m=md.get(sb)
                if not m: continue
                row=box.row(align=True); row.prop(m,"enabled",text="")
                s=row.row(); s.enabled=m.enabled; s.label(text=sb)
                s2=row.row(); s2.enabled=m.enabled; s2.label(text="->")
                s3=row.row(); s3.enabled=m.enabled; s3.scale_x=1.3; s3.prop(m,"target_bone",text="")
            col.separator(factor=.5)

# ── Register ──────────────────────────────────────────────────────────────────
CLASSES=[
    STAR_BoneMapping,STAR_OT_live_updater,STAR_OT_start,STAR_OT_stop,
    STAR_OT_reset_mappings,STAR_OT_fill_from_armature,STAR_OT_detect_rigify,
    STAR_OT_toggle_group,STAR_PT_main,STAR_PT_armature,STAR_PT_bones,
]

def register():
    for c in CLASSES: bpy.utils.register_class(c)
    S=bpy.types.Scene
    S.star_host          = bpy.props.StringProperty(name="Host",default="127.0.0.1")
    S.star_port          = bpy.props.IntProperty(name="Port",default=5005,min=1024,max=65535)
    S.star_armature      = bpy.props.PointerProperty(name="Armature",type=bpy.types.Object,
                               poll=lambda self,obj: obj.type=='ARMATURE')
    S.star_bone_mappings = bpy.props.CollectionProperty(type=STAR_BoneMapping)
    S.star_smooth_alpha  = bpy.props.FloatProperty(name="Yumusatma",default=0.35,min=0.05,max=1.0,precision=2)
    S.star_conf_threshold= bpy.props.FloatProperty(name="Min Gorunurluk",default=0.4,min=0.0,max=1.0,precision=2)
    S.star_max_delta     = bpy.props.FloatProperty(name="Max Sicrama",default=0.3,min=0.01,max=3.14,precision=2)
    print("[STAR v2.4] Addon yuklendi")

def unregister():
    global g_running,g_udp_socket,g_thread
    g_running=False
    if g_thread: g_thread.join(timeout=1.)
    if g_udp_socket: g_udp_socket.close()
    for p in ("star_host","star_port","star_armature","star_bone_mappings",
              "star_smooth_alpha","star_conf_threshold","star_max_delta"):
        if hasattr(bpy.types.Scene,p): delattr(bpy.types.Scene,p)
    for c in reversed(CLASSES): bpy.utils.unregister_class(c)
    print("[STAR v2.4] Addon kaldirildi")

if __name__=="__main__": register()
