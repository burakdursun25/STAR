import bpy

def animate_scale(obj, frame_start=1, frame_end=60, max_scale=2.0):
    """Animates the active object pulsing (scaling up and down)."""
    if not obj:
        print("No object selected to animate.")
        return

    # Clear existing animation data
    obj.animation_data_clear()

    # Keyframe 1: Start scale
    obj.scale = (1.0, 1.0, 1.0)
    obj.keyframe_insert(data_path="scale", frame=frame_start)

    # Keyframe 2: Scale up
    mid_frame = (frame_start + frame_end) // 2
    obj.scale = (max_scale, max_scale, max_scale)
    obj.keyframe_insert(data_path="scale", frame=mid_frame)

    # Keyframe 3: Scale down
    obj.scale = (1.0, 1.0, 1.0)
    obj.keyframe_insert(data_path="scale", frame=frame_end)

    print(f"Scale animation (pulse) added to {obj.name} from frame {frame_start} to {frame_end}.")

if __name__ == "__main__":
    # Get the currently selected/active object in Blender
    active_obj = bpy.context.active_object
    animate_scale(active_obj)
