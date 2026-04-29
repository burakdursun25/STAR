import bpy
import math

def animate_rotation(obj, frame_start=1, frame_end=100):
    """Animates the active object spinning 360 degrees around the Z axis."""
    if not obj:
        print("No object selected to animate.")
        return

    # Clear existing animation data
    obj.animation_data_clear()
    
    # Ensure rotation mode is Euler
    obj.rotation_mode = 'XYZ'

    # Keyframe 1: Start rotation
    obj.rotation_euler[2] = 0.0
    obj.keyframe_insert(data_path="rotation_euler", index=2, frame=frame_start)

    # Keyframe 2: Full spin (360 degrees in radians)
    obj.rotation_euler[2] = math.radians(360.0)
    obj.keyframe_insert(data_path="rotation_euler", index=2, frame=frame_end)
    
    # Make animation linear and looping for all fcurves
    if obj.animation_data and obj.animation_data.action:
        for fcurve in obj.animation_data.action.fcurves:
            for keyframe in fcurve.keyframe_points:
                keyframe.interpolation = 'LINEAR'

    print(f"Rotation animation added to {obj.name} from frame {frame_start} to {frame_end}.")

if __name__ == "__main__":
    # Get the currently selected/active object in Blender
    active_obj = bpy.context.active_object
    animate_rotation(active_obj)
