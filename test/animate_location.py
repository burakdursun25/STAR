import bpy

def animate_location(obj, location, frame):
    """Sets the object's location and inserts a keyframe at the specified frame."""
    if not obj:
        print("No object selected to animate.")
        return

    # Update location and insert keyframe
    obj.location = location
    obj.keyframe_insert(data_path="location", frame=frame)

# Test kodları kaldırıldı. Bu dosya artık sadece fonksiyon barındırıyor.
