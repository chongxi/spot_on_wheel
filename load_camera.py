import omni
import omni.usd
import omni.kit.commands
import math
import numpy as np
from pxr import Gf, Sdf
from omni.isaac.sensor import Camera

def load_robot_camera(camera_path="/World/Spot/body/front_cam", 
                      position=[0.5, 0.0, 0.0], 
                      rotation={"x": 90, "y": -70, "z": 0},
                      resolution=(640, 480),
                      frequency=25):
    """
    Minimal function to load a camera to a robot with specified configuration.
    
    Args:
        camera_path: USD path where the camera will be created
        position: [x, y, z] position relative to parent
        rotation: {"x", "y", "z"} rotation in degrees
        resolution: Camera resolution tuple
        frequency: Camera capture frequency
    
    Returns:
        Camera: Initialized Isaac Sim Camera object
    """
    try:
        # Get USD context
        usd_context = omni.usd.get_context()
        
        # 1. Create camera prim
        omni.kit.commands.execute('CreatePrimWithDefaultXform',
            prim_type='Camera',
            prim_path=camera_path,
            attributes={})
        
        # 2. Set camera position
        omni.kit.commands.execute('ChangeProperty',
            prop_path=Sdf.Path(f'{camera_path}.xformOp:translate'),
            value=Gf.Vec3d(position[0], position[1], position[2]),
            prev=Gf.Vec3d(0.0, 0.0, 0.0),
            usd_context_name=usd_context.get_name())
        
        # 3. Set camera rotation
        rotation_x = Gf.Rotation(Gf.Vec3d(1, 0, 0), rotation["x"])
        rotation_y = Gf.Rotation(Gf.Vec3d(0, 1, 0), rotation["y"])  
        rotation_z = Gf.Rotation(Gf.Vec3d(0, 0, 1), rotation["z"])
        
        # Combine rotations (ZYX order)
        combined_rotation = rotation_z * rotation_y * rotation_x
        quat = combined_rotation.GetQuat()
        
        omni.kit.commands.execute('ChangeProperty',
            prop_path=Sdf.Path(f'{camera_path}.xformOp:orient'),
            value=quat,
            prev=Gf.Quatd(1.0, Gf.Vec3d(0.0, 0.0, 0.0)),
            usd_context_name=usd_context.get_name())
        
        # 4. Initialize camera sensor
        camera = Camera(
            prim_path=camera_path,
            frequency=frequency,
            resolution=resolution
        )
        
        # 5. Initialize the camera (call this before simulation starts)
        camera.initialize()
        
        print(f"Camera created successfully at {camera_path}")
        return camera
        
    except Exception as e:
        print(f"Error creating camera: {e}")
        return None
