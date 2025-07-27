import numpy as np
from vispy import app, scene
from vispy.color import Color
import math
import socket
import json
import threading
import queue
import time

class SpotRobot:
    def __init__(self, canvas):
        """Initialize Boston Dynamics Spot robot with accurate skeleton visualization"""
        self.canvas = canvas
        self.view = canvas.central_widget.add_view()
        
        # Use TurntableCamera with Z-axis only rotation
        self.view.camera = scene.cameras.TurntableCamera(fov=60, distance=3, elevation=20)
        
        # Lock elevation and setup z-axis only rotation
        self.fixed_elevation = 20
        self.view.camera.elevation = self.fixed_elevation
        
        # Override the camera's mouse interaction for Z-axis only rotation
        @self.canvas.events.mouse_move.connect
        def on_mouse_move(event):
            if event.button == 1 and event.is_dragging:
                # Calculate rotation based on horizontal mouse movement only
                dx = event.pos[0] - event.last_event.pos[0]
                
                # Update azimuth (rotation around z-axis)
                self.view.camera.azimuth += dx * 0.5
                
                # Keep elevation locked
                self.view.camera.elevation = self.fixed_elevation
                
                # Update the display
                self.canvas.update()
        
        # Initialize joint positions based on actual Spot dimensions
        self.joint_positions = self._initialize_joints()
        self.bone_connections = self._create_skeleton_connections()
        
        # Create visual elements
        self._create_skeleton_visuals()
        
        # Setup camera
        self.view.camera.set_range()
        
        # Animation mode: 'procedural' or 'isaac_sim'
        self.animation_mode = 'procedural'
        self.time = 0
        
        # Procedural animation parameters (for fallback)
        self.running_speed = 2.0
        self.stride_length = 0.8
        self.body_bounce = 0.05
        self.leg_lift = 0.15
        
    def _initialize_joints(self):
        """Initialize joint positions for Spot's default standing pose"""
        # Joint indices mapping (16 components - no body point)
        self.joint_names = {
            'fl_hip': 0,
            'fl_uleg': 1,
            'fl_lleg': 2,
            'fl_foot': 3,
            'fr_hip': 4,
            'fr_uleg': 5,
            'fr_lleg': 6,
            'fr_foot': 7,
            'hl_hip': 8,
            'hl_uleg': 9,
            'hl_lleg': 10,
            'hl_foot': 11,
            'hr_hip': 12,
            'hr_uleg': 13,
            'hr_lleg': 14,
            'hr_foot': 15,
        }
        
        # Initialize with Spot's default standing pose (16 points)
        positions = np.zeros((16, 3), dtype=np.float32)
        
        # Body height for standing pose
        body_z = 0.0
        leg_stance_width = 0.3  # Legs positioned under body for stability
        
        # Front Left Leg - standing pose
        positions[0] = [0.29785, 0.055, body_z]                    # fl_hip
        positions[1] = [0.29785, 0.16594, body_z]                  # fl_uleg (same height as hip)
        positions[2] = [0.29785, leg_stance_width, -0.3205]        # fl_lleg (positioned under body)
        positions[3] = [0.29785, leg_stance_width, -0.657]         # fl_foot (on ground)
        
        # Front Right Leg - standing pose
        positions[4] = [0.29785, -0.055, body_z]                   # fr_hip
        positions[5] = [0.29785, -0.16594, body_z]                 # fr_uleg (same height as hip)
        positions[6] = [0.29785, -leg_stance_width, -0.3205]       # fr_lleg (positioned under body)
        positions[7] = [0.29785, -leg_stance_width, -0.657]        # fr_foot (on ground)
        
        # Hind Left Leg - standing pose
        positions[8] = [-0.29785, 0.055, body_z]                   # hl_hip
        positions[9] = [-0.29785, 0.16594, body_z]                 # hl_uleg (same height as hip)
        positions[10] = [-0.29785, leg_stance_width, -0.3205]      # hl_lleg (positioned under body)
        positions[11] = [-0.29785, leg_stance_width, -0.657]       # hl_foot (on ground)
        
        # Hind Right Leg - standing pose
        positions[12] = [-0.29785, -0.055, body_z]                 # hr_hip
        positions[13] = [-0.29785, -0.16594, body_z]               # hr_uleg (same height as hip)
        positions[14] = [-0.29785, -leg_stance_width, -0.3205]     # hr_lleg (positioned under body)
        positions[15] = [-0.29785, -leg_stance_width, -0.657]      # hr_foot (on ground)
        
        return positions
    
    def _create_skeleton_connections(self):
        """Define bone connections based on actual Spot robot kinematic chain"""
        connections = [
            # Connect the four hips together to form the body frame (no body point needed)
            [0, 4],   # fl_hip to fr_hip (front connection)
            [8, 12],  # hl_hip to hr_hip (rear connection)
            [0, 8],   # fl_hip to hl_hip (left side)
            [4, 12],  # fr_hip to hr_hip (right side)
            
            # Hip to upper leg connections (4 hip y-axis joints: fl_hy, fr_hy, hl_hy, hr_hy)
            [0, 1],   # fl_hip to fl_uleg
            [4, 5],   # fr_hip to fr_uleg
            [8, 9],   # hl_hip to hl_uleg
            [12, 13], # hr_hip to hr_uleg
            
            # Upper leg to lower leg connections (4 knee joints: fl_kn, fr_kn, hl_kn, hr_kn)
            [1, 2],   # fl_uleg to fl_lleg
            [5, 6],   # fr_uleg to fr_lleg
            [9, 10],  # hl_uleg to hl_lleg
            [13, 14], # hr_uleg to hr_lleg
            
            # Lower leg to foot connections (4 fixed ankle joints)
            [2, 3],   # fl_lleg to fl_foot
            [6, 7],   # fr_lleg to fr_foot
            [10, 11], # hl_lleg to hl_foot
            [14, 15], # hr_lleg to hr_foot
        ]
        
        return np.array(connections, dtype=np.uint32)
    
    def _create_skeleton_visuals(self):
        """Create the visual elements for the Spot robot skeleton"""
        # Joint markers with different colors for different parts (16 points)
        joint_colors = np.ones((16, 4), dtype=np.float32)
        
        # Front Left Leg - red
        joint_colors[0:4] = [1.0, 0.2, 0.2, 1.0]
        
        # Front Right Leg - green
        joint_colors[4:8] = [0.2, 1.0, 0.2, 1.0]
        
        # Hind Left Leg - yellow
        joint_colors[8:12] = [1.0, 1.0, 0.2, 1.0]
        
        # Hind Right Leg - magenta
        joint_colors[12:16] = [1.0, 0.2, 1.0, 1.0]
        
        # Create joint markers with bigger size
        self.joints = scene.Markers(
            pos=self.joint_positions,
            size=15,  # Increased from 8 to 20
            face_color=joint_colors,
            edge_color='white',
            edge_width=1,
            parent=self.view.scene
        )
        
        # Create bone lines
        self.bones = scene.Line(
            pos=self.joint_positions,
            connect=self.bone_connections,
            color='cyan',
            width=4,
            method='gl',
            antialias=True,
            parent=self.view.scene
        )
        
        # Create transparent ground plane with checkerboard pattern
        self._create_ground_plane()

    
    def _create_ground_plane(self):
        """Create a transparent ground plane with checkerboard pattern like in raytracing"""
        # Create a checkerboard pattern plane
        plane_size = 4
        checker_size = 0.2
        
        # Create checkerboard vertices and colors
        vertices = []
        colors = []
        faces = []
        
        # Generate checkerboard squares
        idx = 0
        for i in range(int(-plane_size/checker_size), int(plane_size/checker_size)):
            for j in range(int(-plane_size/checker_size), int(plane_size/checker_size)):
                x1, x2 = i * checker_size, (i + 1) * checker_size
                y1, y2 = j * checker_size, (j + 1) * checker_size
                z = -0.67  # Ground level
                
                # Add 4 vertices for this square
                vertices.extend([
                    [x1, y1, z],
                    [x2, y1, z],
                    [x2, y2, z],
                    [x1, y2, z]
                ])
                
                # Determine color based on checkerboard pattern
                if (i + j) % 2 == 0:
                    square_color = [0.9, 0.9, 0.9, 0.3]  # Light squares
                else:
                    square_color = [0.3, 0.3, 0.3, 0.3]  # Dark squares
                
                # Add colors for 4 vertices
                colors.extend([square_color] * 4)
                
                # Add faces (2 triangles per square)
                faces.extend([
                    [idx, idx + 1, idx + 2],
                    [idx, idx + 2, idx + 3]
                ])
                
                idx += 4
        
        # Create mesh
        vertices = np.array(vertices, dtype=np.float32)
        colors = np.array(colors, dtype=np.float32)
        faces = np.array(faces, dtype=np.uint32)
        
        self.ground_plane = scene.Mesh(
            vertices=vertices,
            faces=faces,
            vertex_colors=colors,
            parent=self.view.scene
        )
        
        # Enable transparency
        self.ground_plane.set_gl_state(
            blend=True,
            blend_func=('src_alpha', 'one_minus_src_alpha'),
            depth_test=True
        )
    
    
    def _calculate_leg_position(self, leg_phase, hip_pos, leg_id, is_front=True):
        """Calculate leg position for proper Spot trot gait"""
        # Trot gait parameters
        stride_progress = (leg_phase % (2 * math.pi)) / (2 * math.pi)
        
        # Define stance and swing phases for trot gait
        stance_duration = 0.6  # 60% of cycle is stance (foot on ground)
        swing_duration = 0.4   # 40% of cycle is swing (foot in air)
        
        # Keep legs in stable position (no lateral movement)
        if hip_pos[1] > 0:  # Left side
            leg_y = 0.25
        else:  # Right side  
            leg_y = -0.25
        
        if stride_progress < stance_duration:  # Stance phase - foot on ground
            # During stance, foot moves backward relative to body (body moves forward)
            stance_progress = stride_progress / stance_duration
            foot_offset_x = self.stride_length * (0.5 - stance_progress)
            foot_z = -0.657  # On ground
                
        else:  # Swing phase - foot in air
            # During swing, foot lifts up and moves forward
            swing_progress = (stride_progress - stance_duration) / swing_duration
            foot_offset_x = self.stride_length * (-0.5 + swing_progress)
            
            # Foot follows an arc during swing (characteristic Spot movement)
            foot_z = -0.657 + self.leg_lift * math.sin(swing_progress * math.pi)
        
        # Calculate final positions maintaining proper Spot kinematics
        foot_x = hip_pos[0] + foot_offset_x
        
        # Calculate intermediate joint positions (proper leg kinematics)
        # Lower leg position - interpolated between hip and foot
        lleg_x = hip_pos[0] + foot_offset_x * 0.3
        lleg_z = (hip_pos[2] + foot_z) * 0.5  # Midpoint between hip and foot
        
        return {
            'hip': hip_pos,  # Hip follows body movement
            'uleg': [hip_pos[0], 0.16594 if hip_pos[1] > 0 else -0.16594, hip_pos[2]],  # Fixed USD offset
            'lleg': [lleg_x, leg_y, lleg_z],  # Calculated intermediate position
            'foot': [foot_x, leg_y, foot_z]  # Final foot position
        }
    
    def update_from_isaac_sim_socket(self, robot_data):
        """Update robot positions from Isaac Sim socket data"""
        # Map Isaac Sim component names to our joint indices
        isaac_to_joint_mapping = {
            'fl_hip': 0, 'fl_uleg': 1, 'fl_lleg': 2, 'fl_foot': 3,
            'fr_hip': 4, 'fr_uleg': 5, 'fr_lleg': 6, 'fr_foot': 7,
            'hl_hip': 8, 'hl_uleg': 9, 'hl_lleg': 10, 'hl_foot': 11,
            'hr_hip': 12, 'hr_uleg': 13, 'hr_lleg': 14, 'hr_foot': 15,
        }
        
        # Update joint positions from Isaac Sim data
        components = robot_data.get('components', {})
        
        # First pass: find the lowest Z value (should be the feet on the ground)
        min_z = float('inf')
        for component_name in ['fl_foot', 'fr_foot', 'hl_foot', 'hr_foot']:
            if component_name in components:
                z_pos = components[component_name]['position'][2]
                min_z = min(min_z, z_pos)
        
        # Calculate offset to place feet at ground level (-0.657 in our coordinate system)
        z_offset = min_z - (-0.657)
        
        for component_name, joint_idx in isaac_to_joint_mapping.items():
            if component_name in components:
                # Convert Isaac Sim coordinates to our coordinate system
                isaac_pos = components[component_name]['position']
                
                # Apply coordinate conversion with corrected Z offset
                self.joint_positions[joint_idx] = [
                    isaac_pos[0],           # X
                    isaac_pos[1] + 2.0,     # Y (offset to center around 0)
                    isaac_pos[2] - z_offset # Z (offset to align feet with ground)
                ]
        
        # Update visuals
        self.joints.set_data(pos=self.joint_positions, size=15)
        self.bones.set_data(pos=self.joint_positions)
    
    # Keep the existing procedural animation as fallback
    def update_animation(self, dt):
        """Update animation - either procedural or from Isaac Sim"""
        if self.animation_mode == 'procedural':
            self._update_procedural_animation(dt)
    
    def _update_procedural_animation(self, dt):
        """Original procedural animation (fallback)"""
        self.time += dt
        
        # Body movement - slight bounce and forward motion
        body_bounce = self.body_bounce * math.sin(self.time * 8)
        body_z = body_bounce
        
        # Trot gait timing - diagonal pairs move together
        gait_frequency = 4  # Steps per second
        phase_offset = self.time * gait_frequency
        
        # Diagonal pair 1: Front Left + Rear Right (phase 0)
        fl_phase = phase_offset
        rr_phase = phase_offset  # Same phase as front left
        
        # Diagonal pair 2: Front Right + Rear Left (phase π - opposite)
        fr_phase = phase_offset + math.pi
        rl_phase = phase_offset + math.pi  # Same phase as front right
        
        # Front Left leg (indices 0,1,2,3)
        fl_hip_pos = [0.29785, 0.055, body_z]
        fl_positions = self._calculate_leg_position(fl_phase, fl_hip_pos, 'fl', True)
        self.joint_positions[0] = fl_positions['hip']
        self.joint_positions[1] = fl_positions['uleg']
        self.joint_positions[2] = fl_positions['lleg']
        self.joint_positions[3] = fl_positions['foot']
        
        # Front Right leg (indices 4,5,6,7)
        fr_hip_pos = [0.29785, -0.055, body_z]
        fr_positions = self._calculate_leg_position(fr_phase, fr_hip_pos, 'fr', True)
        self.joint_positions[4] = fr_positions['hip']
        self.joint_positions[5] = fr_positions['uleg']
        self.joint_positions[6] = fr_positions['lleg']
        self.joint_positions[7] = fr_positions['foot']
        
        # Hind Left leg (indices 8,9,10,11) - moves with Front Right
        hl_hip_pos = [-0.29785, 0.055, body_z]
        hl_positions = self._calculate_leg_position(rl_phase, hl_hip_pos, 'hl', False)
        self.joint_positions[8] = hl_positions['hip']
        self.joint_positions[9] = hl_positions['uleg']
        self.joint_positions[10] = hl_positions['lleg']
        self.joint_positions[11] = hl_positions['foot']
        
        # Hind Right leg (indices 12,13,14,15) - moves with Front Left
        hr_hip_pos = [-0.29785, -0.055, body_z]
        hr_positions = self._calculate_leg_position(rr_phase, hr_hip_pos, 'hr', False)
        self.joint_positions[12] = hr_positions['hip']
        self.joint_positions[13] = hr_positions['uleg']
        self.joint_positions[14] = hr_positions['lleg']
        self.joint_positions[15] = hr_positions['foot']
        
        # Update visuals
        self.joints.set_data(pos=self.joint_positions, size=15)
        self.bones.set_data(pos=self.joint_positions)
    
    def set_animation_mode(self, mode):
        """Switch between 'procedural' and 'isaac_sim' animation modes"""
        self.animation_mode = mode


class SocketReceiver:
    def __init__(self, port=5555):
        self.port = port
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.bind(('127.0.0.1', port))
        self.socket.setblocking(False)
        self.data_queue = queue.Queue()
        self.running = True
        self.thread = threading.Thread(target=self._receive_loop)
        self.thread.daemon = True
        self.thread.start()
        print(f"Socket receiver initialized on port {port}")
        
    def _receive_loop(self):
        """Background thread to receive socket data"""
        while self.running:
            try:
                data, addr = self.socket.recvfrom(65536)
                robot_data = json.loads(data.decode('utf-8'))
                # Put latest data in queue, removing old data if queue is full
                if self.data_queue.full():
                    try:
                        self.data_queue.get_nowait()
                    except:
                        pass
                self.data_queue.put(robot_data)
            except socket.error:
                # No data available
                time.sleep(0.001)
            except Exception as e:
                print(f"Error receiving socket data: {e}")
                
    def get_latest_data(self):
        """Get the latest robot data from queue"""
        latest_data = None
        # Get all available data and keep only the latest
        while not self.data_queue.empty():
            try:
                latest_data = self.data_queue.get_nowait()
            except:
                break
        return latest_data
    
    def close(self):
        """Close the socket receiver"""
        self.running = False
        self.thread.join()
        self.socket.close()


class SpotVisualization:
    def __init__(self, use_socket=True, socket_port=5555):
        """Main application class for Boston Dynamics Spot robot visualization"""
        self.canvas = scene.SceneCanvas(
            keys='interactive',
            size=(1200, 800),
            show=True,
            title="Boston Dynamics Spot Robot - Socket Visualization"
        )
        
        # Create Spot robot
        self.spot = SpotRobot(self.canvas)
        
        # Initialize socket receiver if enabled
        self.use_socket = use_socket
        self.socket_receiver = None
        if self.use_socket:
            try:
                self.socket_receiver = SocketReceiver(socket_port)
                self.spot.set_animation_mode('isaac_sim')
                print("Socket mode enabled - waiting for Isaac Sim data...")
            except Exception as e:
                print(f"Failed to initialize socket receiver: {e}")
                print("Falling back to procedural animation")
                self.use_socket = False
                self.spot.set_animation_mode('procedural')
        
        # Setup animation timer
        self.timer = app.Timer(interval=1/60, connect=self.update_frame, start=True)
        self.last_time = 0
        self.last_data_time = time.time()
        
        # Add instructions
        self.add_instructions()
        
        # Camera controls
        self.setup_camera_controls()
    
    def add_instructions(self):
        """Add instruction text to the canvas"""
        mode_text = "Socket Mode - Receiving from Isaac Sim" if self.use_socket else "Procedural Animation Mode"
        instructions = [
            "Boston Dynamics Spot Robot",
            "16 Body Components | 12 Joints",
            f"Mode: {mode_text}",
            "",
            "Controls:",
            "- Mouse drag: Rotate around Z-axis only",
            "- Mouse wheel: Zoom in/out",
            "- Space: Pause/Resume animation",
            "- R: Reset camera view",
            "- 1: Side view",
            "- 2: Front view",
            "- 3: Top view",
            "- 4: Bottom view",
            "- P: Toggle procedural mode (fallback)",
            "- Q: Quit"
        ]
        
        instruction_text = "\n".join(instructions)
        self.text = scene.Text(
            instruction_text,
            pos=(10, 220),
            anchor_x='left',
            anchor_y='top',
            font_size=11,
            color='white',
            parent=self.canvas.scene
        )
    
    def setup_camera_controls(self):
        """Setup additional camera controls"""
        @self.canvas.events.key_press.connect
        def on_key_press(event):
            if event.text == ' ':
                # Toggle pause
                if self.timer.running:
                    self.timer.stop()
                else:
                    self.timer.start()
            elif event.text == 'r':
                # Reset camera view
                self.spot.view.camera.elevation = 20
                self.spot.view.camera.azimuth = 0
                self.spot.view.camera.distance = 3
                self.spot.view.camera.set_range()
            elif event.text == '1':
                # Side view
                self.spot.view.camera.elevation = 0
                self.spot.view.camera.azimuth = 90
                self.spot.view.camera.distance = 3
            elif event.text == '2':
                # Front view
                self.spot.view.camera.elevation = 0
                self.spot.view.camera.azimuth = 0
                self.spot.view.camera.distance = 3
            elif event.text == '3':
                # Top view (rotated 90 degrees)
                self.spot.view.camera.elevation = 90
                self.spot.view.camera.azimuth = 90
                self.spot.view.camera.distance = 3
            elif event.text == '4':
                # Bottom view (rotated 90 degrees)
                self.spot.view.camera.elevation = -90
                self.spot.view.camera.azimuth = 90
                self.spot.view.camera.distance = 3
            elif event.text == 'p':
                # Toggle procedural mode
                if self.spot.animation_mode == 'procedural':
                    self.spot.set_animation_mode('isaac_sim')
                    print("Switched to Isaac Sim mode")
                else:
                    self.spot.set_animation_mode('procedural')
                    print("Switched to procedural mode")
            elif event.text == 'q':
                self.close()
        
        @self.canvas.events.mouse_wheel.connect
        def on_mouse_wheel(event):
            # Zoom in/out with mouse wheel
            zoom_factor = 1.1 ** event.delta[1]
            self.spot.view.camera.distance *= zoom_factor
            self.canvas.update()
    
    def update_frame(self, event):
        """Update animation frame"""
        current_time = event.elapsed
        dt = current_time - self.last_time
        self.last_time = current_time
        
        # Check for socket data if in socket mode
        if self.use_socket and self.socket_receiver:
            robot_data = self.socket_receiver.get_latest_data()
            if robot_data:
                self.spot.update_from_isaac_sim_socket(robot_data)
                self.last_data_time = time.time()
            else:
                # If no data for 2 seconds, switch to procedural
                if time.time() - self.last_data_time > 2.0:
                    if self.spot.animation_mode != 'procedural':
                        print("No socket data received - switching to procedural animation")
                        self.spot.set_animation_mode('procedural')
        
        # Update procedural animation if in that mode
        if self.spot.animation_mode == 'procedural':
            self.spot.update_animation(dt)
        
        # Update canvas
        self.canvas.update()

    def close(self):
        """Clean up resources and close"""
        if self.socket_receiver:
            self.socket_receiver.close()
        self.canvas.close()


def main():
    """Main function to run the Spot robot visualization"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Spot Robot Skeleton Visualization')
    parser.add_argument('--no-socket', action='store_true', help='Disable socket mode and use procedural animation')
    parser.add_argument('--port', type=int, default=5555, help='Socket port to listen on (default: 5555)')
    args = parser.parse_args()
    
    print("Starting Boston Dynamics Spot Robot Visualization...")
    print("Structure: 16 Body Components, 12 Joints")
    print("Controls:")
    print("- Mouse drag: Rotate around Z-axis only")
    print("- Mouse wheel: Zoom in/out")
    print("- Space: Pause/Resume animation")
    print("- R: Reset camera view")
    print("- 1/2/3/4: Side/Front/Top/Bottom views")
    print("- P: Toggle procedural mode")
    print("- Q: Quit")
    
    use_socket = not args.no_socket
    app_instance = SpotVisualization(use_socket=use_socket, socket_port=args.port)
    
    app.run()


if __name__ == '__main__':
    main()