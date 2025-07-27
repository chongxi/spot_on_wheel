import numpy as np
from vispy import app, scene
from vispy.scene import visuals
import math
import time
from collections import deque
import colorsys

# Import the SocketReceiver from the original file
from spot_wheel_vispy_socket import SocketReceiver


class MinimalMultiView:
    """Minimal working multi-view with robot and action plots"""
    
    def __init__(self, use_socket=True, socket_port=5555, center_skeleton=True):
        # Create canvas
        self.canvas = scene.SceneCanvas(
            keys='interactive',
            size=(1600, 900),
            show=True,
            title="Spot Robot Multi-View: Skeleton + Actions"
        )
        
        # Store center_skeleton option
        self.center_skeleton = center_skeleton
        
        # Create grid layout
        grid = self.canvas.central_widget.add_grid()
        
        # Create boxes for different views
        self.robot_box = grid.add_widget(row=0, col=0, row_span=2, bgcolor=(0.1, 0.1, 0.1, 1))
        self.action_box = grid.add_widget(row=0, col=1, bgcolor=(0.15, 0.15, 0.15, 1))
        self.command_box = grid.add_widget(row=1, col=1, bgcolor=(0.1, 0.1, 0.1, 1))
        
        # Setup robot view
        self._setup_robot_view()
        
        # Setup action plot
        self._setup_action_plot()
        
        # Setup command plot
        self._setup_command_plot()
        
        # Sync the x-axis of action and command plots
        self._sync_plot_cameras()
        
        # Socket setup
        self.use_socket = use_socket
        self.socket_receiver = None
        if self.use_socket:
            try:
                self.socket_receiver = SocketReceiver(socket_port)
                self.spot.set_animation_mode('isaac_sim')
                print("Socket mode enabled - waiting for Isaac Sim data...")
            except Exception as e:
                print(f"Failed to initialize socket receiver: {e}")
                self.use_socket = False
                self.spot.set_animation_mode('procedural')
        
        # Animation timer
        self.timer = app.Timer(interval=1/60, connect=self.update_frame, start=True)
        self.last_time = 0
        self.last_data_time = time.time()
        
        # Setup keyboard controls
        self._setup_controls()
        
    def _setup_robot_view(self):
        """Setup the robot skeleton view"""
        # Create a custom SpotRobot without initializing visuals yet
        self.spot = type('SpotRobot', (), {})()  # Empty object
        
        # Set up the view first
        self.spot.view = self.robot_box.add_view()
        self.spot.view.camera = scene.cameras.TurntableCamera(fov=60, distance=3, elevation=20)
        self.spot.canvas = self.canvas
        
        # Initialize robot properties
        self.spot.joint_positions = self._initialize_joints()
        self.spot.bone_connections = self._create_skeleton_connections()
        self.spot.animation_mode = 'procedural'
        self.spot.time = 0
        self.spot.running_speed = 2.0
        self.spot.stride_length = 0.8
        self.spot.body_bounce = 0.05
        self.spot.leg_lift = 0.15
        
        # Now create visuals in the correct view
        self._create_skeleton_visuals_for_spot()
        
        # Copy necessary methods from SpotRobot class
        from spot_wheel_vispy_socket import SpotRobot
        self.spot.set_animation_mode = lambda mode: setattr(self.spot, 'animation_mode', mode)
        self.spot.update_animation = SpotRobot.update_animation.__get__(self.spot)
        self.spot._update_procedural_animation = SpotRobot._update_procedural_animation.__get__(self.spot)
        self.spot._calculate_leg_position = SpotRobot._calculate_leg_position.__get__(self.spot)
        
        # Create a wrapper for update_from_isaac_sim_socket to handle centering
        original_update = SpotRobot.update_from_isaac_sim_socket.__get__(self.spot)
        
        def centered_update(robot_data):
            # Call original update
            original_update(robot_data)
            
            # If centering is enabled, adjust all positions to center the body
            if self.center_skeleton:
                # Calculate the center of the body frame (average of the 4 hip joints)
                # Indices: 0=FL hip, 4=FR hip, 8=HL hip, 12=HR hip
                body_center = (self.spot.joint_positions[0] + 
                              self.spot.joint_positions[4] + 
                              self.spot.joint_positions[8] + 
                              self.spot.joint_positions[12]) / 4.0
                body_center[2] = 0  # Keep vertical offset
                
                # Subtract body center from all joints
                self.spot.joint_positions -= body_center
                
                # Update visuals
                self.spot.joints.set_data(pos=self.spot.joint_positions)
                self.spot.bones.set_data(pos=self.spot.joint_positions)
        
        self.spot.update_from_isaac_sim_socket = centered_update
        
        # Add title as text overlay on canvas
        self.robot_title = scene.Text(
            "Robot Skeleton View",
            pos=(10, 10),
            anchor_x='left',
            anchor_y='bottom',
            font_size=14,
            color='white',
            parent=self.canvas.scene
        )
    
    def _initialize_joints(self):
        """Initialize joint positions for Spot's default standing pose"""
        # Copy from SpotRobot class
        positions = np.zeros((16, 3), dtype=np.float32)
        body_z = 0.0
        leg_stance_width = 0.3
        
        # Front Left Leg
        positions[0] = [0.29785, 0.055, body_z]
        positions[1] = [0.29785, 0.16594, body_z]
        positions[2] = [0.29785, leg_stance_width, -0.3205]
        positions[3] = [0.29785, leg_stance_width, -0.657]
        
        # Front Right Leg
        positions[4] = [0.29785, -0.055, body_z]
        positions[5] = [0.29785, -0.16594, body_z]
        positions[6] = [0.29785, -leg_stance_width, -0.3205]
        positions[7] = [0.29785, -leg_stance_width, -0.657]
        
        # Hind Left Leg
        positions[8] = [-0.29785, 0.055, body_z]
        positions[9] = [-0.29785, 0.16594, body_z]
        positions[10] = [-0.29785, leg_stance_width, -0.3205]
        positions[11] = [-0.29785, leg_stance_width, -0.657]
        
        # Hind Right Leg
        positions[12] = [-0.29785, -0.055, body_z]
        positions[13] = [-0.29785, -0.16594, body_z]
        positions[14] = [-0.29785, -leg_stance_width, -0.3205]
        positions[15] = [-0.29785, -leg_stance_width, -0.657]
        
        return positions
    
    def _create_skeleton_connections(self):
        """Define bone connections"""
        connections = [
            [0, 4], [8, 12], [0, 8], [4, 12],  # Body frame
            [0, 1], [4, 5], [8, 9], [12, 13],  # Hip to upper leg
            [1, 2], [5, 6], [9, 10], [13, 14], # Upper to lower leg
            [2, 3], [6, 7], [10, 11], [14, 15] # Lower leg to foot
        ]
        return np.array(connections, dtype=np.uint32)
    
    def _create_skeleton_visuals_for_spot(self):
        """Create visual elements for the robot"""
        # Joint colors
        joint_colors = np.ones((16, 4), dtype=np.float32)
        joint_colors[0:4] = [1.0, 0.2, 0.2, 1.0]    # Front Left - red
        joint_colors[4:8] = [0.2, 1.0, 0.2, 1.0]    # Front Right - green
        joint_colors[8:12] = [1.0, 1.0, 0.2, 1.0]   # Hind Left - yellow
        joint_colors[12:16] = [1.0, 0.2, 1.0, 1.0]  # Hind Right - magenta
        
        # Create joint markers
        self.spot.joints = scene.Markers(
            pos=self.spot.joint_positions,
            size=15,
            face_color=joint_colors,
            edge_color='white',
            edge_width=1,
            parent=self.spot.view.scene
        )
        
        # Create bone lines
        self.spot.bones = scene.Line(
            pos=self.spot.joint_positions,
            connect=self.spot.bone_connections,
            color='cyan',
            width=4,
            method='gl',
            antialias=True,
            parent=self.spot.view.scene
        )
        
        # Create ground plane
        self._create_ground_plane()
    
    def _create_ground_plane(self):
        """Create ground plane"""
        plane_size = 4
        checker_size = 0.2
        vertices = []
        colors = []
        faces = []
        
        idx = 0
        for i in range(int(-plane_size/checker_size), int(plane_size/checker_size)):
            for j in range(int(-plane_size/checker_size), int(plane_size/checker_size)):
                x1, x2 = i * checker_size, (i + 1) * checker_size
                y1, y2 = j * checker_size, (j + 1) * checker_size
                z = -0.67
                
                vertices.extend([
                    [x1, y1, z], [x2, y1, z],
                    [x2, y2, z], [x1, y2, z]
                ])
                
                if (i + j) % 2 == 0:
                    square_color = [0.9, 0.9, 0.9, 0.3]
                else:
                    square_color = [0.3, 0.3, 0.3, 0.3]
                
                colors.extend([square_color] * 4)
                faces.extend([
                    [idx, idx + 1, idx + 2],
                    [idx, idx + 2, idx + 3]
                ])
                idx += 4
        
        vertices = np.array(vertices, dtype=np.float32)
        colors = np.array(colors, dtype=np.float32)
        faces = np.array(faces, dtype=np.uint32)
        
        self.spot.ground_plane = scene.Mesh(
            vertices=vertices,
            faces=faces,
            vertex_colors=colors,
            parent=self.spot.view.scene
        )
        
        self.spot.ground_plane.set_gl_state(
            blend=True,
            blend_func=('src_alpha', 'one_minus_src_alpha'),
            depth_test=True
        )
        
    def _setup_action_plot(self):
        """Setup the action timeseries plot"""
        # Create view
        self.action_view = self.action_box.add_view()
        self.action_view.camera = 'panzoom'
        self.action_view.camera.set_range(x=(0, 200), y=(-3, 3))  # Synced with command plot
        
        # Data buffers
        self.action_history_length = 200
        self.n_joints = 12
        self.action_time_buffer = deque(maxlen=self.action_history_length)
        self.action_buffers = [deque(maxlen=self.action_history_length) for _ in range(self.n_joints)]
        self.action_time_counter = 0
        
        # Create lines for each joint
        self.action_lines = []
        colors = self._generate_colors(self.n_joints)
        
        for i in range(self.n_joints):
            line = visuals.Line(color=colors[i], width=2)
            self.action_view.add(line)
            self.action_lines.append(line)
        
        # Add grid lines
        self._add_grid_lines(self.action_view, self.action_history_length)
        
        # Add title
        self.action_title = scene.Text(
            "Raw Policy Actions (All Values Plotted)",
            pos=(610, 10),  # Adjusted position for second column
            anchor_x='left',
            anchor_y='bottom',
            font_size=14,
            color='white',
            parent=self.canvas.scene
        )
        
        
    def _setup_command_plot(self):
        """Setup the command plot"""
        # Create view
        self.command_view = self.command_box.add_view()
        self.command_view.camera = 'panzoom'
        self.command_view.camera.set_range(x=(0, 200), y=(-3, 3))
        
        # Data buffers
        self.command_time_buffer = deque(maxlen=200)
        self.vx_buffer = deque(maxlen=200)
        self.vy_buffer = deque(maxlen=200)
        self.wz_buffer = deque(maxlen=200)
        self.command_time_counter = 0
        
        # Create lines
        self.vx_line = visuals.Line(color='red', width=3)
        self.vy_line = visuals.Line(color='green', width=3)
        self.wz_line = visuals.Line(color='blue', width=3)
        
        self.command_view.add(self.vx_line)
        self.command_view.add(self.vy_line)
        self.command_view.add(self.wz_line)
        
        # Add grid
        self._add_grid_lines(self.command_view, 200)
        
        # Add title and legend
        self.command_title = scene.Text(
            "User Commands: vx (red), vy (green), wz (blue)",
            pos=(610, 460),  # Adjusted for bottom right
            anchor_x='left',
            anchor_y='bottom',
            font_size=14,
            color='white',
            parent=self.canvas.scene
        )
        
    def _add_grid_lines(self, view, x_max):
        """Add horizontal grid lines to a plot view"""
        for y in [-3, -2, -1, 0, 1, 2, 3]:
            line = visuals.Line(
                pos=np.array([[0, y], [x_max, y]]),
                color=(0.3, 0.3, 0.3, 0.5),
                width=1
            )
            view.add(line)
    
    def _generate_colors(self, n):
        """Generate distinct colors for each joint"""
        colors = []
        for i in range(n):
            hue = i / n
            rgb = colorsys.hsv_to_rgb(hue, 0.8, 0.9)
            colors.append(rgb)
        return colors
    
    def _sync_plot_cameras(self):
        """Sync the x-axis pan and zoom between action and command plots"""
        # Link the cameras together - this will sync their x-axis
        # The 'link' method synchronizes camera state between two cameras
        self.action_view.camera.link(self.command_view.camera, axis='x')
    
    def _setup_controls(self):
        """Setup keyboard controls"""
        @self.canvas.events.key_press.connect
        def _(event):  # Anonymous function since we don't need the name
            if event.text == ' ':
                if self.timer.running:
                    self.timer.stop()
                else:
                    self.timer.start()
            elif event.text == 'q':
                self.close()
    
    def _normalize_time_data(self, time_buffer, history_length):
        """Normalize time data for plotting"""
        if len(time_buffer) > 1:
            x_data = np.array(list(time_buffer))
            return (x_data - x_data[0]) * (history_length / max(1, x_data[-1] - x_data[0]))
        return None
    
    def update_action_plot(self, action_data):
        """Update action plot with new data"""
        # Add time point
        self.action_time_buffer.append(self.action_time_counter)
        self.action_time_counter += 1
        
        # Add action data
        for i in range(self.n_joints):
            value = action_data[i] if i < len(action_data) else 0.0
            self.action_buffers[i].append(value)
        
        # Update lines
        x_normalized = self._normalize_time_data(self.action_time_buffer, self.action_history_length)
        if x_normalized is not None:
            for i in range(self.n_joints):
                y_data = np.array(list(self.action_buffers[i]))
                if len(y_data) > 0:
                    pos = np.column_stack([x_normalized[-len(y_data):], y_data])
                    self.action_lines[i].set_data(pos=pos)
    
    def update_command_plot(self, command_data):
        """Update command plot"""
        self.command_time_buffer.append(self.command_time_counter)
        self.command_time_counter += 1
        
        # Extract values with defaults
        vx = command_data[0] if len(command_data) > 0 else 0.0
        vy = command_data[1] if len(command_data) > 1 else 0.0
        wz = command_data[2] if len(command_data) > 2 else 0.0
        
        self.vx_buffer.append(vx)
        self.vy_buffer.append(vy)
        self.wz_buffer.append(wz)
        
        # Update lines
        x_normalized = self._normalize_time_data(self.command_time_buffer, 200)
        if x_normalized is not None:
            for buffer, line in [(self.vx_buffer, self.vx_line),
                               (self.vy_buffer, self.vy_line),
                               (self.wz_buffer, self.wz_line)]:
                y_data = np.array(list(buffer))
                if len(y_data) > 0:
                    pos = np.column_stack([x_normalized[-len(y_data):], y_data])
                    line.set_data(pos=pos)
    
    def update_frame(self, event):
        """Update all views"""
        current_time = event.elapsed
        dt = current_time - self.last_time
        self.last_time = current_time
        
        # Check for socket data
        if self.use_socket and self.socket_receiver:
            robot_data = self.socket_receiver.get_latest_data()
            if robot_data:
                # Ensure we're in isaac_sim mode when receiving data
                if self.spot.animation_mode != 'isaac_sim':
                    print("Switching to Isaac Sim mode - data received")
                    self.spot.set_animation_mode('isaac_sim')
                
                # Update robot visualization
                self.spot.update_from_isaac_sim_socket(robot_data)
                
                # Update plots if action data available
                if 'action' in robot_data:
                    action_data = robot_data['action']
                    
                    # Plot all actions to see what's happening
                    if 'scaled' in action_data:
                        self.update_action_plot(action_data['scaled'])

                    # Always update command plot as it changes continuously
                    if 'command' in action_data:
                        self.update_command_plot(action_data['command'])
                
                # Joint position data is available but not plotted
                # Add another plot if needed for joint positions
                
                self.last_data_time = time.time()
            else:
                # Switch to procedural if no data
                if time.time() - self.last_data_time > 2.0:
                    if self.spot.animation_mode != 'procedural':
                        print("No socket data - switching to procedural")
                        self.spot.set_animation_mode('procedural')
        
        # Update procedural animation if needed
        if self.spot.animation_mode == 'procedural':
            self.spot.update_animation(dt)
            
            # Generate fake data for testing
            fake_actions = [0.5 * math.sin(current_time * 2 + i) for i in range(12)]
            self.update_action_plot(fake_actions)
            
            fake_command = [
                math.sin(current_time),
                math.cos(current_time),
                0.5 * math.sin(current_time * 0.5)
            ]
            self.update_command_plot(fake_command)
        
        # Update canvas
        self.canvas.update()
    
    def close(self):
        """Clean up and close"""
        if self.socket_receiver:
            self.socket_receiver.close()
        self.canvas.close()


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Spot Robot Multi-View Visualization')
    parser.add_argument('--no-socket', action='store_true', help='Disable socket mode')
    parser.add_argument('--port', type=int, default=5555, help='Socket port (default: 5555)')
    parser.add_argument('--center-skeleton', action='store_true', help='Keep skeleton centered in view (ignore position)')
    args = parser.parse_args()
    
    print("Starting Spot Robot Multi-View Visualization...")
    print("Layout: Robot Skeleton + Action Timeseries")
    print("Controls:")
    print("- Space: Pause/Resume")
    print("- Q: Quit")
    if args.center_skeleton:
        print("- Skeleton centering: ENABLED")
    
    use_socket = not args.no_socket
    MinimalMultiView(use_socket=use_socket, socket_port=args.port, center_skeleton=args.center_skeleton)
    
    app.run()


if __name__ == '__main__':
    main()