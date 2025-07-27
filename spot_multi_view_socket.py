import numpy as np
from vispy import app, scene
from vispy.scene import visuals
from vispy.color import Color
import math
import socket
import json
import threading
import queue
import time
from collections import deque

# Import the SpotRobot class from the original file
from spot_wheel_vispy_socket import SpotRobot, SocketReceiver


class ActionPlotter:
    """Plots action timeseries data in multiple subplots"""
    
    def __init__(self, box, n_joints=12, history_length=200):
        self.box = box
        self.n_joints = n_joints
        self.history_length = history_length
        
        # Create view with 2D camera
        self.view = box.add_view()
        self.view.camera = 'panzoom'
        self.view.camera.set_range(x=(0, history_length), y=(-1.5, 1.5))
        
        # Initialize data buffers for each joint
        self.time_buffer = deque(maxlen=history_length)
        self.action_buffers = [deque(maxlen=history_length) for _ in range(n_joints)]
        
        # Create lines for each joint action
        self.lines = []
        colors = self._generate_colors(n_joints)
        
        for i in range(n_joints):
            line = visuals.Line(color=colors[i], width=2)
            self.view.add(line)
            self.lines.append(line)
            
        # Add grid lines
        self._add_grid()
        
        # Add joint labels
        self._add_labels()
        
        # Initialize time counter
        self.time_counter = 0
        
    def _generate_colors(self, n):
        """Generate distinct colors for each joint"""
        import colorsys
        colors = []
        for i in range(n):
            hue = i / n
            # Convert HSL to RGB
            rgb = colorsys.hsv_to_rgb(hue, 0.8, 0.9)
            colors.append(rgb)
        return colors
    
    def _add_grid(self):
        """Add grid lines to the plot"""
        # Horizontal grid lines
        for y in [-1.0, -0.5, 0.0, 0.5, 1.0]:
            line = visuals.Line(
                pos=np.array([[0, y], [self.history_length, y]]),
                color=(0.3, 0.3, 0.3, 0.5),
                width=1
            )
            self.view.add(line)
            
        # Vertical grid lines (every 50 time steps)
        for x in range(0, self.history_length + 1, 50):
            line = visuals.Line(
                pos=np.array([[x, -1.5], [x, 1.5]]),
                color=(0.3, 0.3, 0.3, 0.5),
                width=1
            )
            self.view.add(line)
    
    def _add_labels(self):
        """Add labels for joint names"""
        joint_names = [
            'fl_hx', 'fr_hx', 'hl_hx', 'hr_hx',
            'fl_hy', 'fr_hy', 'hl_hy', 'hr_hy', 
            'fl_kn', 'fr_kn', 'hl_kn', 'hr_kn'
        ]
        
        # Create legend text
        legend_text = "Joint Actions:\n"
        colors = self._generate_colors(self.n_joints)
        
        for i, name in enumerate(joint_names[:self.n_joints]):
            color_rgb = colors[i]
            legend_text += f"{name} "
            
        self.legend = scene.Text(
            legend_text,
            pos=(10, 10),
            anchor_x='left',
            anchor_y='bottom',
            font_size=10,
            color='white',
            parent=self.box.scene
        )
    
    def update_data(self, action_data):
        """Update plot with new action data"""
        # Add new time point
        self.time_buffer.append(self.time_counter)
        self.time_counter += 1
        
        # Add action data for each joint
        for i in range(self.n_joints):
            if i < len(action_data):
                self.action_buffers[i].append(action_data[i])
            else:
                self.action_buffers[i].append(0.0)
        
        # Update line data
        if len(self.time_buffer) > 1:
            x_data = np.array(list(self.time_buffer))
            
            # Normalize x_data to fit in view
            x_normalized = (x_data - x_data[0]) * (self.history_length / max(1, x_data[-1] - x_data[0]))
            
            for i in range(self.n_joints):
                y_data = np.array(list(self.action_buffers[i]))
                if len(y_data) > 0:
                    pos = np.column_stack([x_normalized[-len(y_data):], y_data])
                    self.lines[i].set_data(pos=pos)


class CommandPlotter:
    """Plots command data (vx, vy, wz)"""
    
    def __init__(self, box, history_length=200):
        self.box = box
        self.history_length = history_length
        
        # Create view
        self.view = box.add_view()
        self.view.camera = 'panzoom'
        self.view.camera.set_range(x=(0, history_length), y=(-3, 3))
        
        # Initialize data buffers
        self.time_buffer = deque(maxlen=history_length)
        self.vx_buffer = deque(maxlen=history_length)
        self.vy_buffer = deque(maxlen=history_length)
        self.wz_buffer = deque(maxlen=history_length)
        
        # Create lines
        self.vx_line = visuals.Line(color='red', width=3)
        self.vy_line = visuals.Line(color='green', width=3)
        self.wz_line = visuals.Line(color='blue', width=3)
        
        self.view.add(self.vx_line)
        self.view.add(self.vy_line)
        self.view.add(self.wz_line)
        
        # Add grid
        self._add_grid()
        
        # Add legend
        self.legend = scene.Text(
            "Commands: vx (red), vy (green), wz (blue)",
            pos=(10, 10),
            anchor_x='left',
            anchor_y='bottom',
            font_size=10,
            color='white',
            parent=self.box.scene
        )
        
        self.time_counter = 0
    
    def _add_grid(self):
        """Add grid lines"""
        # Horizontal grid lines
        for y in [-3, -2, -1, 0, 1, 2, 3]:
            line = visuals.Line(
                pos=np.array([[0, y], [self.history_length, y]]),
                color=(0.3, 0.3, 0.3, 0.5),
                width=1
            )
            self.view.add(line)
    
    def update_data(self, command_data):
        """Update command plot"""
        self.time_buffer.append(self.time_counter)
        self.time_counter += 1
        
        # Extract command values
        if len(command_data) >= 3:
            self.vx_buffer.append(command_data[0])
            self.vy_buffer.append(command_data[1])
            self.wz_buffer.append(command_data[2])
        else:
            self.vx_buffer.append(0.0)
            self.vy_buffer.append(0.0)
            self.wz_buffer.append(0.0)
        
        # Update lines
        if len(self.time_buffer) > 1:
            x_data = np.array(list(self.time_buffer))
            x_normalized = (x_data - x_data[0]) * (self.history_length / max(1, x_data[-1] - x_data[0]))
            
            # Update vx line
            vx_data = np.array(list(self.vx_buffer))
            if len(vx_data) > 0:
                pos = np.column_stack([x_normalized[-len(vx_data):], vx_data])
                self.vx_line.set_data(pos=pos)
            
            # Update vy line
            vy_data = np.array(list(self.vy_buffer))
            if len(vy_data) > 0:
                pos = np.column_stack([x_normalized[-len(vy_data):], vy_data])
                self.vy_line.set_data(pos=pos)
            
            # Update wz line
            wz_data = np.array(list(self.wz_buffer))
            if len(wz_data) > 0:
                pos = np.column_stack([x_normalized[-len(wz_data):], wz_data])
                self.wz_line.set_data(pos=pos)


class SpotMultiView:
    """Main application with robot visualization and action plots"""
    
    def __init__(self, use_socket=True, socket_port=5555):
        # Create canvas
        self.canvas = scene.SceneCanvas(
            keys='interactive',
            size=(1600, 900),
            show=True,
            title="Spot Robot Multi-View: Skeleton + Actions"
        )
        
        # Create grid layout
        grid = self.canvas.central_widget.add_grid()
        
        # Create boxes for different views
        robot_box = grid.add_widget(row=0, col=0, row_span=2, bgcolor=(0.1, 0.1, 0.1, 1))
        action_raw_box = grid.add_widget(row=0, col=1, bgcolor=(0.15, 0.15, 0.15, 1))
        action_scaled_box = grid.add_widget(row=1, col=1, bgcolor=(0.15, 0.15, 0.15, 1))
        command_box = grid.add_widget(row=2, col=0, col_span=2, bgcolor=(0.1, 0.1, 0.1, 1))
        
        # Create robot visualization (reuse existing class)
        self.spot = SpotRobot(self.canvas)
        # Override the view to use our box
        self.spot.view = robot_box.add_view()
        self.spot.view.camera = scene.cameras.TurntableCamera(fov=60, distance=3, elevation=20)
        
        # Re-create visuals in the new view
        self.spot._create_skeleton_visuals()
        
        # Create action plotters
        self.action_raw_plotter = ActionPlotter(action_raw_box, n_joints=12)
        self.action_scaled_plotter = ActionPlotter(action_scaled_box, n_joints=12)
        self.command_plotter = CommandPlotter(command_box)
        
        # Add titles
        self._add_titles(robot_box, action_raw_box, action_scaled_box, command_box)
        
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
                print("Falling back to procedural animation")
                self.use_socket = False
                self.spot.set_animation_mode('procedural')
        
        # Animation timer
        self.timer = app.Timer(interval=1/60, connect=self.update_frame, start=True)
        self.last_time = 0
        self.last_data_time = time.time()
        
        # Setup controls
        self.setup_controls()
    
    def _add_titles(self, robot_box, action_raw_box, action_scaled_box, command_box):
        """Add titles to each box"""
        # Robot view title
        scene.Text(
            "Robot Skeleton View",
            pos=(10, 10),
            anchor_x='left',
            anchor_y='top',
            font_size=14,
            color='white',
            parent=robot_box.scene
        )
        
        # Action raw title
        scene.Text(
            "Raw Policy Actions",
            pos=(10, 10),
            anchor_x='left',
            anchor_y='top',
            font_size=14,
            color='white',
            parent=action_raw_box.scene
        )
        
        # Action scaled title
        scene.Text(
            "Scaled Joint Positions",
            pos=(10, 10),
            anchor_x='left',
            anchor_y='top',
            font_size=14,
            color='white',
            parent=action_scaled_box.scene
        )
        
        # Command title
        scene.Text(
            "User Commands",
            pos=(10, 10),
            anchor_x='left',
            anchor_y='top',
            font_size=14,
            color='white',
            parent=command_box.scene
        )
    
    def setup_controls(self):
        """Setup keyboard controls"""
        @self.canvas.events.key_press.connect
        def on_key_press(event):
            if event.text == ' ':
                if self.timer.running:
                    self.timer.stop()
                else:
                    self.timer.start()
            elif event.text == 'q':
                self.close()
    
    def update_frame(self, event):
        """Update all views"""
        current_time = event.elapsed
        dt = current_time - self.last_time
        self.last_time = current_time
        
        # Check for socket data
        if self.use_socket and self.socket_receiver:
            robot_data = self.socket_receiver.get_latest_data()
            if robot_data:
                # Update robot visualization
                self.spot.update_from_isaac_sim_socket(robot_data)
                
                # Update action plots
                if 'action' in robot_data:
                    action_data = robot_data['action']
                    
                    # Update raw action plot
                    if 'raw' in action_data:
                        self.action_raw_plotter.update_data(action_data['raw'])
                    
                    # Update scaled action plot
                    if 'scaled' in action_data:
                        self.action_scaled_plotter.update_data(action_data['scaled'])
                    
                    # Update command plot
                    if 'command' in action_data:
                        self.command_plotter.update_data(action_data['command'])
                
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
            
            # Generate fake action data for testing
            fake_actions = [0.5 * math.sin(current_time * 2 + i) for i in range(12)]
            self.action_raw_plotter.update_data(fake_actions)
            
            scaled_actions = [a * 0.2 for a in fake_actions]
            self.action_scaled_plotter.update_data(scaled_actions)
            
            fake_command = [
                math.sin(current_time),
                math.cos(current_time),
                0.5 * math.sin(current_time * 0.5)
            ]
            self.command_plotter.update_data(fake_command)
        
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
    args = parser.parse_args()
    
    print("Starting Spot Robot Multi-View Visualization...")
    print("Layout: Robot Skeleton + Action Timeseries")
    print("Controls:")
    print("- Space: Pause/Resume")
    print("- Q: Quit")
    
    use_socket = not args.no_socket
    app_instance = SpotMultiView(use_socket=use_socket, socket_port=args.port)
    
    app.run()


if __name__ == '__main__':
    main()