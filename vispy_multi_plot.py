# -*- coding: utf-8 -*-
"""
Four plots in four boxes layout (ultra-concise animated)
"""

import numpy as np
import sys
from vispy import scene, app
from vispy.scene import visuals

# Setup
canvas = scene.SceneCanvas(keys='interactive', size=(800, 600), title='Sin animation')
canvas.show()
grid = canvas.central_widget.add_grid()

# Data and layout
x = np.linspace(0, 10, 100)
colors = ['red', 'blue', 'green', 'yellow']
funcs = [lambda x, p: np.sin(x + p),
         lambda x, p: np.cos(x + p), 
         lambda x, p: np.sin(2*x + p),
         lambda x, p: np.cos(2*x + p)]

# Create boxes - plain and simple
box1 = grid.add_widget(row=0, col=0, bgcolor=(0.1, 0.1, 0.1, 1))  # Top left
box2 = grid.add_widget(row=0, col=1, bgcolor=(0.1, 0.1, 0.1, 1))  # Top right
box3 = grid.add_widget(row=1, col=0, bgcolor=(0.15, 0.15, 0.15, 1))  # Bottom left
box4 = grid.add_widget(row=1, col=1, bgcolor=(0.15, 0.15, 0.15, 1))  # Bottom right

boxes = [box1, box2, box3, box4]

lines = []
# Store only last N points for 3D trail
trail_length = 50  # Number of points to keep
trail_history = []
head_marker = None

for i, (box, color) in enumerate(zip(boxes, colors)):
    view = box.add_view()
    
    if i == 3:  # box4 - 3D plot
        view.camera = 'arcball'
        view.camera.set_range(x=(-1.5, 1.5), y=(-1.5, 1.5), z=(-1.5, 1.5))
        
        # Single line for the trail
        trail_line = visuals.Line(width=4)
        view.add(trail_line)
        
        # Add a marker at the head of the trail
        head_marker = visuals.Markers()
        head_marker.set_data(pos=np.array([[0, 0, 0]]), 
                           face_color=(1.0, 0.8, 0.0, 1.0),  # Orange face
                           edge_color=(0.8, 0.6, 0.0, 1.0),  # Darker orange edge
                           edge_width=2,
                           size=15,
                           symbol='o')
        view.add(head_marker)
        
        lines.append(trail_line)
    else:  # 2D plots
        view.camera = 'panzoom'
        view.camera.set_range(x=(0, 10), y=(-1.5, 1.5))
        line = visuals.Line(color=color, width=4)
        view.add(line)
        lines.append(line)

# Animation
phase = 0

def update(event):
    global phase, trail_history
    phase += 0.1
    
    for i, (line, func) in enumerate(zip(lines, funcs)):
        if i == 3:  # box4 - 3D plot using last values from other boxes
            # Use the last x value (rightmost point on each 2D curve)
            last_x = x[-1]  # 10.0
            
            # Get the last values from each of the first three functions
            head_x = funcs[0](last_x, phase) # sin(10 + phase) - from box1
            head_y = funcs[1](last_x, phase) * funcs[0](last_x, phase) # cos(10 + phase) - from box2
            head_z = funcs[2](last_x, phase) * funcs[1](last_x, phase) # sin(20 + phase) - from box3
            
            # Add current position to trail history
            trail_history.append([head_x, head_y, head_z])
            
            # Keep only the last trail_length points
            if len(trail_history) > trail_length:
                trail_history.pop(0)  # Remove oldest point
            
            if len(trail_history) > 1:
                # Create ordered transparency: oldest=0, newest=1
                alphas = np.linspace(0.1, 1.0, len(trail_history))  # Start from 0.1 to avoid completely invisible
                
                # Create colors with ordered alpha
                base_color = np.array([1.0, 0.8, 0.0])  # Orange
                colors_with_alpha = np.zeros((len(trail_history), 4))
                colors_with_alpha[:, :3] = base_color
                colors_with_alpha[:, 3] = alphas
                
                # Update trail line with only last N points
                line.set_data(pos=np.array(trail_history), color=colors_with_alpha)
                
                # Update head marker position
                head_marker.set_data(pos=np.array([[head_x, head_y, head_z]]), 
                                   face_color=(1.0, 0.8, 0.0, 1.0),
                                   edge_color=(0.8, 0.6, 0.0, 1.0),
                                   edge_width=2,
                                   size=15,
                                   symbol='o')
        else:  # 2D plots
            y_data = func(x, phase)
            line.set_data(pos=np.column_stack([x, y_data]))

timer = app.Timer(interval=0.033, connect=update, start=True)

if __name__ == '__main__':
    app.run()