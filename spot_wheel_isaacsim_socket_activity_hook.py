# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

import argparse
import sys
import socket
import json
import threading

import carb
import numpy as np
import torch
import omni.appwindow
from isaacsim.core.api import World
from isaacsim.sensors.physics import ContactSensor
from isaacsim.storage.native import get_assets_root_path
from omni.isaac.core.objects import DynamicCuboid
# from isaacsim.robot.policy.examples.robots import SpotFlatTerrainPolicy
from MySpotPolicy2 import SpotFlatTerrainPolicy
from extract_weights import extract_pytorch_model_from_jit
# from spot_policy_controller import SpotPolicyController
# from SpotFlatTerrainPolicy import SpotFlatTerrainPolicy
# from MySpotPolicy import SpotFlatTerrainPolicy
from isaacsim.core.utils.stage import add_reference_to_stage, get_stage_units
from load_camera import load_robot_camera

from pxr import Usd, UsdGeom, UsdPhysics, Gf
import omni.usd

parser = argparse.ArgumentParser()
parser.add_argument("--test", default=False, action="store_true", help="Run in test mode")
parser.add_argument("--socket-port", default=5555, type=int, help="Socket port for broadcasting robot state")
args, unknown = parser.parse_known_args()

assets_root_path = get_assets_root_path()
if assets_root_path is None:
    carb.log_error("Could not find Isaac Sim assets folder")
    simulation_app.close()
    sys.exit()


class SocketBroadcaster:
    def __init__(self, port=5555):
        self.port = port
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        self.broadcast_address = ('127.0.0.1', port)
        print(f"Socket broadcaster initialized on port {port}")
        
    def send_robot_state(self, robot_data):
        try:
            message = json.dumps(robot_data)
            self.socket.sendto(message.encode('utf-8'), self.broadcast_address)
        except Exception as e:
            print(f"Error sending socket data: {e}")
            
    def close(self):
        self.socket.close()


class SpotContactController:
    def __init__(self, physics_dt, render_dt):
        self._world = World(stage_units_in_meters=1.0, physics_dt=physics_dt, rendering_dt=render_dt)
        self._world.scene.add_default_ground_plane()
        
        # Initialize socket broadcaster
        self.broadcaster = SocketBroadcaster(args.socket_port)
        
        # Create a thin block in front of where Spot will spawn
        block_size = [1.0, 1.0, 0.02]  # 1m x 1m x 5cm thick
        block_position = [1.5, 0.0, 0.025]  # In front of Spot, half height above ground
        
        # Create block with physics
        self._block = self._world.scene.add(
            DynamicCuboid(
                prim_path="/World/ContactBlock",
                name="contact_block",
                position=block_position,
                scale=block_size,
                color=np.array([0.5, 0.5, 0.8]),  # Light blue color
                mass=1000.0  # Heavy block so it doesn't move
            )
        )
        
        add_reference_to_stage(usd_path='./treadmill_glass.usd', prim_path="/World/treadmill")
        stage = omni.usd.get_context().get_stage()
        treadmill_prim = stage.GetPrimAtPath("/World/treadmill")
        xformable = UsdGeom.Xformable(treadmill_prim)
        xformable.AddTranslateOp().Set(Gf.Vec3f(0.0, -2.0, 5.3))
        xformable.AddRotateXOp().Set(90)

        treadmill_mesh_prim = stage.GetPrimAtPath("/World/treadmill/WheelAssembly/RotatingWheel/WheelMesh")
        if not UsdPhysics.RigidBodyAPI(treadmill_mesh_prim):
            UsdPhysics.CollisionAPI.Apply(treadmill_mesh_prim)
            UsdPhysics.RigidBodyAPI.Apply(treadmill_mesh_prim)
            UsdPhysics.MassAPI.Apply(treadmill_mesh_prim).CreateMassAttr().Set(0.5)

        # Create Spot robot with policy controller
        self._spot = SpotFlatTerrainPolicy(
            prim_path="/World/Spot",
            name="Spot",
            position=np.array([0, -2, 10.9]),  # Start position
            # position=np.array([0, 2, 0.9]),  # Start position
        )

        # self._spot.policy = torch.jit.load("./assets/spot_policy_ftd3_hop.pt")
        # self._spot.jit_model = torch.jit.load("./assets/spot_policy_custom_rslrl.pt")
        # self._spot.policy = torch.jit.load("./official_spot_assets/spot_policy.pt")
        # model, jit_model = extract_pytorch_model_from_jit("./assets/spot_policy_custom_rslrl.pt", with_hooks=True)
        # model, jit_model = extract_pytorch_model_from_jit("./assets/spot_policy_ftd3_hop.pt", with_hooks=True)
        # model, jit_model = extract_pytorch_model_from_jit("./assets/spot_fast_td3_final.pt", with_hooks=True)
        model, jit_model = extract_pytorch_model_from_jit("./assets/rslrl_ppo_3B_policy.pt", with_hooks=True)
        self._spot.policy = model
        # self._spot.jit_model = jit_model

        self._spot._decimation = 10

        # self._spot = SpotPolicyController(
        #     prim_path="/World/Spot",
        #     name="Spot",
        #     position=np.array([0, -2, 10.9]),  # Start position
        #     policy_file_path="./assets/spot_policy_ftd3_hop.pt"
        # )

        joint_path = "/World/Spot/body/fixed_joint"
        joint = UsdPhysics.FixedJoint.Define(stage, joint_path)
        joint.CreateBody1Rel().SetTargets(["/World/Spot/body"])
        
        self._base_command = np.zeros(3)
        
        # Keyboard command mappings
        self._input_keyboard_mapping = {
            # forward command
            "NUMPAD_8": [2.0, 0.0, 0.0],
            "UP": [2.0, 0.0, 0.0],
            # back command
            "NUMPAD_2": [-1.0, 0.0, 0.0],
            "DOWN": [-1.0, 0.0, 0.0],
            # left command
            "NUMPAD_6": [0.0, -1.0, 0.0],
            "RIGHT": [0.0, -1.0, 0.0],
            # right command
            "NUMPAD_4": [0.0, 1.0, 0.0],
            "LEFT": [0.0, 1.0, 0.0],
            # yaw command (positive)
            "NUMPAD_7": [0.0, 0.0, 1.0],
            "N": [0.0, 0.0, 1.0],
            # yaw command (negative)
            "NUMPAD_9": [0.0, 0.0, -1.0],
            "M": [0.0, 0.0, -1.0],
        }
        
        self.needs_reset = False
        self._block_sensor = None
        self._world.reset()
        
        # Physics timing debug
        self.physics_step_count = 0
        self.physics_start_time = None
        self.last_physics_report_time = None
        
    def setup(self):
        """Set up keyboard listener, contact sensor, and physics callback"""
        self._appwindow = omni.appwindow.get_default_app_window()
        self._input = carb.input.acquire_input_interface()
        self._keyboard = self._appwindow.get_keyboard()
        self._sub_keyboard = self._input.subscribe_to_keyboard_events(self._keyboard, self._sub_keyboard_event)
        
        # Initialize spot after world reset
        self._spot.initialize()
        
        # Add contact sensors to Spot's feet instead of the block
        spot_foot_names = ["fl_foot", "fr_foot", "hl_foot", "hr_foot"]
        self._foot_sensors = []
        
        for i, foot_name in enumerate(spot_foot_names):
            foot_sensor = self._world.scene.add(
                ContactSensor(
                    prim_path=f"/World/Spot/{foot_name}/contact_sensor",
                    name=f"spot_foot_sensor_{i}",
                    min_threshold=0,
                    max_threshold=10000000,
                    radius=0.1,  # Small radius around the foot
                    translation=[0, 0, 0],  # At the foot center
                )
            )
            foot_sensor.add_raw_contact_data_to_frame()
            self._foot_sensors.append(foot_sensor)

        self.side_camera = load_robot_camera(
            camera_path="/World/Spot/body/side_cam",
            position=[0.0, 2.0, -0.25],  # side position
            rotation={"x": -90, "y": 0, "z": 180}, # side view, look how robot walks
            resolution=(448, 448),
            frequency=30
        )
        
        self._world.add_physics_callback("spot_controller", callback_fn=self.on_physics_step)
        
    def on_physics_step(self, step_size):
        """Physics callback to control robot and read contact sensor"""
        # Track physics timing
        import time
        current_time = time.time()
        
        if self.physics_start_time is None:
            self.physics_start_time = current_time
            self.last_physics_report_time = current_time
        
        self.physics_step_count += 1
        
        # Report physics rate every second
        if current_time - self.last_physics_report_time >= 1.0:
            elapsed = current_time - self.last_physics_report_time
            physics_rate = self.physics_step_count / (current_time - self.physics_start_time)
            recent_rate = (self.physics_step_count - (current_time - self.physics_start_time - 1) * physics_rate) / elapsed
            print(f"\n[PHYSICS] Step {self.physics_step_count}: Average rate = {physics_rate:.1f} Hz, Recent rate = {recent_rate:.1f} Hz, step_size = {step_size:.4f}s")
            self.last_physics_report_time = current_time
        
        if self.needs_reset:
            self._world.reset(True)
            self._spot.initialize()
            self.needs_reset = False
            return
        else:
            # Apply movement commands
            scaled_command = self._base_command * np.array([3.0, 0.5, 1.0])
            self._spot.forward(step_size, scaled_command)
            
            # Collect robot state data for socket broadcast
            robot_state = {
                "timestamp": self._world.current_time,
                "components": {}
            }
            
            # Read robot's main body pose
            position, orientation = self._spot.robot.get_world_pose()
            robot_state["main_body"] = {
                "position": position.tolist(),
                "orientation": orientation.tolist()
            }

            # Access the robot's USD prim and get individual body component transforms
            stage = omni.usd.get_context().get_stage()
            robot_prim_path = "/World/Spot"
            
            # Get transforms for individual body components
            body_components = [
                "body",
                "fl_hip", "fl_uleg", "fl_lleg", "fl_foot",
                "fr_hip", "fr_uleg", "fr_lleg", "fr_foot", 
                "hl_hip", "hl_uleg", "hl_lleg", "hl_foot",
                "hr_hip", "hr_uleg", "hr_lleg", "hr_foot"
            ]
            
            for component in body_components:
                component_path = f"{robot_prim_path}/{component}"
                component_prim = stage.GetPrimAtPath(component_path)
                
                if component_prim.IsValid():
                    # Get the world transform matrix using the correct Isaac Sim method
                    world_matrix = omni.usd.get_world_transform_matrix(component_prim)
                    
                    # Extract position and rotation from world transform
                    world_translation = world_matrix.ExtractTranslation()
                    world_rotation = world_matrix.ExtractRotationQuat()
                    
                    robot_state["components"][component] = {
                        "position": [world_translation[0], world_translation[1], world_translation[2]],
                        "rotation": [
                            world_rotation.GetReal(),
                            world_rotation.GetImaginary()[0],
                            world_rotation.GetImaginary()[1],
                            world_rotation.GetImaginary()[2]
                        ]
                    }

            # Read joint positions and velocities
            joint_states = self._spot.robot.get_joints_state()
            joint_positions = joint_states.positions
            joint_velocities = joint_states.velocities
            joint_efforts = self._spot.robot.get_measured_joint_efforts()
            joint_names = self._spot.robot.dof_names
            
            robot_state["joints"] = {
                "names": joint_names,
                "positions": joint_positions.tolist(),
                "velocities": joint_velocities.tolist(),
                "efforts": joint_efforts.tolist() if joint_efforts is not None else []
            }
            
            # Read linear and angular velocities
            linear_velocity = self._spot.robot.get_linear_velocity()
            angular_velocity = self._spot.robot.get_angular_velocity()
            robot_state["velocities"] = {
                "linear": linear_velocity.tolist(),
                "angular": angular_velocity.tolist()
            }
            
            # Add neural network activations if available
            if hasattr(self._spot.policy, 'get_activations'):
                activations = self._spot.policy.get_activations()
                if activations:
                    robot_state["neural_activations"] = {}
                    for layer_name, activation in activations.items():
                        robot_state["neural_activations"][layer_name] = {
                            "values": activation.tolist(),
                            "shape": list(activation.shape),
                            "mean": float(activation.mean()),
                            "std": float(activation.std())
                        }
            
            # print(linear_velocity, angular_velocity)
            
            # Broadcast robot state via socket
            self.broadcaster.send_robot_state(robot_state)
            
            # Track if any foot is touching the block
            block_contacted = False
            
            # Get and print contact sensor data from all feet
            for i, sensor in enumerate(self._foot_sensors):
                if sensor and sensor.is_valid():
                    contact_data = sensor.get_current_frame()
                    if contact_data and contact_data.get('contacts'):
                        # Check each contact to see if it's with the block
                        for contact in contact_data['contacts']:
                            if contact['body1'] == '/World/ContactBlock':
                                block_contacted = True
                                break
            
            # Change block color based on contact
            visual_material = self._block.get_applied_visual_material()
            if block_contacted:
                visual_material.set_color(np.array([1.0, 0.0, 0.0]))  # Red
            else:
                visual_material.set_color(np.array([0.5, 0.5, 0.8]))  # Light blue
        
    def run(self):
        """Main simulation loop"""
        print("\nSpot Contact Sensor Demo with Socket Broadcasting")
        print("=================================================")
        print(f"Broadcasting robot state on port {args.socket_port}")
        print("A blue block is placed in front of Spot.")
        print("The block has a contact sensor that will detect when Spot steps on it.")
        print("\nKeyboard Controls:")
        print("Arrow Keys/Numpad 8,2,4,6: Move forward/back/left/right")
        print("N/Numpad 7: Rotate left")
        print("M/Numpad 9: Rotate right")
        print("\nMove Spot onto the blue block to see contact data!\n")
        
        while simulation_app.is_running():
            self._world.step(render=True)
            if self._world.is_stopped():
                self.needs_reset = True
                
    def _sub_keyboard_event(self, event, *args, **kwargs):
        """Handle keyboard input events"""
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if event.input.name in self._input_keyboard_mapping:
                self._base_command += np.array(self._input_keyboard_mapping[event.input.name])
                
        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            if event.input.name in self._input_keyboard_mapping:
                self._base_command -= np.array(self._input_keyboard_mapping[event.input.name])
        return True
    
    def close(self):
        """Clean up resources"""
        self.broadcaster.close()


def main():
    """Main function"""
    physics_dt = 1 / 200.0
    render_dt = 1 / 60.0
    
    controller = SpotContactController(physics_dt=physics_dt, render_dt=render_dt)
    controller.setup()
    controller.run()
    controller.close()


if __name__ == "__main__":
    main()
    simulation_app.close()


