# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

from typing import Optional

import numpy as np
import torch
import omni
import omni.kit.commands
from isaacsim.core.utils.rotations import quat_to_rot_matrix
from isaacsim.core.utils.types import ArticulationAction
# from isaacsim.robot.policy.examples.controllers import PolicyController
from isaac_policy_controller.policy_controller import PolicyController
from isaacsim.storage.native import get_assets_root_path
from gait_patterns import walk_gait_spot, trot_gait_spot, gallop_gait_spot


class SpotFlatTerrainPolicy(PolicyController):
    """The Spot quadruped"""

    def __init__(
        self,
        prim_path: str,
        root_path: Optional[str] = None,
        name: str = "spot",
        usd_path: Optional[str] = None,
        position: Optional[np.ndarray] = None,
        orientation: Optional[np.ndarray] = None,
    ) -> None:
        """
        Initialize robot and load RL policy.

        Args:
            prim_path (str) -- prim path of the robot on the stage
            root_path (Optional[str]): The path to the articulation root of the robot
            name (str) -- name of the quadruped
            usd_path (str) -- robot usd filepath in the directory
            position (np.ndarray) -- position of the robot
            orientation (np.ndarray) -- orientation of the robot

        """
        assets_root_path = get_assets_root_path()
        if usd_path == None:
            usd_path = assets_root_path + "/Isaac/Robots/BostonDynamics/spot/spot.usd"

        super().__init__(name, prim_path, root_path, usd_path, position, orientation)

        self.load_policy(
            assets_root_path + "/Isaac/Samples/Policies/Spot_Policies/spot_policy.pt",
            assets_root_path + "/Isaac/Samples/Policies/Spot_Policies/spot_env.yaml",
        )
        self._action_scale = 0.2
        self._previous_action = np.zeros(12)
        self._policy_counter = 0
        self._decimation = 10  # Decimation factor for policy updates
        # Access the articulation controller to modify joint properties
        self.ctr = self.robot.get_articulation_controller()
        
        # # Define joint indices based on the actual order
        # Order: fl_hx, fr_hx, hl_hx, hr_hx, fl_hy, fr_hy, hl_hy, hr_hy, fl_kn, fr_kn, hl_kn, hr_kn
        hip_x_indices = [0, 1, 2, 3]      # hx joints
        hip_y_indices = [4, 5, 6, 7]      # hy joints
        knee_indices = [8, 9, 10, 11]     # kn joints
        
        # # Set max efforts (torque limits in Nm)
        self.max_efforts = np.zeros(12)
        self.max_efforts[hip_x_indices] = 150.0   # Hip X-axis torque limit (was 45.0)
        self.max_efforts[hip_y_indices] = 150.0   # Hip Y-axis torque limit (was 45.0)
        self.max_efforts[knee_indices] = 150.0   # Knee torque limit (was 115.0)
        # ctr.set_max_efforts(max_efforts)
        
        # # Set PD gains (stiffness and damping)
        self.stiffnesses = np.zeros(12)
        self.stiffnesses[hip_x_indices] = 100.0   # Hip X stiffness (was 60.0)
        self.stiffnesses[hip_y_indices] = 100.0   # Hip Y stiffness (was 60.0)
        self.stiffnesses[knee_indices] = 100.0   # Knee stiffness (was 60.0)
        
        self.dampings = np.zeros(12)
        self.dampings[hip_x_indices] = 3.0      # Hip X damping (was 1.5)
        self.dampings[hip_y_indices] = 3.0      # Hip Y damping (was 1.5)
        self.dampings[knee_indices] = 3.0       # Knee damping (was 1.5)
        
        # ctr.set_gains(stiffnesses, dampings)
        self._cpg_nsteps = 180  # Number of steps in the CPG cycle
        # self._cpg_output_matrix, _ = walk_gait_spot(
        self._cpg_output_matrix, _ = trot_gait_spot(
            n=self._cpg_nsteps,
            bone_length_uleg=0.67/2,
            bone_length_lleg=0.67/2,
            ground_z=-0.67
        )

        # Reorder from leg-by-leg to joint-type grouping
        # From: fl_hx(0), fl_hy(1), fl_kn(2), fr_hx(3), fr_hy(4), fr_kn(5), hl_hx(6), hl_hy(7), hl_kn(8), hr_hx(9), hr_hy(10), hr_kn(11)
        # To:   fl_hx(0), fr_hx(1), hl_hx(2), hr_hx(3), fl_hy(4), fr_hy(5), hl_hy(6), hr_hy(7), fl_kn(8), fr_kn(9), hl_kn(10), hr_kn(11)
        reorder_indices = [0, 3, 6,  9,  # all hx joints: fl_hx, fr_hx, hl_hx, hr_hx
                           1, 4, 7, 10,  # all hy joints: fl_hy, fr_hy, hl_hy, hr_hy
                           2, 5, 8, 11]  # all kn joints: fl_kn, fr_kn, hl_kn, hr_kn
        self._cpg_output_matrix = self._cpg_output_matrix[reorder_indices, :]

    def _compute_observation(self, command):
        """
        Compute the observation vector for the policy

        Argument:
        command (np.ndarray) -- the robot command (v_x, v_y, w_z)

        Returns:
        np.ndarray -- The observation vector.

        """
        lin_vel_I = self.robot.get_linear_velocity()
        ang_vel_I = self.robot.get_angular_velocity()
        pos_IB, q_IB = self.robot.get_world_pose()

        R_IB = quat_to_rot_matrix(q_IB)
        R_BI = R_IB.transpose()
        lin_vel_b = np.matmul(R_BI, lin_vel_I)
        ang_vel_b = np.matmul(R_BI, ang_vel_I)
        gravity_b = np.matmul(R_BI, np.array([0.0, 0.0, -1.0]))

        obs = np.zeros(48)
        # Base lin vel
        obs[:3] = lin_vel_b
        # Base ang vel
        obs[3:6] = ang_vel_b
        # Gravity
        obs[6:9] = gravity_b
        # Command
        obs[9:12] = command
        # Joint states
        current_joint_pos = self.robot.get_joint_positions()
        current_joint_vel = self.robot.get_joint_velocities()
        obs[12:24] = current_joint_pos - self.default_pos
        obs[24:36] = current_joint_vel
        # Previous Action
        obs[36:48] = self._previous_action

        return obs

    def forward(self, dt, command):
        """
        Compute the desired torques and apply them to the articulation

        Argument:
        dt (float) -- Timestep update in the world.
        command (np.ndarray) -- the robot command (v_x, v_y, w_z)

        """
        if self._policy_counter % self._decimation == 0:
            obs = self._compute_observation(command)
            self.action = self._compute_action(obs)
            self._previous_action = self.action.copy()
        self._current_action = self.default_pos + (self.action * self._action_scale)
        # self._current_action = np.array([ 0.2,  -0.2, 0.2,  -0.2,  
        #                                   0.8,  0.8,  0.8,  0.8, 
        #                                   -1.2, -1.36, -1.2, -1.36])

        # Get action from CPG
        #     self.cpg_action = self._cpg_output_matrix[:, self._policy_counter%self._cpg_nsteps].ravel().numpy()
        # self._current_action = self.cpg_action

        self.ctr.set_max_efforts(self.max_efforts)
        self.ctr.set_gains(self.stiffnesses, self.dampings)


        # print(f"Action at step {self._policy_counter}: {self._current_action}")
        print(np.array2string(self._current_action, 
                              formatter={'float_kind':lambda x: f"{x:.3f}"}))

        action = ArticulationAction(joint_positions=self._current_action)
        self.robot.apply_action(action)

        self._policy_counter += 1
