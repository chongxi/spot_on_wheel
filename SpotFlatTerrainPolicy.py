# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

import io
from typing import Optional
from abc import ABC, abstractmethod

import carb
import numpy as np
import omni
import omni.kit.commands
import torch
import yaml
from isaacsim.core.utils.rotations import quat_to_rot_matrix
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.core.prims import SingleArticulation
from isaacsim.core.utils.prims import define_prim, get_prim_at_path
from isaacsim.storage.native import get_assets_root_path


class BaseController(ABC):
    """Base controller class"""

    def __init__(self, name: str) -> None:
        self._name = name

    @abstractmethod
    def forward(self, *args, **kwargs) -> ArticulationAction:
        """A controller should take inputs and returns an ArticulationAction"""
        raise NotImplementedError

    def reset(self) -> None:
        """Resets state of the controller."""
        return


def parse_env_config(config_path: str) -> dict:
    """Parse environment configuration from YAML file"""
    file_content = omni.client.read_file(config_path)[2]
    config_string = file_content.decode('utf-8')
    return yaml.safe_load(config_string)


def get_articulation_props(data: dict) -> dict:
    """Gets the articulation properties from the environment configuration data."""
    return data.get("scene", {}).get("robot", {}).get("spawn", {}).get("articulation_props", {})


def get_physics_properties(data: dict) -> tuple:
    """Gets the physics properties from the environment configuration data."""
    decimation = data.get("decimation", 4)
    dt = data.get("sim", {}).get("dt", 0.005)
    render_interval = data.get("sim", {}).get("render_interval", 2)
    return decimation, dt, render_interval


def get_robot_joint_properties(data: dict, dof_names: list) -> tuple:
    """Gets robot joint properties from the environment configuration data."""
    # Default values for Spot
    num_joints = len(dof_names)
    max_effort = np.array([40.0] * num_joints)
    max_vel = np.array([20.0] * num_joints)
    stiffness = np.array([400.0] * num_joints)
    damping = np.array([40.0] * num_joints)
    default_pos = np.array([0.0, 0.9, -1.8] * 4)  # 12 joints for Spot
    default_vel = np.zeros(num_joints)
    
    # Try to get from config if available
    robot_config = data.get("scene", {}).get("robot", {})
    if "joints" in robot_config:
        joints_config = robot_config["joints"]
        if "max_effort" in joints_config:
            max_effort = np.array(joints_config["max_effort"])
        if "max_velocity" in joints_config:
            max_vel = np.array(joints_config["max_velocity"])
        if "stiffness" in joints_config:
            stiffness = np.array(joints_config["stiffness"])
        if "damping" in joints_config:
            damping = np.array(joints_config["damping"])
        if "default_position" in joints_config:
            default_pos = np.array(joints_config["default_position"])
    
    return max_effort, max_vel, stiffness, damping, default_pos, default_vel


class PolicyController(BaseController):
    """
    A controller that loads and executes a policy from a file.

    Args:
        name (str): The name of the controller.
        prim_path (str): The path to the prim in the stage.
        root_path (Optional[str], None): The path to the articulation root of the robot
        usd_path (Optional[str], optional): The path to the USD file. Defaults to None.
        position (Optional[np.ndarray], optional): The initial position of the robot. Defaults to None.
        orientation (Optional[np.ndarray], optional): The initial orientation of the robot. Defaults to None.

    Attributes:
        robot (SingleArticulation): The robot articulation.
    """

    def __init__(
        self,
        name: str,
        prim_path: str,
        root_path: Optional[str] = None,
        usd_path: Optional[str] = None,
        position: Optional[np.ndarray] = None,
        orientation: Optional[np.ndarray] = None,
    ) -> None:
        super().__init__(name)
        
        prim = get_prim_at_path(prim_path)

        if not prim.IsValid():
            prim = define_prim(prim_path, "Xform")
            if usd_path:
                prim.GetReferences().AddReference(usd_path)
            else:
                carb.log_error("unable to add robot usd, usd_path not provided")

        if root_path == None:
            self.robot = SingleArticulation(prim_path=prim_path, name=name, position=position, orientation=orientation)
        else:
            self.robot = SingleArticulation(prim_path=root_path, name=name, position=position, orientation=orientation)

    def load_policy(self, policy_file_path, policy_env_path) -> None:
        """
        Loads a policy from a file.

        Args:
            policy_file_path (str): The path to the policy file.
            policy_env_path (str): The path to the environment configuration file.
        """
        file_content = omni.client.read_file(policy_file_path)[2]
        file = io.BytesIO(memoryview(file_content).tobytes())
        self.policy = torch.jit.load(file)
        self.policy_env_params = parse_env_config(policy_env_path)

        self._decimation, self._dt, self.render_interval = get_physics_properties(self.policy_env_params)

    def initialize(
        self,
        physics_sim_view: omni.physics.tensors.SimulationView = None,
        effort_modes: str = "force",
        control_mode: str = "position",
        set_gains: bool = True,
        set_limits: bool = True,
        set_articulation_props: bool = True,
    ) -> None:
        """
        Initializes the robot and sets up the controller.

        Args:
            physics_sim_view (optional): The physics simulation view.
            effort_modes (str, optional): The effort modes. Defaults to "force".
            control_mode (str, optional): The control mode. Defaults to "position".
            set_gains (bool, optional): Whether to set the joint gains. Defaults to True.
            set_limits (bool, optional): Whether to set the limits. Defaults to True.
            set_articulation_props (bool, optional): Whether to set the articulation properties. Defaults to True.
        """
        self.robot.initialize(physics_sim_view=physics_sim_view)
        self.robot.get_articulation_controller().set_effort_modes(effort_modes)
        self.robot.get_articulation_controller().switch_control_mode(control_mode)
        max_effort, max_vel, stiffness, damping, self.default_pos, self.default_vel = get_robot_joint_properties(
            self.policy_env_params, self.robot.dof_names
        )
        if set_gains:
            self.robot._articulation_view.set_gains(stiffness, damping)
        if set_limits:
            self.robot._articulation_view.set_max_efforts(max_effort)
            self.robot._articulation_view.set_max_joint_velocities(max_vel)
        if set_articulation_props:
            self._set_articulation_props()

    def _set_articulation_props(self) -> None:
        """
        Sets the articulation root properties from the policy environment parameters.
        """
        articulation_prop = get_articulation_props(self.policy_env_params)

        solver_position_iteration_count = articulation_prop.get("solver_position_iteration_count")
        solver_velocity_iteration_count = articulation_prop.get("solver_velocity_iteration_count")
        stabilization_threshold = articulation_prop.get("stabilization_threshold")
        enabled_self_collisions = articulation_prop.get("enabled_self_collisions")
        sleep_threshold = articulation_prop.get("sleep_threshold")

        if solver_position_iteration_count not in [None, float("inf")]:
            self.robot.set_solver_position_iteration_count(solver_position_iteration_count)
        if solver_velocity_iteration_count not in [None, float("inf")]:
            self.robot.set_solver_velocity_iteration_count(solver_velocity_iteration_count)
        if stabilization_threshold not in [None, float("inf")]:
            self.robot.set_stabilization_threshold(stabilization_threshold)
        if isinstance(enabled_self_collisions, bool):
            self.robot.set_enabled_self_collisions(enabled_self_collisions)
        if sleep_threshold not in [None, float("inf")]:
            self.robot.set_sleep_threshold(sleep_threshold)

    def _compute_action(self, obs: np.ndarray) -> np.ndarray:
        """
        Computes the action from the observation using the loaded policy.

        Args:
            obs (np.ndarray): The observation.

        Returns:
            np.ndarray: The action.
        """
        with torch.no_grad():
            obs = torch.from_numpy(obs).view(1, -1).float()
            action = self.policy(obs).detach().view(-1).numpy()
        return action

    def _compute_observation(self) -> NotImplementedError:
        """
        Computes the observation. Not implemented.
        """

        raise NotImplementedError(
            "Compute observation need to be implemented, expects np.ndarray in the structure specified by env yaml"
        )

    def forward(self) -> NotImplementedError:
        """
        Forwards the controller. Not implemented.
        """
        raise NotImplementedError(
            "Forward needs to be implemented to compute and apply robot control from observations"
        )

    def post_reset(self) -> None:
        """
        Called after the controller is reset.
        """
        self.robot.post_reset()


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

        action = ArticulationAction(joint_positions=self.default_pos + (self.action * self._action_scale))
        self.robot.apply_action(action)

        self._policy_counter += 1