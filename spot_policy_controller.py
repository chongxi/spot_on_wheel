import io
import torch
from isaaclab.utils.math import matrix_from_quat
from isaacsim.core.utils.rotations import quat_to_rot_matrix
from isaacsim.storage.native import get_assets_root_path
from isaacsim.core.prims import SingleArticulation
from isaacsim.core.utils.stage import add_reference_to_stage
from typing import Optional
import numpy as np
from isaacsim.core.utils.types import ArticulationAction


class SpotPolicyController:
    """
    Minimal Spot policy wrapper. Loads a TorchScript policy and returns actions given observation and command.
    Now fully vectorized: all methods accept and return torch tensors for n environments.
    """
    def __init__(
        self,
        prim_path: str,
        name: str = "spot",
        policy_file_path: str = "./assets/spot_policy_ftd3_hop.pt",
        usd_path: Optional[str] = None,
        position: Optional[np.ndarray] = None,
        orientation: Optional[np.ndarray] = None,
    ) -> None:
        """
        Initialize robot and load RL policy.

        Args:
            prim_path (str) -- prim path of the robot on the stage
            name (str) -- name of the quadruped
            usd_path (str) -- robot usd filepath in the directory
            position (np.ndarray) -- position of the robot
            orientation (np.ndarray) -- orientation of the robot

        """
        assets_root_path = get_assets_root_path()
        if usd_path == None:
            usd_path = assets_root_path + "/Isaac/Robots/BostonDynamics/spot/spot.usd"
        add_reference_to_stage(usd_path=usd_path, prim_path=prim_path)

        self.robot = SingleArticulation(prim_path=prim_path, name=name, position=position, orientation=orientation)

        self.load_policy(policy_file_path)
        self._action_scale = 15
        self._previous_action = np.zeros(12)
        self._policy_counter = 0
        
        # Add default joint positions (you may need to adjust these values)
        self.default_pos = np.zeros(12)  # or set to actual default pose values


    def load_policy(self, policy_file_path: str):
        with open(policy_file_path, "rb") as f:
            self.policy = torch.jit.load(io.BytesIO(f.read()))


    def __call__(self, observation: torch.Tensor) -> torch.Tensor:
        """
        Run the policy on the given batch of observations.
        Args:
            observation (torch.Tensor): [num_envs, obs_dim] observation input.
        Returns:
            torch.Tensor: [num_envs, action_dim] policy output (actions).
        """
        # Ensure policy is on the same device as observation
        if next(self.policy.parameters()).device != observation.device:
            self.policy = self.policy.to(observation.device)
        with torch.no_grad():
            obs_tensor = observation.float()
            action = self.policy(obs_tensor).detach()
        return action

    def compute_observation(self, command):
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

        # Add safety checks for dimensions
        if lin_vel_I is None or len(lin_vel_I) == 0:
            lin_vel_I = np.zeros(3)
        if ang_vel_I is None or len(ang_vel_I) == 0:
            ang_vel_I = np.zeros(3)
        if q_IB is None or len(q_IB) == 0:
            q_IB = np.array([1.0, 0.0, 0.0, 0.0])  # identity quaternion

        # Ensure arrays are proper shape
        lin_vel_I = np.array(lin_vel_I).flatten()
        ang_vel_I = np.array(ang_vel_I).flatten()
        q_IB = np.array(q_IB).flatten()
        
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
        
        # Safety checks for joint data
        if current_joint_pos is None or len(current_joint_pos) == 0:
            current_joint_pos = np.zeros(12)
        if current_joint_vel is None or len(current_joint_vel) == 0:
            current_joint_vel = np.zeros(12)
            
        obs[12:24] = current_joint_pos - self.default_pos
        obs[24:36] = current_joint_vel
        # Previous Action
        obs[36:48] = self._previous_action

        return obs

    def compute_command(self, goal, state=None) -> torch.Tensor:
        """
        Compute the command vector (e.g., navigation command) for the policy.
        Args:
            goal: The target or goal (type as needed).
            state: Optionally, the current state (type as needed).
        Returns:
            torch.Tensor: The command vector for the policy.
        """
        # TODO: Implement this method for your environment
        raise NotImplementedError(
            "Implement compute_command for your Spot environment."
        )

    def compute_action(self, obs: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            action = self(obs)
        return action

    def get_action(
        self,
        lin_vel_I,
        ang_vel_I,
        q_IB,
        command,
        previous_action,
        default_pos,
        joint_pos,
        joint_vel,
    ) -> torch.Tensor:
        """
        Compute the observation from the robot state and command, then run the policy to get the action (vectorized).
        All arguments are batched torch tensors.
        Returns:
            actions: [num_envs, action_dim] torch.Tensor
        """
        obs = self.compute_observation(
            lin_vel_I,
            ang_vel_I,
            q_IB,
            command,
            previous_action,
            default_pos,
            joint_pos,
            joint_vel,
        )
        actions = self(obs)
        return actions


    def forward(self, dt, command):
        """
        Compute the desired torques and apply them to the articulation

        Argument:
        dt (float) -- Timestep update in the world.
        command (np.ndarray) -- the robot command (v_x, v_y, w_z)

        """
        if self._policy_counter % 10 == 0:
            obs = self.compute_observation(command)
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
            action_tensor = self(obs_tensor)
            self.action = action_tensor.squeeze(0).detach().cpu().numpy()  # Remove batch dimension
            self._previous_action = self.action.copy()

        # Import and apply action
        joint_positions = self.default_pos + (self.action * self._action_scale)
        action = ArticulationAction(joint_positions=joint_positions)
        try:
            self.robot.apply_action(action)
        except Exception as e:
            print(f"Error applying action: {e}")
        self._policy_counter += 1



class SpotRoughPolicyController(SpotPolicyController):
    def __init__(self, policy_file_path: str):
        super().__init__(policy_file_path)

    def compute_observation(
        self,
        base_lin_vel,
        base_ang_vel,
        projected_gravity,
        command,
        previous_action,
        default_pos,
        joint_pos,
        joint_vel,
    ) -> torch.Tensor:
        """
        Compute the observation vector for the policy for all environments (vectorized).
        Args:
            lin_vel_I: [num_envs, 3] torch.Tensor
            ang_vel_I: [num_envs, 3] torch.Tensor
            q_IB: [num_envs, 4] torch.Tensor (quaternion)
            command: [num_envs, 3] torch.Tensor
            previous_action: [num_envs, 12] torch.Tensor
            default_pos: [num_envs, 12] torch.Tensor
            joint_pos: [num_envs, 12] torch.Tensor
            joint_vel: [num_envs, 12] torch.Tensor
        Returns:
            obs: [num_envs, 48] torch.Tensor
        """
        obs = torch.zeros(
            (base_lin_vel.shape[0], 48), device=base_lin_vel.device, dtype=base_lin_vel.dtype
        )
        obs[:, 0:3] = base_lin_vel
        obs[:, 3:6] = base_ang_vel
        obs[:, 6:9] = projected_gravity
        obs[:, 9:12] = command
        obs[:, 12:24] = joint_pos - default_pos
        obs[:, 24:36] = joint_vel
        obs[:, 36:48] = previous_action
        return obs

    def compute_command(self, goal, state=None) -> torch.Tensor:
        """
        Compute the command vector (e.g., navigation command) for the policy.
        Args:
            goal: The target or goal (type as needed).
            state: Optionally, the current state (type as needed).
        Returns:
            torch.Tensor: The command vector for the policy.
        """
        # TODO: Implement this method for your environment
        raise NotImplementedError(
            "Implement compute_command for your Spot environment."
        )

    def get_action(
        self,
        base_lin_vel,
        base_ang_vel,
        projected_gravity,
        command,
        previous_action,
        default_pos,
        joint_pos,
        joint_vel,
    ) -> torch.Tensor:
        """
        Compute the observation from the robot state and command, then run the policy to get the action (vectorized).
        All arguments are batched torch tensors.
        Returns:
            actions: [num_envs, action_dim] torch.Tensor
        """
        obs = self.compute_observation(
            base_lin_vel,
            base_ang_vel,
            projected_gravity,
            command,
            previous_action,
            default_pos,
            joint_pos,
            joint_vel,
        )
        actions = self(obs)
        return actions

class SpotRoughWithHeightPolicyController(SpotPolicyController):
    def __init__(self, policy_file_path: str):
        super().__init__(policy_file_path)

    def compute_observation(
        self,
        base_lin_vel,
        base_ang_vel,
        projected_gravity,
        command,
        previous_action,
        default_pos,
        joint_pos,
        joint_vel,
        height_obs,
    ) -> torch.Tensor:
        """
        Compute the observation vector for the policy for all environments (vectorized).
        Args:
            lin_vel_I: [num_envs, 3] torch.Tensor
            ang_vel_I: [num_envs, 3] torch.Tensor
            q_IB: [num_envs, 4] torch.Tensor (quaternion)
            command: [num_envs, 3] torch.Tensor
            previous_action: [num_envs, 12] torch.Tensor
            default_pos: [num_envs, 12] torch.Tensor
            joint_pos: [num_envs, 12] torch.Tensor
            joint_vel: [num_envs, 12] torch.Tensor
        Returns:
            obs: [num_envs, 48] torch.Tensor
        """
        obs = torch.zeros(
            (base_lin_vel.shape[0], 57), device=base_lin_vel.device, dtype=base_lin_vel.dtype
        )
        obs[:, 0:3] = base_lin_vel
        obs[:, 3:6] = base_ang_vel
        obs[:, 6:9] = projected_gravity
        obs[:, 9:12] = command
        obs[:, 12:24] = joint_pos - default_pos
        obs[:, 24:36] = joint_vel
        obs[:, 36:48] = previous_action
        obs[:, 48:57] = height_obs
        return obs

    def compute_command(self, goal, state=None) -> torch.Tensor:
        """
        Compute the command vector (e.g., navigation command) for the policy.
        Args:
            goal: The target or goal (type as needed).
            state: Optionally, the current state (type as needed).
        Returns:
            torch.Tensor: The command vector for the policy.
        """
        # TODO: Implement this method for your environment
        raise NotImplementedError(
            "Implement compute_command for your Spot environment."
        )

    def get_action(
        self,
        base_lin_vel,
        base_ang_vel,
        projected_gravity,
        command,
        previous_action,
        default_pos,
        joint_pos,
        joint_vel,
        height_obs,
    ) -> torch.Tensor:
        """
        Compute the observation from the robot state and command, then run the policy to get the action (vectorized).
        All arguments are batched torch tensors.
        Returns:
            actions: [num_envs, action_dim] torch.Tensor
        """
        obs = self.compute_observation(
            base_lin_vel,
            base_ang_vel,
            projected_gravity,
            command,
            previous_action,
            default_pos,
            joint_pos,
            joint_vel,
            height_obs,
        )
        actions = self(obs)
        return actions


if __name__ == "__main__":
    # Simple test for SpotPolicyController
    import numpy as np

    class DummyRobot:
        def get_linear_velocity(self):
            return np.array([1.0, 0.0, 0.0])

        def get_angular_velocity(self):
            return np.array([0.0, 0.0, 0.1])

        def get_world_pose(self):
            return np.array([0.0, 0.0, 0.5]), np.array(
                [1.0, 0.0, 0.0, 0.0]
            )  # pos, quat

        def get_joint_positions(self):
            return np.ones(12)

        def get_joint_velocities(self):
            return np.zeros(12)

    # Dummy data
    dummy_policy_path = "/home/user/cognitiverl/source/cognitiverl/cognitiverl/tasks/direct/cognitiverl/spot_policy.pt"  # Absolute path
    dummy_command = np.array([0.5, 0.0, 0.1])
    dummy_previous_action = np.zeros(12)
    dummy_default_pos = np.ones(12)
    robot = DummyRobot()

    # Test compute_observation and policy call
    try:
        controller = SpotPolicyController(dummy_policy_path)
        obs = controller.compute_observation(
            torch.tensor(robot.get_linear_velocity()),
            torch.tensor(robot.get_angular_velocity()),
            torch.tensor(robot.get_world_pose()[1]),
            torch.tensor(dummy_command),
            torch.tensor(dummy_previous_action),
            torch.tensor(dummy_default_pos),
            torch.tensor(robot.get_joint_positions()),
            torch.tensor(robot.get_joint_velocities()),
        )
        print("Observation vector:", obs)
        print("Observation vector shape:", obs.shape)
        # Now use the computed observation as input to the policy
        action = controller(obs)
        print("Policy output:", action)
    except Exception as e:
        print("[Test] compute_observation or policy call failed:", e)
