from typing import List, Optional, Union
import copy
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal


class BaseAgent(nn.Module, ABC):
    """
    Clean base class for all agents.

    Features:
    - Checkpointing
    - EMA support
    - Device and precision management
    - Abstract methods for get_action and get_value
    """

    def __init__(
        self,
        device: Union[str, torch.device] = "cuda:0",
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()

        self.device = torch.device(device) if isinstance(device, str) else device
        self.dtype = dtype
        self._ema_agent = None
        self._ema_decay = 0.999

    @abstractmethod
    def get_action(self, x: torch.Tensor) -> torch.Tensor:
        """Compute action from input."""
        pass

    @abstractmethod
    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        """Compute state-value from input."""
        pass

    def build_networks(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: list[int],
        activation: type[nn.Module] = nn.ELU,
        output_activation: Optional[nn.Module] = None,
    ) -> nn.Sequential:
        """
        Build a neural network with given specifications.

        Args:
            input_dim: Input dimension
            output_dim: Output dimension
            hidden_dims: List of hidden layer dimensions
            activation: Activation function for hidden layers
            output_activation: Activation function for output layer (optional)

        Returns:
            nn.Sequential network
        """
        layers = []

        # Build hidden layers
        for i, hidden_dim in enumerate(hidden_dims):
            layer_input_dim = input_dim if i == 0 else hidden_dims[i - 1]
            layers.extend([nn.Linear(layer_input_dim, hidden_dim), activation()])

        # Output layer
        final_input_dim = hidden_dims[-1] if hidden_dims else input_dim
        layers.append(nn.Linear(final_input_dim, output_dim))

        # Optional output activation
        if output_activation is not None:
            layers.append(output_activation())

        return nn.Sequential(*layers)

    def to_device(self, device: Union[str, torch.device]) -> "BaseAgent":
        """Move model to specified device."""
        self.device = torch.device(device) if isinstance(device, str) else device
        return self.to(self.device)

    def set_precision(self, dtype: torch.dtype) -> "BaseAgent":
        """Set the precision of the model."""
        self.dtype = dtype
        return self.to(dtype)

    def create_ema_agent(self, decay: float = 0.999) -> "BaseAgent":
        """Create EMA copy of the agent."""
        ema_agent = copy.deepcopy(self)
        ema_agent.eval()

        # Disable gradients for EMA agent
        for param in ema_agent.parameters():
            param.requires_grad = False

        # Store decay rate
        ema_agent._ema_decay = decay

        self._ema_agent = ema_agent
        return ema_agent

    def update_ema(self, decay: float | None = None):
        """Update EMA weights using torch.lerp."""
        if self._ema_agent is None:
            raise ValueError("EMA agent not created. Call create_ema_agent() first.")

        if decay is None:
            decay = getattr(self._ema_agent, "_ema_decay", 0.999)

        decay = float(decay) if decay is not None else 0.999

        with torch.no_grad():
            for ema_param, param in zip(
                self._ema_agent.parameters(), self.parameters()
            ):
                ema_param.data.lerp_(param.data, 1.0 - decay)

    @property
    def ema(self) -> Optional["BaseAgent"]:
        """Get the EMA agent."""
        return self._ema_agent

    def save_checkpoint(
        self,
        path: str,
        optimizer: Optional[torch.optim.Optimizer] = None,
        step: int = 0,
        **kwargs,
    ):
        """Save checkpoint."""
        checkpoint = {
            "model_state_dict": self.state_dict(),
            "step": step,
            **kwargs,
        }

        if optimizer is not None:
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()

        if self._ema_agent is not None:
            checkpoint["ema_model_state_dict"] = self._ema_agent.state_dict()

        torch.save(checkpoint, path)

    def load_checkpoint(
        self,
        path: str,
        optimizer: Optional[torch.optim.Optimizer] = None,
        load_ema: bool = False,
        strict: bool = True,
    ) -> dict:
        """Load checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)

        # Load model weights
        if load_ema and "ema_model_state_dict" in checkpoint:
            state_dict = checkpoint["ema_model_state_dict"]
        else:
            state_dict = checkpoint["model_state_dict"]

        self.load_state_dict(state_dict, strict=strict)

        # Load optimizer state
        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        return checkpoint



class MLPPPOAgent(BaseAgent):
    """
    MLP PPO Agent for low-level policy. 
    """

    def __init__(
        self,
        n_obs: int,
        n_act: int,
        actor_hidden_dims: List[int] = [512, 256, 128],
        critic_hidden_dims: List[int] = [512, 256, 128],
        activation: type[nn.Module] = nn.ELU,
        noise_std_type: str = "scalar",
        init_noise_std: float = 1.0,
        device: str = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__(device=device, dtype=dtype)

        self.n_obs = n_obs
        self.n_act = n_act
        self.noise_std_type = noise_std_type

        # Build networks using base class method
        self.actor = self.build_networks(
            input_dim=n_obs,
            output_dim=n_act,
            hidden_dims=actor_hidden_dims,
            activation=activation,
        )

        self.critic = self.build_networks(
            input_dim=n_obs,
            output_dim=1,
            hidden_dims=critic_hidden_dims,
            activation=activation,
        )

        # Initialize noise parameters
        if noise_std_type == "scalar":
            self.actor_std = nn.Parameter(init_noise_std * torch.ones(n_act))
        elif noise_std_type == "log":
            self.actor_std = nn.Parameter(torch.log(init_noise_std * torch.ones(n_act)))
        else:
            raise ValueError(f"Invalid noise_std_type: {noise_std_type}")

        Normal.set_default_validate_args(False)

        # Move to device and set precision
        self.to(self.device, self.dtype)

    def get_action(self, x: torch.Tensor) -> torch.Tensor:
        """Compute action from input."""
        return self.actor(x)

    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        """Compute state-value from input."""
        return self.critic(x)

    def get_action_and_value(
        self, x: torch.Tensor, action: torch.Tensor | None = None
    ) -> tuple:
        """Compute action, log-prob, entropy, and value."""
        action_mean = self.actor(x)
        action_std = self.actor_std.expand_as(action_mean)

        if self.noise_std_type == "log":
            action_std = torch.clamp(action_std, -20.0, 2.0)
            action_std = torch.exp(action_std)
        elif self.noise_std_type == "scalar":
            action_std = F.softplus(action_std)

        dist = Normal(action_mean, action_std)
        if action is None:
            action = dist.sample()

        return (
            action,
            dist.log_prob(action).sum(dim=-1),
            dist.entropy().sum(dim=-1),
            self.critic(x),
            action_mean,
            action_std,
        )

    def forward(self, x):
        return self.get_action(x)


class MLPFastTD3Actor(nn.Module):
    '''
    Fast TD3 Actor for low-level policy. 
    '''
    def __init__(
        self,
        n_obs: int,
        n_act: int,
        num_envs: int,
        init_scale: float = 0.01,
        hidden_dims: list[int] = [512, 256, 128],
        activation: type[nn.Module] = nn.ReLU,
        output_activation: type[nn.Module] = nn.Tanh,
        std_min: float = 0.05,
        std_max: float = 0.8,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__()
        self.n_act = n_act
        actor_layers = []
        for i in range(len(hidden_dims)):
            if i == 0:
                actor_layers.append(nn.Linear(n_obs, hidden_dims[i]))
            else:
                actor_layers.append(nn.Linear(hidden_dims[i - 1], hidden_dims[i]))
            actor_layers.append(activation())
        self.net = nn.Sequential(*actor_layers)
        self.net.to(device)
        self.fc_mu = nn.Sequential(
            nn.Linear(hidden_dims[-1], n_act),
            output_activation(),
        )
        nn.init.normal_(self.fc_mu[0].weight, 0.0, init_scale)
        nn.init.constant_(self.fc_mu[0].bias, 0.0)
        self.fc_mu.to(device)
        self.device = device

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x)
        x = self.fc_mu(x)
        return x