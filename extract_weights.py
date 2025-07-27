from model import MLPFastTD3Actor, MLPPPOAgent
import torch 


class ModelWithHooks:
    """Wrapper class to add activation hooks to a model"""
    def __init__(self, model):
        self.model = model
        self.activations = {}
        self.hooks = []
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward hooks on all linear layers in the actor network"""
        def get_activation(name):
            def hook(module, input, output):
                self.activations[name] = output.detach().cpu().numpy()
            return hook
        
        # Register hooks for actor network layers
        if hasattr(self.model, 'actor'):
            for i, (layer_name, module) in enumerate(self.model.actor.named_children()):
                if isinstance(module, torch.nn.Linear):
                    hook = module.register_forward_hook(get_activation(f'actor_hidden_{i}'))
                    self.hooks.append(hook)
    
    def remove_hooks(self):
        """Remove all registered hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def get_activations(self):
        """Get the last recorded activations"""
        return self.activations
    
    def clear_activations(self):
        """Clear stored activations"""
        self.activations = {}
    
    def __call__(self, *args, **kwargs):
        """Forward pass through the model"""
        return self.model(*args, **kwargs)
    
    def __getattr__(self, name):
        """Delegate attribute access to the wrapped model"""
        return getattr(self.model, name)


def extract_pytorch_model_from_jit(jit_model_path, with_hooks=False):
    """Extract PyTorch model with trained weights from JIT model"""
    
    # Load JIT model
    jit_model = torch.jit.load(jit_model_path)
    
    # Create PyTorch model (infer dimensions from state dict)
    state_dict = jit_model.state_dict()
    
    # Create model
    # model = MLPFastTD3Actor(n_obs=48, n_act=12, num_envs=1, init_scale=0.01)
    model = MLPPPOAgent(n_obs=48, n_act=12)
    
    # Clean state dict
    new_jit_dict = {}
    for k, v in state_dict.items():
        new_jit_dict[k.replace('model.', '')] = v
    
    # Load weights
    model.load_state_dict(new_jit_dict, strict=False)
    
    # Wrap with hooks if requested
    if with_hooks:
        model = ModelWithHooks(model)
    
    return model, jit_model


def test_model(model, jit_model):

    with torch.no_grad():
        obs = torch.randn(1, 48)

        # Test for fast_td3 model
        # normalized_obs = jit_model.normalizer(obs)
        # action = model(normalized_obs)
        # action = torch.clamp(action, -1, 1) * jit_model.action_bounds
        # print(action, jit_model.action_bounds)

        # Test for PPO model with hooks
        if isinstance(model, ModelWithHooks):
            # Clear previous activations
            model.clear_activations()
            
            # Forward pass - this will trigger the hooks
            action = model(obs)
            print("Action output:", action)
            
            # Get and print hidden layer activations
            activations = model.get_activations()
            print("\n=== Hidden Layer Activations ===")
            for layer_name, activation in activations.items():
                print(f"{layer_name}: shape={activation.shape}, mean={activation.mean():.4f}, std={activation.std():.4f}")
                print(f"  First 10 values: {activation[0][:10]}")
        else:
            action = model(obs)
            print("Action output:", action)

        # jit model direct output
        action = jit_model(obs)
        print("\nJIT model output:", action)


if __name__ == "__main__":
    # model, jit_model = extract_pytorch_model_from_jit("./assets/spot_policy_ftd3_hop.pt", with_hooks=True)
    # model, jit_model = extract_pytorch_model_from_jit("./assets/spot_policy_custom_rslrl.pt", with_hooks=True)
    model, jit_model = extract_pytorch_model_from_jit("./official_spot_assets/spot_policy.pt", with_hooks=True)
    test_model(model, jit_model)