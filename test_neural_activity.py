#!/usr/bin/env python3
"""Test script to verify neural activity extraction works correctly"""

from extract_weights import extract_pytorch_model_from_jit
import torch
import numpy as np

def test_neural_activity():
    print("Testing neural activity extraction...")
    
    # Load model with hooks
    model, jit_model = extract_pytorch_model_from_jit("./assets/rslrl_ppo_7B_policy.pt", with_hooks=True)
    
    # Create test observation
    obs = torch.randn(1, 48)
    
    print("\nRunning forward pass through model...")
    with torch.no_grad():
        # Clear any previous activations
        model.clear_activations()
        
        # Forward pass
        action = model(obs)
        print(f"Action output shape: {action.shape}")
        print(f"Action values: {action[0][:5]}...")  # Show first 5 action values
        
        # Get activations
        activations = model.get_activations()
        
        print("\n=== Hidden Layer Activations ===")
        for layer_name, activation in sorted(activations.items()):
            print(f"\n{layer_name}:")
            print(f"  Shape: {activation.shape}")
            print(f"  Mean: {activation.mean():.4f}")
            print(f"  Std: {activation.std():.4f}")
            print(f"  Min: {activation.min():.4f}")
            print(f"  Max: {activation.max():.4f}")
            print(f"  First 10 values: {activation[0][:10]}")
            
            # Check for dead neurons (always zero)
            zero_count = np.sum(activation == 0)
            total_neurons = activation.size
            print(f"  Zero activations: {zero_count}/{total_neurons} ({zero_count/total_neurons*100:.1f}%)")
    
    print("\nTest completed successfully!")

if __name__ == "__main__":
    test_neural_activity()