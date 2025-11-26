#!/usr/bin/env python3
"""
Quick sanity check for HumanoidUNet architecture
"""
import torch
from src.model.humanoid_unet import HumanoidUNet

def test_humanoid_unet():
    print("="*80)
    print("Testing HumanoidUNet Architecture")
    print("="*80)

    # Create model with config matching train_humanoid.yaml
    model = HumanoidUNet(
        embedded_dim=67,
        latent_dim=8,
        condition_dim=67,
        time_emb_dim=128,
        output_dim=67,
        hidden_dims=[256, 512, 1024, 1024, 512, 256],
        use_input_embeddings=True,
        input_emb_dim=128,
        dropout=0.1
    )

    print("\n✅ Model created successfully!")
    print(model)

    # Get architecture info
    info = model.get_architecture_info()
    print("\n📊 Architecture Details:")
    print(f"   Model type: {info['model_type']}")
    print(f"   Embedded dim: {info['embedded_dim']}")
    print(f"   Latent dim: {info['latent_dim']}")
    print(f"   Condition dim: {info['condition_dim']}")
    print(f"   Output dim: {info['output_dim']}")
    print(f"   Hidden dims: {info['hidden_dims']}")
    print(f"   Time emb dim: {info['time_emb_dim']}")
    print(f"   Use input embeddings: {info['use_input_embeddings']}")
    print(f"   Input emb dim: {info['input_emb_dim']}")
    print(f"   Dropout: {info['dropout']}")
    print(f"   Total parameters: {info['total_parameters']:,}")
    print(f"   Trainable parameters: {info['trainable_parameters']:,}")

    # Test forward pass
    batch_size = 4
    x_t = torch.randn(batch_size, 67)          # Embedded state
    t = torch.rand(batch_size)                  # Time in [0, 1]
    z = torch.randn(batch_size, 8)             # Latent vector
    condition = torch.randn(batch_size, 67)    # Start state condition

    print(f"\n🔧 Testing forward pass with batch_size={batch_size}...")
    print(f"   x_t shape: {x_t.shape}")
    print(f"   t shape: {t.shape}")
    print(f"   z shape: {z.shape}")
    print(f"   condition shape: {condition.shape}")

    with torch.no_grad():
        velocity = model(x_t, t, z, condition)

    print(f"\n✅ Forward pass successful!")
    print(f"   Output (velocity) shape: {velocity.shape}")
    print(f"   Expected shape: ({batch_size}, 67)")

    assert velocity.shape == (batch_size, 67), f"Shape mismatch: {velocity.shape}"

    print("\n✅ All tests passed!")
    print("="*80)

    # Compare with UniversalUNet
    print("\n📊 Comparison with UniversalUNet:")
    from src.model.universal_unet import UniversalUNet

    old_model = UniversalUNet(
        input_dim=142,  # embedded(67) + condition(67) + latent(8)
        output_dim=67,
        time_emb_dim=128,  # Note: UniversalUNet uses 'time_emb_dim', not 'time_embed_dim'
        hidden_dims=[256, 512, 512, 256]
    )

    old_params = sum(p.numel() for p in old_model.parameters())
    new_params = info['total_parameters']

    print(f"   UniversalUNet parameters: {old_params:,}")
    print(f"   HumanoidUNet parameters: {new_params:,}")
    print(f"   Difference: {new_params - old_params:,} ({(new_params/old_params - 1)*100:.1f}% increase)")
    print(f"   Reason: Deeper architecture + input embeddings")

    print("\n✅ HumanoidUNet is ready for training!")

if __name__ == "__main__":
    test_humanoid_unet()
