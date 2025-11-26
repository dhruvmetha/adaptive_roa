"""
Utilities for loading pretrained contrastive encoders

Provides functions to load pretrained encoders and integrate them with
downstream tasks (classification, flow matching, etc.).
"""
import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional, Literal
from src.contrastive.contrastive_learner import ContrastiveLearner
from src.model.contrastive_encoder import ContrastiveEncoder


def load_pretrained_encoder(
    checkpoint_path: str,
    freeze: bool = True,
    device: Optional[torch.device] = None
) -> ContrastiveEncoder:
    """
    Load pretrained contrastive encoder from checkpoint

    Args:
        checkpoint_path: Path to Lightning checkpoint (.ckpt file)
        freeze: If True, freeze encoder weights (default: True)
        device: Device to load encoder on (default: None = auto-detect)

    Returns:
        Pretrained encoder model

    Example:
        >>> encoder = load_pretrained_encoder(
        ...     "outputs/humanoid_contrastive_repr/.../checkpoints/best.ckpt",
        ...     freeze=True
        ... )
        >>> embeddings = encoder(normalized_states)  # [B, embedding_dim]
    """
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Load Lightning module
    print(f"Loading pretrained encoder from: {checkpoint_path}")

    # Determine device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load checkpoint
    learner = ContrastiveLearner.load_from_checkpoint(
        checkpoint_path,
        map_location=device,
        strict=False  # Allow loading even if system config differs
    )

    # Extract encoder
    encoder = learner.encoder

    # Freeze if requested
    if freeze:
        encoder.eval()
        for param in encoder.parameters():
            param.requires_grad = False
        print("✅ Encoder loaded (frozen)")
    else:
        print("✅ Encoder loaded (trainable)")

    print(f"   Input dim: {encoder.input_dim}")
    print(f"   Embedding dim: {encoder.embedding_dim}")

    return encoder


def load_encoder_with_system(
    checkpoint_path: str,
    system,
    freeze: bool = True,
    device: Optional[torch.device] = None
) -> tuple[ContrastiveEncoder, any]:
    """
    Load pretrained encoder along with system for normalization

    Convenience function that loads both encoder and system for
    easy integration into downstream tasks.

    Args:
        checkpoint_path: Path to Lightning checkpoint
        system: DynamicalSystem instance (for normalization)
        freeze: If True, freeze encoder weights
        device: Device to load encoder on

    Returns:
        Tuple of (encoder, system)

    Example:
        >>> from src.systems.humanoid import HumanoidSystem
        >>> system = HumanoidSystem()
        >>> encoder, system = load_encoder_with_system(
        ...     "outputs/humanoid_contrastive_repr/.../checkpoints/best.ckpt",
        ...     system=system,
        ...     freeze=True
        ... )
        >>>
        >>> # Use in downstream task
        >>> raw_states = ...  # [B, 67]
        >>> norm_states = system.normalize_state(raw_states)
        >>> embeddings = encoder(norm_states)  # [B, embedding_dim]
    """
    encoder = load_pretrained_encoder(checkpoint_path, freeze=freeze, device=device)
    return encoder, system


class EncoderWrapper(nn.Module):
    """
    Wrapper that combines encoder with system normalization

    Useful for downstream tasks that need normalized embeddings from raw states.
    """

    def __init__(self, encoder: ContrastiveEncoder, system):
        """
        Initialize encoder wrapper

        Args:
            encoder: Pretrained contrastive encoder
            system: DynamicalSystem instance for normalization
        """
        super().__init__()
        self.encoder = encoder
        self.system = system

    def forward(self, raw_states: torch.Tensor) -> torch.Tensor:
        """
        Encode raw states to embeddings (handles normalization internally)

        Args:
            raw_states: Raw state tensor [B, state_dim]

        Returns:
            Embeddings [B, embedding_dim]
        """
        # Normalize using system
        normalized_states = self.system.normalize_state(raw_states)

        # Encode
        embeddings = self.encoder(normalized_states)

        return embeddings

    def freeze(self):
        """Freeze encoder weights"""
        self.encoder.eval()
        for param in self.encoder.parameters():
            param.requires_grad = False

    def unfreeze(self):
        """Unfreeze encoder weights"""
        self.encoder.train()
        for param in self.encoder.parameters():
            param.requires_grad = True


def create_encoder_wrapper(
    checkpoint_path: str,
    system,
    freeze: bool = True,
    device: Optional[torch.device] = None
) -> EncoderWrapper:
    """
    Create encoder wrapper with normalization

    Convenience function to create EncoderWrapper from checkpoint.

    Args:
        checkpoint_path: Path to Lightning checkpoint
        system: DynamicalSystem instance
        freeze: If True, freeze encoder weights
        device: Device to load encoder on

    Returns:
        EncoderWrapper instance

    Example:
        >>> from src.systems.humanoid import HumanoidSystem
        >>> system = HumanoidSystem()
        >>> wrapper = create_encoder_wrapper(
        ...     "outputs/humanoid_contrastive_repr/.../checkpoints/best.ckpt",
        ...     system=system,
        ...     freeze=True
        ... )
        >>>
        >>> # Use with raw states (normalization handled internally)
        >>> raw_states = ...  # [B, 67]
        >>> embeddings = wrapper(raw_states)  # [B, embedding_dim]
    """
    encoder = load_pretrained_encoder(checkpoint_path, freeze=freeze, device=device)
    wrapper = EncoderWrapper(encoder, system)

    if freeze:
        wrapper.freeze()

    return wrapper


# =====================================================================
# EXAMPLE USAGE PATTERNS
# =====================================================================

def example_frozen_encoder():
    """
    Example: Using frozen encoder as feature extractor for classification
    """
    from src.systems.humanoid import HumanoidSystem
    import torch.nn as nn

    # Load pretrained encoder (frozen)
    system = HumanoidSystem()
    encoder = load_pretrained_encoder(
        "outputs/humanoid_contrastive_repr/checkpoints/best.ckpt",
        freeze=True
    )

    # Create classifier head
    classifier = nn.Sequential(
        nn.Linear(encoder.embedding_dim, 64),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(64, 3)  # 3 classes: success, failure, separatrix
    )

    # Forward pass
    raw_states = torch.randn(32, 67)  # [B, 67]
    norm_states = system.normalize_state(raw_states)
    embeddings = encoder(norm_states)  # [B, embedding_dim]
    logits = classifier(embeddings)  # [B, 3]

    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Logits shape: {logits.shape}")


def example_finetuning():
    """
    Example: Fine-tuning encoder jointly with downstream task
    """
    from src.systems.humanoid import HumanoidSystem
    import torch.nn as nn
    import torch.optim as optim

    # Load pretrained encoder (trainable)
    system = HumanoidSystem()
    encoder = load_pretrained_encoder(
        "outputs/humanoid_contrastive_repr/checkpoints/best.ckpt",
        freeze=False  # Allow fine-tuning
    )

    # Create classifier
    classifier = nn.Linear(encoder.embedding_dim, 3)

    # Combine into model
    class DownstreamModel(nn.Module):
        def __init__(self, encoder, classifier):
            super().__init__()
            self.encoder = encoder
            self.classifier = classifier

        def forward(self, x):
            embeddings = self.encoder(x)
            return self.classifier(embeddings)

    model = DownstreamModel(encoder, classifier)

    # Optimizer with different learning rates (lower LR for pretrained encoder)
    optimizer = optim.Adam([
        {'params': encoder.parameters(), 'lr': 1e-5},  # Low LR for pretrained
        {'params': classifier.parameters(), 'lr': 1e-3}  # Higher LR for new head
    ])

    print("Model ready for fine-tuning")
    print(f"Encoder LR: 1e-5")
    print(f"Classifier LR: 1e-3")


if __name__ == "__main__":
    print("="*80)
    print("Contrastive Encoder Loader Examples")
    print("="*80)
    print()

    print("Example 1: Frozen encoder for feature extraction")
    print("-"*80)
    example_frozen_encoder()
    print()

    print("Example 2: Fine-tuning encoder on downstream task")
    print("-"*80)
    example_finetuning()
    print()
