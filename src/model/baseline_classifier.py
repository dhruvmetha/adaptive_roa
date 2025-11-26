import torch
import torch.nn as nn
from typing import List

class MLPClassifier(nn.Module):
    """
    Simple Multi-Layer Perceptron for binary classification.
    
    Args:
        input_dim: Dimension of input features
        hidden_dims: List of hidden layer dimensions
        dropout: Dropout probability
        activation: Activation function name ('relu', 'silu', 'tanh')
    """
    def __init__(self, 
                 input_dim: int, 
                 hidden_dims: List[int] = [128, 128], 
                 dropout: float = 0.1,
                 activation: str = 'relu'):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        # Activation map
        activations = {
            'relu': nn.ReLU(),
            'silu': nn.SiLU(),
            'tanh': nn.Tanh(),
            'gelu': nn.GELU()
        }
        act_fn = activations.get(activation.lower(), nn.ReLU())
        
        # Build hidden layers
        for dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(act_fn)
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = dim
            
        # Output layer (single logit for binary classification)
        layers.append(nn.Linear(prev_dim, 1))
        
        self.net = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.net(x)

