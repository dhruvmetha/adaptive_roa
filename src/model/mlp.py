import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_channels: list):
        super().__init__()
        self.layers = nn.ModuleList()
        
        current_dim = input_dim
        for hidden_dim in hidden_channels:
            self.layers.append(nn.Linear(current_dim, hidden_dim))
            self.layers.append(nn.ReLU())
            current_dim = hidden_dim
        
        self.layers.append(nn.Linear(current_dim, output_dim))
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x