import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, hidden_dim: int, hidden_dim_out: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.hidden_dim_out = hidden_dim_out
        self.linear = nn.Linear(hidden_dim, hidden_dim_out, dtype=torch.bfloat16)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        return self.linear(input_tensor)

batch_size = 256
seq_len = 1
hidden_dim = 4096
hidden_dim_out = 8192

def get_inputs():
    input_tensor = torch.randn((batch_size, seq_len, hidden_dim), dtype=torch.bfloat16)
    return [input_tensor]

def get_init_inputs():
    return [hidden_dim, hidden_dim_out]