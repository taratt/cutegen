import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, hidden_dim: int, seq_len: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len
        self.layernorm = nn.LayerNorm(hidden_dim, dtype=torch.bfloat16)
        self.gemm = nn.Linear(hidden_dim, hidden_dim, dtype=torch.bfloat16)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        return input_tensor + self.gemm(self.layernorm(input_tensor))

batch_size = 1
seq_len = 2048
hidden_dim = 4096

def get_inputs():
    input_tensor = torch.randn((batch_size, seq_len, hidden_dim), dtype=torch.bfloat16)
    return [input_tensor]

def get_init_inputs():
    return [hidden_dim, seq_len]