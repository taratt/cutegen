import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return torch.bmm(A, B)

batch = 4
M = 4096
N = 2048
K = 8192

def get_inputs():
    A = torch.randn((batch, M, K), dtype=torch.bfloat16)
    B = torch.randn((batch, K, N), dtype=torch.bfloat16)
    return [A, B]

def get_init_inputs():
    # the model is stateless – nothing extra to initialise
    return []