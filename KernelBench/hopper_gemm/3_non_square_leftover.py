import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return torch.matmul(A, B)

M = 256
N = 14336
K = 4096

def get_inputs():
    A = torch.randn((M, K), dtype=torch.bfloat16)
    B = torch.randn((K, N), dtype=torch.bfloat16)
    return [A, B]

def get_init_inputs():
    # the model is stateless – nothing extra to initialise
    return []