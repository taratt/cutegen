import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        C = torch.matmul(A, B)
        C = torch.relu(C)
        C = C + bias
        return C

N = 4096

def get_inputs():
    A = torch.randn((N, N), dtype=torch.bfloat16)
    B = torch.randn((N, N), dtype=torch.bfloat16)
    bias = torch.randn((N, N), dtype=torch.bfloat16)
    return [A, B, bias]

def get_init_inputs():
    return []