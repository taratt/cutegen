import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        C = torch.matmul(A, B)
        C = torch.softmax(C, dim=1)
        return C

M = 4096
N = 4096
K = 4096

def get_inputs():
    A = torch.randn((M, K), dtype=torch.bfloat16)
    B = torch.randn((K, N), dtype=torch.bfloat16)
    return [A, B]

def get_init_inputs():
    return []