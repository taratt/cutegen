import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor, scale_a: torch.Tensor, scale_b: torch.Tensor) -> torch.Tensor:
        output = torch.matmul(A.to(torch.float32), B.to(torch.float32)).to(torch.float8_e4m3fn)
        return output

N = 4096

def get_inputs():
    A = torch.randn((N, N), dtype=torch.float32).to(torch.float8_e4m3fn)
    B = torch.randn((N, N), dtype=torch.float32).to(torch.float8_e4m3fn)
    scale_a = torch.tensor(1.0)
    scale_b = torch.tensor(1.0)
    return [A, B, scale_a, scale_b]

def get_init_inputs():
    # the model is stateless – nothing extra to initialise
    return []