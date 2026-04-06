import torch
import torch.nn as nn
import torch.nn.functional as F

class ModelArgs:
    def __init__(self):
        self.dim = 4096
        self.intermediate_size = 14336

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        config = ModelArgs()
        self.w2 = nn.Linear(config.intermediate_size, config.dim, bias=False, dtype=torch.bfloat16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(x)

batch_size = 256
sequence_length = 1
intermediate_size = 14336

def get_inputs():
    x = torch.randn(batch_size, sequence_length, intermediate_size, device='cuda', dtype=torch.bfloat16)
    return [x]

def get_init_inputs():
    return []