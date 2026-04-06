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
        self.w1 = nn.Linear(config.dim, config.intermediate_size, bias=False, dtype=torch.bfloat16)
        self.w3 = nn.Linear(config.dim, config.intermediate_size, bias=False, dtype=torch.bfloat16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w1(x) * self.w3(x)

batch_size = 256
sequence_length = 1
embedding_dimension = 4096

def get_inputs():
    x = torch.randn(batch_size, sequence_length, embedding_dimension, device='cuda', dtype=torch.bfloat16)
    return [x]

def get_init_inputs():
    return []