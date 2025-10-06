import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, bias=False, dtype=torch.bfloat16)
    
    def forward(self, x):
        x = self.conv1(x)
        return x

# Test code
batch_size = 100

def get_inputs():
    return [torch.randn((batch_size, 3, 224, 224), dtype=torch.bfloat16)]

def get_init_inputs():
    return []