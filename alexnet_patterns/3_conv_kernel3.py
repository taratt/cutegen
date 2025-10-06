import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, num_classes=1000):
        super(Model, self).__init__()
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, padding=1)
    
    def forward(self, x):
        x = self.conv3(x)
        return x

# Test code
batch_size = 500
num_classes = 1000

def get_inputs():
    return [torch.randn(batch_size, 256, 13, 13)]

def get_init_inputs():
    return [num_classes]