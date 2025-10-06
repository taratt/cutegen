import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super(Model, self).__init__()
        self.conv1d = nn.Conv1d(in_channels, out_channels, kernel_size, bias=False, dtype=torch.bfloat16)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            return self.conv1d(x)

# Test code
batch_size = 1
in_channels = 2048
out_channels = 2048
kernel_size = 3
length = 1024

def get_inputs():
    x = torch.ones((batch_size, in_channels, length), dtype=torch.bfloat16)
    return [x]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]  # Provide in_channels, out_channels, kernel_size for initialization