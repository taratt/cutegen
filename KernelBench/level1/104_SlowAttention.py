import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Implements single-headed scaled dot-product attention:
        Attention(Q, K, V) = softmax(QKᵀ / sqrt(d_k)) V
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        """
        Args:
            Q (torch.Tensor): Query tensor of shape (B, H, L, D)
            K (torch.Tensor): Key   tensor of shape (B, H, L, D)
            V (torch.Tensor): Value tensor of shape (B, H, L, D)
        Returns:
            torch.Tensor: Output tensor of shape (B, H, L, D)
        """
        d_k = Q.size(-1)
        scores = torch.matmul(Q, K.transpose(-2, -1)) * (1.0 / math.sqrt(d_k))
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, V)
        return out

batch_size = 2
num_heads = 32
sequence_length = 512
embedding_dimension = 2048

def get_inputs():
    Q = torch.randn(batch_size, num_heads, sequence_length, embedding_dimension, device='cuda', dtype=torch.float16)
    K = torch.randn(batch_size, num_heads, sequence_length, embedding_dimension, device='cuda', dtype=torch.float16)
    V = torch.randn(batch_size, num_heads, sequence_length, embedding_dimension, device='cuda', dtype=torch.float16)
    return [Q, K, V]

def get_init_inputs():
    return []