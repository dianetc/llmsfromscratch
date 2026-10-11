"""
Reference solution for the RoPE exercise.
"""
import torch
import torch.nn as nn


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta, d_k, max_seq_len, device=None):
        super().__init__()
        assert d_k % 2 == 0, "d_k must be even"
        # Step 1: one frequency per pair of dimensions: theta_i = theta^(-2i / d_k)
        i = torch.arange(d_k // 2, device=device, dtype=torch.float32)
        inv_freq = theta ** (-2 * i / d_k)                                         # (d_k/2,)
        # Step 2: all positions 0 .. max_seq_len-1
        positions = torch.arange(max_seq_len, device=device, dtype=torch.float32)  # (max_seq_len,)
        # Step 3: angle[m, i] = m * theta_i  (an "outer product")
        angles = positions[:, None] * inv_freq[None, :]                            # (max_seq_len, d_k/2)
        # Step 4: precompute cos and sin; buffers move with .to(device) but are not trained
        self.register_buffer("cos", angles.cos(), persistent=False)
        self.register_buffer("sin", angles.sin(), persistent=False)

    def forward(self, x, token_positions):
        # x: (..., seq_len, d_k), token_positions: (..., seq_len)
        cos = self.cos[token_positions]                     # (..., seq_len, d_k/2)
        sin = self.sin[token_positions]
        x_even, x_odd = x[..., 0::2], x[..., 1::2]           # the two halves of every pair
        rot_even = x_even * cos - x_odd * sin                # 2D rotation of each pair
        rot_odd = x_even * sin + x_odd * cos
        return torch.stack([rot_even, rot_odd], dim=-1).flatten(-2)   # interleave back: (..., seq_len, d_k)
