# Copyright (c) Meta Platforms, Inc. and affiliates.
# Modifications Copyright (c) 2025 WindVChen.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/master/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/models/vision_transformer.py

import logging
import os
import warnings

import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F

XFORMERS_AVAILABLE = False

try:
    from xformers.ops import memory_efficient_attention, unbind
    XFORMERS_AVAILABLE = True
except ImportError:
    pass


class CrossAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        context_dim: int = None,
        num_heads: int = 8,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
        qk_norm: bool = False,
        fused_attn: bool = True,  # use F.scaled_dot_product_attention or not
        rope=None,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.fused_attn = fused_attn
        
        # Set context dimension (for key/value) - defaults to same as query dimension
        context_dim = context_dim or dim
        
        # Separate projections for query vs key/value
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(context_dim, dim * 2, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)
        self.rope = rope

    def forward(self, x: Tensor, context: Tensor = None, pos=None, context_pos=None) -> Tensor:
        """
        Args:
            x: Query tensor of shape (B, N_q, C)
            context: Key/Value tensor of shape (B, N_kv, C_context). If None, uses x (self-attention)
            pos: Positional encoding for queries
            context_pos: Positional encoding for keys/values
        """
        # Use self-attention if no context is provided
        if context is None:
            raise ValueError("For self-attention, please use the Attention class instead.")
            
        B, N_q, C = x.shape
        B_ctx, N_kv, C_ctx = context.shape
        assert B == B_ctx, "Batch size must match between query and context"
        
        # Project queries and key-values separately
        q = self.q(x).reshape(B, N_q, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        kv = self.kv(context).reshape(B, N_kv, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)
        
        # Apply normalization
        q, k = self.q_norm(q), self.k_norm(k)

        # Apply rope encoding if available
        if self.rope is not None:
            if pos is not None:
                q = self.rope(q, pos)
            if context_pos is not None:
                k = self.rope(k, context_pos)

        # Compute attention
        if self.fused_attn:
            x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p if self.training else 0.0)
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N_q, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    

class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
        qk_norm: bool = False,
        fused_attn: bool = True,  # use F.scaled_dot_product_attention or not
        rope=None,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.fused_attn = fused_attn

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)
        self.rope = rope

    def forward(self, x: Tensor, pos=None, context: Tensor = None, context_pos=None) -> Tensor:
        if context is not None or context_pos is not None:
            raise ValueError("For cross-attention, please use the CrossAttention class instead.")

        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        # Check if any position has special marker (-1) that indicates no rope encoding
        has_special_positions = False
        if pos is not None:
            has_special_positions = -1 in pos
        
        # Apply rope encoding if available
        if self.rope is not None:
            # Set positions with -1 to 0 for rope encoding
            pos_for_rope = torch.where(pos == -1, 0, pos)
            q_r = self.rope(q, pos_for_rope)
            k_r = self.rope(k, pos_for_rope)

        # Use fused attention when possible and no special positions exist
        if self.fused_attn and not has_special_positions:
            x = F.scaled_dot_product_attention(q_r if self.rope is not None else q, k_r if self.rope is not None else k, v, dropout_p=self.attn_drop.p if self.training else 0.0)
        elif has_special_positions and self.rope is not None:
            # Memory-efficient attention computation with selective rope application
            # Find positions with pos=-1 (same across batch dimension)
            pos_mask = (pos[0] == -1).any(dim=-1)  # (N,) - same for all batch items
            others_indices = torch.where(pos_mask)[0]
            img_indices = torch.where(~pos_mask)[0]

            # For img tokens: compute attention with mixed keys
            if len(img_indices) > 0:
                q_img_rope = q_r[..., img_indices, :]
                
                # Build k_mixed efficiently using scatter instead of clone + overwrite
                # This reduces peak memory usage
                k_mixed = torch.empty_like(k)
                k_mixed[..., img_indices, :] = k_r[..., img_indices, :]
                k_mixed[..., others_indices, :] = k[..., others_indices, :]

                x_img = F.scaled_dot_product_attention(
                    q_img_rope, k_mixed, v, 
                    dropout_p=self.attn_drop.p if self.training else 0.0
                )
            
            # For other tokens: use original q and k (no rope)
            if len(others_indices) > 0:
                q_others_norope = q[..., others_indices, :]
                x_others = F.scaled_dot_product_attention(
                    q_others_norope, k, v, 
                    dropout_p=self.attn_drop.p if self.training else 0.0
                )
            
            # Pre-allocate output tensor with correct dtype from attention results
            if len(img_indices) > 0:
                x = torch.empty_like(q, dtype=x_img.dtype)
                x[..., img_indices, :] = x_img
            
            # Assign other tokens result if computed
            if len(others_indices) > 0:
                x[..., others_indices, :] = x_others

        else:
            raise NotImplementedError("Invalid attention configuration")

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MemEffAttention(Attention):
    def forward(self, x: Tensor, attn_bias=None, pos=None) -> Tensor:
        assert pos is None
        if not XFORMERS_AVAILABLE:
            if attn_bias is not None:
                raise AssertionError("xFormers is required for using nested tensors")
            return super().forward(x)

        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)

        q, k, v = unbind(qkv, 2)

        x = memory_efficient_attention(q, k, v, attn_bias=attn_bias)
        x = x.reshape([B, N, C])

        x = self.proj(x)
        x = self.proj_drop(x)
        return x
