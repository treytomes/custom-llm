"""
model.py — GPT-style language model (~50M parameters)

Architecture:
  - Token embeddings (no learned positional embeddings — RoPE handles position)
  - 12 transformer blocks with pre-norm, causal self-attention, and MLP
  - Rotary Position Embeddings (RoPE) in attention
  - Weight tying between token embedding and output projection
  - ~50M parameters with dim=512, layers=12, heads=8
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Rotary Position Embeddings ─────────────────────────────────────────────────
def precompute_rope_freqs(head_dim: int, max_seq: int, theta: float = 10000.0):
    assert head_dim % 2 == 0, "head_dim must be even for RoPE"

    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    positions = torch.arange(max_seq).float()
    angles = torch.outer(positions, freqs)
    angles = torch.cat([angles, angles], dim=-1)

    return angles.cos(), angles.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """
    Apply rotary embeddings to query or key tensor.

    Args:
        x:   (B, heads, T, head_dim)
        cos: (T, head_dim)
        sin: (T, head_dim)

    Returns:
        Tensor of same shape as x with RoPE applied.
    """
    # Rotate pairs: [-x1, x0, -x3, x2, ...]
    d = x.shape[-1]

    x1 = x[..., : d // 2]
    x2 = x[..., d // 2 :]
    x_rot = torch.cat([-x2, x1], dim=-1)

    cos = cos[:x.shape[2]].unsqueeze(0).unsqueeze(0)
    sin = sin[:x.shape[2]].unsqueeze(0).unsqueeze(0)

    return x * cos + x_rot * sin


# ── Causal Self-Attention ──────────────────────────────────────────────────────
class CausalSelfAttention(nn.Module):

    def __init__(self, dim: int, heads: int, max_seq: int, dropout: float = 0.1):
        super().__init__()

        assert dim % heads == 0, "dim must be divisible by heads"

        self.heads = heads
        self.head_dim = dim // heads

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)

        self.attn_dropout = nn.Dropout(dropout)

        # Retained for compatibility though no longer required
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(max_seq, max_seq)).view(1, 1, max_seq, max_seq)
        )

        cos, sin = precompute_rope_freqs(self.head_dim, max_seq)

        self.register_buffer("rope_cos", cos)
        self.register_buffer("rope_sin", sin)

    def forward(self, x: torch.Tensor):

        B, T, C = x.shape

        qkv = self.qkv(x)

        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, T, self.heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.heads, self.head_dim).transpose(1, 2)

        # Apply RoPE
        q = apply_rope(q, self.rope_cos, self.rope_sin)
        k = apply_rope(k, self.rope_cos, self.rope_sin)

        # FlashAttention / optimized attention kernel
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.attn_dropout.p if self.training else 0.0,
            is_causal=True,
        )

        out = out.transpose(1, 2).contiguous().view(B, T, C)

        return self.out(out)


# ── Transformer Block ──────────────────────────────────────────────────────────
class Block(nn.Module):

    def __init__(self, dim: int, heads: int, max_seq: int, mlp_ratio: int = 4, dropout: float = 0.1):
        super().__init__()

        self.ln1 = nn.LayerNorm(dim)
        self.attn = CausalSelfAttention(dim, heads, max_seq, dropout)

        self.ln2 = nn.LayerNorm(dim)

        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio, bias=False),
            nn.GELU(),
            nn.Linear(dim * mlp_ratio, dim, bias=False),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):

        x = x + self.dropout(self.attn(self.ln1(x)))
        x = x + self.dropout(self.mlp(self.ln2(x)))

        return x


# ── GPT Model ──────────────────────────────────────────────────────────────────
class GPT(nn.Module):

    def __init__(
        self,
        vocab_size: int,
        dim: int = 512,
        layers: int = 12,
        heads: int = 8,
        max_seq: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.tok_emb = nn.Embedding(vocab_size, dim)

        self.blocks = nn.ModuleList([
            Block(dim, heads, max_seq, dropout=dropout)
            for _ in range(layers)
        ])

        self.ln = nn.LayerNorm(dim)

        self.head = nn.Linear(dim, vocab_size, bias=False)

        # weight tying
        self.head.weight = self.tok_emb.weight

        self.max_seq = max_seq

        self.apply(self._init_weights)

    def _init_weights(self, module):

        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, idx: torch.Tensor):

        B, T = idx.shape

        assert T <= self.max_seq, f"Sequence length {T} exceeds max_seq {self.max_seq}"

        x = self.tok_emb(idx)

        for block in self.blocks:
            x = block(x)

        x = self.ln(x)

        logits = self.head(x)

        return logits


# ── Parameter count utility ────────────────────────────────────────────────────
def count_parameters(model: nn.Module):

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        "total": total,
        "trainable": trainable,
        "frozen": total - trainable,
        "total_millions": round(total / 1e6, 2),
    }