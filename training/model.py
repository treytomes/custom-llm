import torch
import torch.nn as nn
import math


class CausalSelfAttention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads

        self.qkv = nn.Linear(dim, dim * 3)
        self.out = nn.Linear(dim, dim)

        self.register_buffer(
            "mask",
            torch.tril(torch.ones(1024, 1024)).view(1,1,1024,1024)
        )


    def forward(self, x, mask=None):
        B, T, C = x.shape

        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, T, self.heads, self.head_dim).transpose(1,2)
        k = k.view(B, T, self.heads, self.head_dim).transpose(1,2)
        v = v.view(B, T, self.heads, self.head_dim).transpose(1,2)

        att = (q @ k.transpose(-2,-1)) / math.sqrt(self.head_dim)

        att = att.masked_fill(self.mask[:,:,:T,:T] == 0, -1e9)

        att = torch.softmax(att, dim=-1)

        out = att @ v
        out = out.transpose(1,2).contiguous().view(B,T,C)

        return self.out(out)


class Block(nn.Module):
    def __init__(self, dim, heads, mlp_ratio=4):
        super().__init__()

        self.ln1 = nn.LayerNorm(dim)
        self.attn = CausalSelfAttention(dim, heads)

        self.ln2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim*mlp_ratio),
            nn.GELU(),
            nn.Linear(dim*mlp_ratio, dim)
        )

        self.dropout = nn.Dropout(0.1)


    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        x = x + self.dropout(self.attn(self.ln1(x)))
        x = x + self.dropout(self.mlp(self.ln2(x)))
        return x


class GPT(nn.Module):
    def __init__(self, vocab_size, dim=512, layers=12, heads=8, max_seq=1024):
        super().__init__()

        self.tok_emb = nn.Embedding(vocab_size, dim)
        self.pos_emb = nn.Embedding(max_seq, dim)

        self.blocks = nn.ModuleList([
            Block(dim, heads) for _ in range(layers)
        ])

        self.ln = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight
        
        self.max_seq = max_seq


    def forward(self, idx):
        B, T = idx.shape

        pos = torch.arange(0, T, device=idx.device).unsqueeze(0)
        x = self.tok_emb(idx) + self.pos_emb(pos)

        for block in self.blocks:
            x = block(x)

        x = self.ln(x)
        logits = self.head(x)

        return logits