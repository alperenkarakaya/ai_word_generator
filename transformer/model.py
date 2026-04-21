"""
Decoder-only Transformer (GPT-tarzı), Phase 1 için minimum gerekli.

Yaklaşım:
  - Token + öğrenilebilir pozisyon embedding'i
  - Pre-norm bloklar: LayerNorm -> Causal Multi-Head Self-Attention -> Residual
                      LayerNorm -> GELU FFN -> Residual
  - Weight tying: çıkış katmanı = token embedding ağırlıklarının transpozu
  - KV-cache desteği (sample.py için)

Phase 1 hedef boyutu (Colab T4 için yaklaşık 30M parametre):
  n_layer=6, d_model=384, n_head=6, d_ff=1536, block_size=256
"""
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 16000
    block_size: int = 256
    n_layer: int = 6
    n_head: int = 6
    d_model: int = 384
    d_ff: int = 1536
    dropout: float = 0.1
    pad_id: int = 0  # build_tokenizer.py'de pad_id=0 ayarlandı

    @property
    def head_dim(self) -> int:
        assert self.d_model % self.n_head == 0
        return self.d_model // self.n_head


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.n_head = cfg.n_head
        self.head_dim = cfg.head_dim
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.attn_drop = nn.Dropout(cfg.dropout)
        self.resid_drop = nn.Dropout(cfg.dropout)

    def forward(self, x, kv_cache=None):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=-1)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # B, h, T, d
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        if kv_cache is not None:
            past_k, past_v = kv_cache
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)
        new_cache = (k, v)

        # PyTorch 2.x: yerleşik scaled_dot_product_attention causal maskeyi de uygular.
        is_causal = kv_cache is None  # cache varken sadece son token query'liyor, mask gereksiz
        out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_drop.p if self.training else 0.0,
            is_causal=is_causal,
        )
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.resid_drop(self.proj(out))
        return out, new_cache


class FeedForward(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.fc1 = nn.Linear(cfg.d_model, cfg.d_ff)
        self.fc2 = nn.Linear(cfg.d_ff, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.drop(self.fc2(F.gelu(self.fc1(x))))


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.ff = FeedForward(cfg)

    def forward(self, x, kv_cache=None):
        h, new_cache = self.attn(self.ln1(x), kv_cache=kv_cache)
        x = x + h
        x = x + self.ff(self.ln2(x))
        return x, new_cache


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model, padding_idx=cfg.pad_id)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        # Weight tying
        self.head.weight = self.tok_emb.weight
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, idx, targets=None, kv_caches=None, pos_offset: int = 0):
        B, T = idx.shape
        assert T <= self.cfg.block_size, f"sequence {T} > block_size {self.cfg.block_size}"
        pos = torch.arange(pos_offset, pos_offset + T, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))

        new_caches = []
        for i, block in enumerate(self.blocks):
            cache_i = kv_caches[i] if kv_caches is not None else None
            x, new_cache = block(x, kv_cache=cache_i)
            new_caches.append(new_cache)

        x = self.ln_f(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=self.cfg.pad_id,
            )
        return logits, loss, new_caches

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
