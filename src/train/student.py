"""Qwen3-layout dense student model over pruned output vocabulary V'."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class StudentConfig:
    n_vocab_in: int = 151936
    n_vprime: int = 32768
    hidden: int = 512
    n_layers: int = 12
    n_heads: int = 4
    n_kv_heads: int = 1
    head_dim: int = 128
    ffn: int = 1536
    rope_theta: float = 1e6
    rms_eps: float = 1e-6
    max_seq: int = 2048


class RMSNorm(nn.Module):
    """RMSNorm computed in float32 and cast back to input dtype."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_fp32 = x.float()
        var = x_fp32.pow(2).mean(dim=-1, keepdim=True)
        normed = x_fp32 * torch.rsqrt(var + self.eps)
        return (normed * self.weight.float()).to(x.dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def _apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    q_rot = (q.float() * cos + _rotate_half(q.float()) * sin).to(q.dtype)
    k_rot = (k.float() * cos + _rotate_half(k.float()) * sin).to(k.dtype)
    return q_rot, k_rot


class StudentAttention(nn.Module):
    """Qwen3-style GQA with per-head QK RMSNorm and RoPE."""

    def __init__(self, cfg: StudentConfig) -> None:
        super().__init__()
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.head_dim = cfg.head_dim

        self.q_proj = nn.Linear(cfg.hidden, cfg.n_heads * cfg.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.hidden, cfg.n_kv_heads * cfg.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.hidden, cfg.n_kv_heads * cfg.head_dim, bias=False)
        self.o_proj = nn.Linear(cfg.n_heads * cfg.head_dim, cfg.hidden, bias=False)
        self.q_norm = RMSNorm(cfg.head_dim, eps=cfg.rms_eps)
        self.k_norm = RMSNorm(cfg.head_dim, eps=cfg.rms_eps)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = x.shape

        q = self.q_proj(x).view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)

        q = self.q_norm(q)
        k = self.k_norm(k)
        q, k = _apply_rope(q, k, cos, sin)

        if self.n_kv_heads != self.n_heads:
            n_rep = self.n_heads // self.n_kv_heads
            k = k.repeat_interleave(n_rep, dim=1)
            v = v.repeat_interleave(n_rep, dim=1)

        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(bsz, seq_len, self.n_heads * self.head_dim)
        return self.o_proj(out)


class StudentMLP(nn.Module):
    """SwiGLU feed-forward network without bias."""

    def __init__(self, cfg: StudentConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(cfg.hidden, cfg.ffn, bias=False)
        self.up_proj = nn.Linear(cfg.hidden, cfg.ffn, bias=False)
        self.down_proj = nn.Linear(cfg.ffn, cfg.hidden, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class StudentBlock(nn.Module):
    """Pre-norm Transformer block."""

    def __init__(self, cfg: StudentConfig) -> None:
        super().__init__()
        self.input_layernorm = RMSNorm(cfg.hidden, eps=cfg.rms_eps)
        self.self_attn = StudentAttention(cfg)
        self.post_attention_layernorm = RMSNorm(cfg.hidden, eps=cfg.rms_eps)
        self.mlp = StudentMLP(cfg)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attn(self.input_layernorm(x), cos, sin)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class Student(nn.Module):
    """Dense Qwen3-layout student model with untied pruned head."""

    def __init__(self, cfg: StudentConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = nn.Embedding(cfg.n_vocab_in, cfg.hidden)
        self.layers = nn.ModuleList([StudentBlock(cfg) for _ in range(cfg.n_layers)])
        self.norm = RMSNorm(cfg.hidden, eps=cfg.rms_eps)
        self.lm_head = nn.Linear(cfg.hidden, cfg.n_vprime, bias=False)

        inv_freq = cfg.rope_theta ** (
            -torch.arange(0, cfg.head_dim, 2, dtype=torch.float32) / cfg.head_dim
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        _, seq_len = tokens.shape
        x = self.embed_tokens(tokens)

        pos = torch.arange(seq_len, device=tokens.device, dtype=torch.float32)
        freqs = torch.outer(pos, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()[None, None, :, :]
        sin = emb.sin()[None, None, :, :]

        for layer in self.layers:
            x = layer(x, cos, sin)

        x = self.norm(x)
        return self.lm_head(x)
