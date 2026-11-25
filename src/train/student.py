"""Qwen3-layout dense student model over pruned output vocabulary V'."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# FFN block types; `ffn` is the hidden width for every one of them.
FFN_TYPES = ("swiglu", "mlp", "mlp_rational", "gr_kan", "bspline_kan")

# Safe-Pade (KAT) fit to SiLU used by RationalGroup(init="silu").
# Derived once, offline: least squares on 8001 uniform points of [-8, 8] (wider than
# the required [-4, 4] so that activations outside the fit window stay sane), then 25
# rounds of iteratively reweighted least squares towards the minimax solution, rounded
# to float32. Max |f - silu| = 1.4e-2 on [-4, 4] and 1.5e-2 on [-8, 8]; budget 5e-2.
# Hardcoded so that the init is deterministic and independent of the global RNG.
_SILU_NUM = (
    -0.0010517576,
    0.531347215,
    0.300934136,
    0.0623935871,
    0.00546962116,
    0.000165717604,
)
_SILU_DEN = (0.153784603, 0.0625163242, 0.00922228303, -0.000132946327)


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
    ffn_type: str = "swiglu"
    rational_groups: int = 8
    rational_m: int = 5
    rational_n: int = 4
    kan_grid: int = 5
    kan_order: int = 3
    kan_range: float = 2.0


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


class RationalGroup(nn.Module):
    """Safe-Pade activation shared within channel groups.

    f(x) = sum_{i=0..m} a_i x^i / (1 + |sum_{j=1..n} b_j x^j|), evaluated in float32.
    Channel c of the last dimension uses group c // (channels / groups); the groups
    share one rational function, so the activation costs no bytes per channel.
    """

    def __init__(
        self,
        channels: int,
        groups: int = 8,
        m: int = 5,
        n: int = 4,
        init: str = "identity",
    ) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError(f"channels must be positive, got {channels}")
        if groups <= 0:
            raise ValueError(f"groups must be positive, got {groups}")
        if channels % groups != 0:
            raise ValueError(f"channels ({channels}) must be divisible by groups ({groups})")
        if m < 0 or n < 0:
            raise ValueError(f"degrees must be non-negative, got m={m}, n={n}")

        numerator = torch.zeros(groups, m + 1, dtype=torch.float32)
        denominator = torch.zeros(groups, n, dtype=torch.float32)
        if init == "identity":
            if m < 1:
                raise ValueError("init='identity' needs m >= 1")
            numerator[:, 1] = 1.0
        elif init == "silu":
            if m < len(_SILU_NUM) - 1 or n < len(_SILU_DEN):
                raise ValueError(
                    f"init='silu' needs m >= {len(_SILU_NUM) - 1} and n >= {len(_SILU_DEN)}, "
                    f"got m={m}, n={n}"
                )
            numerator[:, : len(_SILU_NUM)] = torch.tensor(_SILU_NUM, dtype=torch.float32)
            denominator[:, : len(_SILU_DEN)] = torch.tensor(_SILU_DEN, dtype=torch.float32)
        else:
            raise ValueError(f"Unknown RationalGroup init {init!r}; expected 'identity' or 'silu'")

        self.channels = channels
        self.groups = groups
        self.m = m
        self.n = n
        self.numerator = nn.Parameter(numerator)  # a_0 .. a_m, [groups, m + 1]
        self.denominator = nn.Parameter(denominator)  # b_1 .. b_n, [groups, n]
        self.register_buffer(
            "group_index",
            torch.arange(channels, dtype=torch.long) // (channels // groups),
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_fp32 = x if x.dtype == torch.float32 else x.float()
        num = self.numerator[self.group_index]  # [channels, m + 1]
        den = self.denominator[self.group_index]  # [channels, n]

        powers = [torch.ones_like(x_fp32)]
        for _ in range(max(self.m, self.n)):
            powers.append(powers[-1] * x_fp32)
        pows = torch.stack(powers, dim=-1)  # (..., channels, max(m, n) + 1)

        numerator = (pows[..., : self.m + 1] * num).sum(dim=-1)
        den_poly = (pows[..., 1 : self.n + 1] * den).sum(dim=-1)
        out = numerator / (1.0 + den_poly.abs())
        return out if x.dtype == torch.float32 else out.to(x.dtype)


def bspline_basis(
    x: torch.Tensor,
    grid: int,
    order: int,
    rng: float,
) -> torch.Tensor:
    """Uniform B-spline basis of shape ``x.shape + (grid + order,)``.

    Knots are uniform with spacing h = 2 * rng / grid, t_i = -rng + (i - order) * h for
    i = 0 .. grid + 2 * order, and the bases come from the Cox-de Boor recursion started
    from the order-0 (half-open [t_i, t_{i+1})) indicator. For x in [-rng, rng) the bases
    are non-negative and sum to 1; for x < t_0 or x >= t_last they are all 0.
    """
    if grid < 1:
        raise ValueError(f"grid must be positive, got {grid}")
    if order < 0:
        raise ValueError(f"order must be non-negative, got {order}")

    x_fp32 = x if x.dtype == torch.float32 else x.float()
    h = 2.0 * rng / grid
    n_knots = grid + 2 * order + 1
    idx = torch.arange(n_knots, device=x_fp32.device, dtype=x_fp32.dtype)
    knots = (idx - order) * h - rng

    x_wide = x_fp32.unsqueeze(-1)
    basis = ((x_wide >= knots[:-1]) & (x_wide < knots[1:])).to(x_fp32.dtype)

    for p in range(1, order + 1):
        n_bases = basis.shape[-1] - 1
        i = torch.arange(n_bases, device=x_fp32.device)
        left = (x_wide - knots[i]) / (knots[i + p] - knots[i])
        right = (knots[i + p + 1] - x_wide) / (knots[i + p + 1] - knots[i + 1])
        basis = left * basis[..., :n_bases] + right * basis[..., 1:]

    return basis if x.dtype == torch.float32 else basis.to(x.dtype)


class KANLinear(nn.Module):
    """Efficient-KAN layer: SiLU base map plus per-edge B-spline on a uniform grid."""

    def __init__(
        self,
        in_f: int,
        out_f: int,
        grid: int = 5,
        order: int = 3,
        rng: float = 2.0,
    ) -> None:
        super().__init__()
        if in_f <= 0 or out_f <= 0:
            raise ValueError(f"in_f and out_f must be positive, got {in_f}, {out_f}")
        if grid < 1:
            raise ValueError(f"grid must be positive, got {grid}")
        if order < 0:
            raise ValueError(f"order must be non-negative, got {order}")

        n_basis = grid + order
        self.in_f = in_f
        self.out_f = out_f
        self.grid = grid
        self.order = order
        self.rng = float(rng)

        self.base_weight = nn.Parameter(torch.empty(out_f, in_f))
        self.spline_weight = nn.Parameter(torch.empty(out_f, in_f * n_basis))
        # Base map uses the standard Student init scale; the spline starts small so that
        # a freshly built layer is close to its SiLU base map.
        nn.init.normal_(self.base_weight, mean=0.0, std=0.02)
        spline_std = min(0.02, 0.02 / math.sqrt(in_f * n_basis))
        nn.init.normal_(self.spline_weight, mean=0.0, std=spline_std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        basis = bspline_basis(x, self.grid, self.order, self.rng)
        flat = basis.reshape(*x.shape[:-1], self.in_f * (self.grid + self.order))
        return F.linear(F.silu(x), self.base_weight) + F.linear(flat, self.spline_weight)


class DenseFFN(nn.Module):
    """down(act(up(x))); ``act=None`` means SiLU, i.e. a plain two-matrix MLP."""

    def __init__(self, hidden: int, ffn: int, act: nn.Module | None = None) -> None:
        super().__init__()
        self.up_proj = nn.Linear(hidden, ffn, bias=False)
        self.act = act
        self.down_proj = nn.Linear(ffn, hidden, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.up_proj(x)
        h = F.silu(h) if self.act is None else self.act(h)
        return self.down_proj(h)


class GRKANFFN(nn.Module):
    """down(act2(up(act1(x)))): a rational unit before each linear map (KAT layout)."""

    def __init__(
        self, hidden: int, ffn: int, act1: nn.Module, act2: nn.Module
    ) -> None:
        super().__init__()
        self.act1 = act1
        self.up_proj = nn.Linear(hidden, ffn, bias=False)
        self.act2 = act2
        self.down_proj = nn.Linear(ffn, hidden, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act2(self.up_proj(self.act1(x))))


class BsplineKANFFN(nn.Module):
    """KAN block: kan1(hidden -> ffn), LayerNorm without affine, kan2(ffn -> hidden)."""

    def __init__(self, cfg: StudentConfig) -> None:
        super().__init__()
        self.kan1 = KANLinear(cfg.hidden, cfg.ffn, cfg.kan_grid, cfg.kan_order, cfg.kan_range)
        self.norm = nn.LayerNorm(cfg.ffn, elementwise_affine=False)
        self.kan2 = KANLinear(cfg.ffn, cfg.hidden, cfg.kan_grid, cfg.kan_order, cfg.kan_range)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.kan2(self.norm(self.kan1(x)))


def build_ffn(cfg: StudentConfig) -> nn.Module:
    """Build the FFN block for ``cfg.ffn_type``; placed at ``StudentBlock.mlp``."""
    groups, m, n = cfg.rational_groups, cfg.rational_m, cfg.rational_n
    if cfg.ffn_type == "swiglu":
        return StudentMLP(cfg)
    if cfg.ffn_type == "mlp":
        return DenseFFN(cfg.hidden, cfg.ffn)
    if cfg.ffn_type == "mlp_rational":
        return DenseFFN(cfg.hidden, cfg.ffn, act=RationalGroup(cfg.ffn, groups, m, n, "silu"))
    if cfg.ffn_type == "gr_kan":
        return GRKANFFN(
            cfg.hidden,
            cfg.ffn,
            act1=RationalGroup(cfg.hidden, groups, m, n, "identity"),
            act2=RationalGroup(cfg.ffn, groups, m, n, "silu"),
        )
    if cfg.ffn_type == "bspline_kan":
        return BsplineKANFFN(cfg)
    raise ValueError(
        f"Unknown ffn_type {cfg.ffn_type!r}; expected one of {', '.join(FFN_TYPES)}"
    )


class StudentBlock(nn.Module):
    """Pre-norm Transformer block."""

    def __init__(self, cfg: StudentConfig) -> None:
        super().__init__()
        self.input_layernorm = RMSNorm(cfg.hidden, eps=cfg.rms_eps)
        self.self_attn = StudentAttention(cfg)
        self.post_attention_layernorm = RMSNorm(cfg.hidden, eps=cfg.rms_eps)
        self.mlp = build_ffn(cfg)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attn(self.input_layernorm(x), cos, sin)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class Student(nn.Module):
    """Dense Qwen3-layout student model with untied pruned head."""

    def __init__(self, cfg: StudentConfig) -> None:
        super().__init__()
        if cfg.ffn_type not in FFN_TYPES:
            raise ValueError(
                f"Unknown ffn_type {cfg.ffn_type!r}; expected one of {', '.join(FFN_TYPES)}"
            )
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
