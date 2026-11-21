"""Distillation losses and offline TV overlap bounds."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def kd_forward_kl(
    logits: torch.Tensor,
    topk_vidx: torch.Tensor,
    topk_p: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Forward KL over V' with tail-mass term, averaged over masked positions."""
    s = F.log_softmax(logits.float(), dim=-1)
    valid = topk_vidx >= 0
    safe_idx = topk_vidx.clamp(min=0)

    s_k = torch.gather(s, dim=-1, index=safe_idx)
    p_k = torch.where(valid, topk_p.float(), 0.0)
    q_k = torch.where(valid, s_k.exp(), 0.0)

    p_rest = (1.0 - p_k.sum(dim=-1)).clamp(min=1e-8)
    q_rest = (1.0 - q_k.sum(dim=-1)).clamp(min=1e-8)

    nonzero = valid & (p_k > 0.0)
    term_k = torch.where(nonzero, p_k * (torch.log(p_k.clamp(min=1e-30)) - s_k), 0.0)
    kl_pos = term_k.sum(dim=-1) + p_rest * (torch.log(p_rest) - torch.log(q_rest))

    mask_f = mask.float()
    return (kl_pos * mask_f).sum() / mask_f.sum().clamp(min=1.0)


def hard_ce(
    logits: torch.Tensor,
    tokens: torch.Tensor,
    vprime_map: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Next-token cross-entropy over V', excluding last position and OOV targets."""
    if logits.shape[1] < 2:
        return logits.sum() * 0.0

    pred_logits = logits[:, :-1, :].float()
    next_tokens = tokens[:, 1:]
    targets = vprime_map[next_tokens].long()

    pos_mask = mask[:, :-1] & (targets >= 0)
    ce = F.cross_entropy(
        pred_logits.reshape(-1, pred_logits.shape[-1]),
        targets.clamp(min=0).reshape(-1),
        reduction="none",
    )
    pos_mask_f = pos_mask.float().reshape(-1)
    return (ce * pos_mask_f).sum() / pos_mask_f.sum().clamp(min=1.0)


def tv_overlap(
    logits: torch.Tensor,
    topk_vidx: torch.Tensor,
    topk_p: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-position lower and upper bounds on 1 - TV(teacher, student)."""
    s = F.log_softmax(logits.float(), dim=-1)
    valid = topk_vidx >= 0
    safe_idx = topk_vidx.clamp(min=0)

    q_k = torch.where(valid, torch.gather(s, dim=-1, index=safe_idx).exp(), 0.0)
    p_k = torch.where(valid, topk_p.float(), 0.0)

    lower = torch.minimum(p_k, q_k).sum(dim=-1)
    q_rest = (1.0 - q_k.sum(dim=-1)).clamp(min=0.0)
    p_min = topk_p.float().min(dim=-1).values
    upper = lower + torch.minimum(p_min, q_rest)
    return lower, upper
