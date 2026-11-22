"""Depth override in prune_init: evenly spaced source layers, first and last kept."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")

from train.prune_init import build_pruned_state_dict  # noqa: E402
from train.student import Student, StudentConfig  # noqa: E402

H, FFN_SRC, V, N_SRC, HD = 512, 1600, 64, 6, 128


def _fake_source(n_src: int) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(0)
    w = {"model.embed_tokens.weight": torch.randn(V, H, generator=g), "model.norm.weight": torch.ones(H)}
    for l in range(n_src):
        p = f"model.layers.{l}"
        w[f"{p}.input_layernorm.weight"] = torch.full((H,), float(l + 1))  # marks the source layer
        w[f"{p}.post_attention_layernorm.weight"] = torch.ones(H)
        w[f"{p}.self_attn.q_proj.weight"] = torch.randn(16 * HD, H, generator=g)
        w[f"{p}.self_attn.k_proj.weight"] = torch.randn(8 * HD, H, generator=g)
        w[f"{p}.self_attn.v_proj.weight"] = torch.randn(8 * HD, H, generator=g)
        w[f"{p}.self_attn.o_proj.weight"] = torch.randn(H, 16 * HD, generator=g)
        w[f"{p}.self_attn.q_norm.weight"] = torch.ones(HD)
        w[f"{p}.self_attn.k_norm.weight"] = torch.ones(HD)
        w[f"{p}.mlp.gate_proj.weight"] = torch.randn(FFN_SRC, H, generator=g)
        w[f"{p}.mlp.up_proj.weight"] = torch.randn(FFN_SRC, H, generator=g)
        w[f"{p}.mlp.down_proj.weight"] = torch.randn(H, FFN_SRC, generator=g)
    return w


@pytest.mark.parametrize("n_layers,expected", [(2, [0, 5]), (4, [0, 2, 3, 5]), (6, [0, 1, 2, 3, 4, 5])])
def test_layers_evenly_spaced_and_loadable(n_layers, expected):
    cfg = StudentConfig(n_vocab_in=V, n_vprime=16, n_layers=n_layers, max_seq=32)
    sd = build_pruned_state_dict(_fake_source(N_SRC), list(range(16)), cfg)
    got = [int(sd[f"layers.{i}.input_layernorm.weight"][0].item()) - 1 for i in range(n_layers)]
    assert got == expected
    Student(cfg).load_state_dict(sd)  # every tensor present with the right shape


def test_ffn_override_shapes():
    cfg = StudentConfig(n_vocab_in=V, n_vprime=16, n_layers=2, ffn=1024, max_seq=32)
    sd = build_pruned_state_dict(_fake_source(N_SRC), list(range(16)), cfg)
    assert sd["layers.0.mlp.up_proj.weight"].shape == (1024, H)


@pytest.mark.parametrize("n_layers", [1, N_SRC + 1])
def test_rejects_out_of_range_depth(n_layers):
    cfg = StudentConfig(n_vocab_in=V, n_vprime=16, n_layers=n_layers, max_seq=32)
    with pytest.raises(ValueError):
        build_pruned_state_dict(_fake_source(N_SRC), list(range(16)), cfg)
