"""b10 FFN tooling: ffn_init checkpoint building and the held-out layerwise fit."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from train import ffn_init  # noqa: E402
from train.student import Student, StudentConfig  # noqa: E402

V, VP, H, FFN, LAYERS, HD = 64, 16, 32, 64, 2, 16

# A config dict as written by checkpoints from before the b10 change: no ffn_* keys.
OLD_CONFIG_KEYS = (
    "n_vocab_in",
    "n_vprime",
    "hidden",
    "n_layers",
    "n_heads",
    "n_kv_heads",
    "head_dim",
    "ffn",
    "rope_theta",
    "rms_eps",
    "max_seq",
)


def _old_config(**kw) -> dict:
    full = {
        "n_vocab_in": V,
        "n_vprime": VP,
        "hidden": H,
        "n_layers": LAYERS,
        "n_heads": 2,
        "n_kv_heads": 1,
        "head_dim": HD,
        "ffn": FFN,
        "rope_theta": 1e6,
        "rms_eps": 1e-6,
        "max_seq": 32,
    }
    full.update(kw)
    return {key: full[key] for key in OLD_CONFIG_KEYS}


def _source(tmp_path, **cfg_kw) -> tuple[object, dict]:
    """Write a pre-b10 (swiglu, old config keys) source checkpoint to tmp_path."""
    cfg = _old_config(**cfg_kw)
    model = Student(StudentConfig(**cfg))
    ckpt = {
        "config": cfg,
        "state_dict": {k: v.detach().clone() for k, v in model.state_dict().items()},
        "vprime": list(range(VP)),
        "step": 7,
    }
    path = tmp_path / "src.pt"
    torch.save(ckpt, path)
    return path, ckpt


def test_old_swiglu_checkpoint_still_loads(tmp_path):
    src_path, _ = _source(tmp_path)
    out_path = tmp_path / "out.pt"
    assert ffn_init.main(["--src", str(src_path), "--config", "{}", "--out", str(out_path)]) == 0
    assert out_path.is_file()
    assert not out_path.with_name(out_path.name + ".tmp").exists()

    out = torch.load(out_path, map_location="cpu", weights_only=False)
    assert out["config"]["ffn_type"] == "swiglu"
    assert out["config"]["ffn"] == FFN
    assert out["step"] == 0 and out["vprime"] == list(range(VP))
    assert out["ffn_init"]["fit"] is None
    Student(StudentConfig(**out["config"])).load_state_dict(out["state_dict"])


def test_missing_non_ffn_key_exits_2(tmp_path, capsys):
    src_path, ckpt = _source(tmp_path)
    del ckpt["state_dict"]["norm.weight"]
    del ckpt["state_dict"]["layers.1.self_attn.q_proj.weight"]
    torch.save(ckpt, src_path)
    out_path = tmp_path / "out.pt"

    assert ffn_init.main(["--src", str(src_path), "--config", "{}", "--out", str(out_path)]) == 2
    err = capsys.readouterr().err
    assert "missing" in err
    assert "norm.weight" in err and "layers.1.self_attn.q_proj.weight" in err
    assert not out_path.exists()


def test_unexpected_non_ffn_source_key_exits_2(tmp_path, capsys):
    src_path, ckpt = _source(tmp_path)
    ckpt["state_dict"]["bogus.weight"] = torch.zeros(3)
    torch.save(ckpt, src_path)
    out_path = tmp_path / "out.pt"

    assert ffn_init.main(["--src", str(src_path), "--config", "{}", "--out", str(out_path)]) == 2
    assert "bogus.weight" in capsys.readouterr().err
    assert not out_path.exists()


def test_write_failure_leaves_no_output(tmp_path, monkeypatch, capsys):
    src_path, _ = _source(tmp_path)
    out_path = tmp_path / "out.pt"

    def boom(*_args, **_kwargs):
        raise RuntimeError("disk full")

    monkeypatch.setattr(ffn_init.torch, "save", boom)
    assert ffn_init.main(["--src", str(src_path), "--config", "{}", "--out", str(out_path)]) == 1
    assert "disk full" in capsys.readouterr().err
    assert not out_path.exists()
    assert not out_path.with_name(out_path.name + ".tmp").exists()


def _models(ckpt: dict) -> tuple[Student, Student]:
    src = Student(StudentConfig(**ckpt["config"]))
    src.load_state_dict(ckpt["state_dict"])
    dst = Student(StudentConfig(**ckpt["config"]))
    return src, dst


def test_fit_does_not_train_on_held_out_batches(tmp_path, monkeypatch):
    _, ckpt = _source(tmp_path)
    src, dst = _models(ckpt)
    eval_ids = [1, 2, 3, 4]
    train_id = 11
    batches = [torch.full((1, 8), i, dtype=torch.long) for i in eval_ids]
    batches.append(torch.full((1, 8), train_id, dtype=torch.long))

    # _capture_ffn_io sees every batch, in call order: eval, train steps, eval.
    seen: list[int] = []
    original = ffn_init._capture_ffn_io

    def spy(model, tokens):
        seen.append(int(tokens.flatten()[0]))
        return original(model, tokens)

    monkeypatch.setattr(ffn_init, "_capture_ffn_io", spy)
    steps = 3
    report = ffn_init.fit_ffn_layerwise(src, dst, iter(batches), steps=steps, lr=1e-2)

    n_eval = len(eval_ids)
    assert seen[:n_eval] == eval_ids
    assert seen[n_eval : n_eval + steps] == [train_id] * steps
    assert seen[n_eval + steps :] == eval_ids
    assert len(report) == LAYERS
    assert set(report[0]) == {"layer", "rel_mse_before", "rel_mse_after"}


def test_fit_measures_on_all_batches_when_nothing_can_be_held_out(tmp_path, monkeypatch):
    # Fewer than FIT_EVAL_BATCHES + 1 batches: the fit has to reuse the eval batches.
    _, ckpt = _source(tmp_path)
    src, dst = _models(ckpt)
    ids = [1, 2]
    batches = [torch.full((1, 8), i, dtype=torch.long) for i in ids]

    seen: list[int] = []
    original = ffn_init._capture_ffn_io

    def spy(model, tokens):
        seen.append(int(tokens.flatten()[0]))
        return original(model, tokens)

    monkeypatch.setattr(ffn_init, "_capture_ffn_io", spy)
    ffn_init.fit_ffn_layerwise(src, dst, iter(batches), steps=2, lr=1e-2)
    assert seen[:2] == ids  # measured on all batches
    assert seen[2:4] == ids  # trained on them too
    assert seen[4:] == ids
