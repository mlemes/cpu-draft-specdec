"""Build the FFN-variant initialization checkpoint for experiment b10.

Takes a trained source checkpoint (the format ``_save_checkpoint`` in ``train.py`` writes),
keeps every tensor that is not part of an FFN and replaces all FFNs with freshly
initialized ones of a different ``ffn_type``. With ``--fit`` each new FFN is also trained
layer by layer to reproduce the source FFN output on source hidden states; this is the
``kan-bspline-fit`` initialization of ``experiments/stage-b-distillation/b10-kan-ffn``.
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

if __name__ == "__main__":
    # Run as a script: drop src/train from sys.path so `train` resolves to the package, not this file.
    _here = str(_Path(__file__).resolve().parent)
    _sys.path[:] = [p for p in _sys.path if str(_Path(p or ".").resolve()) != _here]
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import argparse
from dataclasses import asdict
from collections.abc import Iterable
import json
import math
import os
from pathlib import Path

import torch
import torch.nn.functional as F

from train.data import TeacherBatcher
from train.student import FFN_TYPES, Student, StudentConfig
from train.train import UsageError, _parse_data_spec, _sha256_file
from train.vprime import vprime_map

# Only these StudentConfig fields may be changed by --config. Attention, norm and
# embedding geometry must stay identical, otherwise no tensor could be copied.
ALLOWED_OVERRIDES = (
    "ffn_type",
    "ffn",
    "rational_groups",
    "rational_m",
    "rational_n",
    "kan_grid",
    "kan_order",
    "kan_range",
)

# Any parameter of an FFN block lives under "<layer>.mlp."; everything else is copied.
FFN_MARKER = ".mlp."

DEFAULT_FIT_TOKENS = 4_000_000
DEFAULT_FIT_STEPS = 1954  # two epochs over 4M tokens at 4 x 1024 tokens per batch
DEFAULT_FIT_LR = 3e-3
DEFAULT_FIT_BATCH_SEQS = 4
FIT_EVAL_BATCHES = 4


def _capture_ffn_io(model: Student, tokens: torch.Tensor) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Run ``model`` without gradients and return per-layer (FFN input, FFN output)."""
    captured: list[tuple[torch.Tensor, torch.Tensor]] = []

    def _hook(module: torch.nn.Module, args: tuple, output: torch.Tensor) -> None:
        captured.append((args[0].detach(), output.detach()))

    handles = [layer.mlp.register_forward_hook(_hook) for layer in model.layers]
    try:
        with torch.no_grad():
            model(tokens)
    finally:
        for handle in handles:
            handle.remove()
    if len(captured) != len(model.layers):
        raise RuntimeError(
            f"captured {len(captured)} FFN outputs for {len(model.layers)} layers"
        )
    return captured


def _rel_mse(pred: torch.Tensor, target: torch.Tensor) -> tuple[float, float]:
    """Return (squared-error sum, target squared sum) in float32."""
    pred_f = pred.float()
    target_f = target.float()
    return (
        float((pred_f - target_f).pow(2).sum().item()),
        float(target_f.pow(2).sum().item()),
    )


def fit_ffn_layerwise(
    src: Student,
    dst: Student,
    batches: Iterable[torch.Tensor],
    steps: int,
    lr: float,
) -> list[dict]:
    """Fit every ``dst.layers[l].mlp`` to the source FFN output of ``src``.

    ``batches`` yields int64 token tensors [B, T]. The first ``FIT_EVAL_BATCHES`` batches
    are held out and never trained on: they are only used to measure rel_mse, both before
    and after the fit. Up to ``steps`` further batches are kept and the fit cycles over
    that list, so each layer sees exactly ``steps`` Adam steps on mean squared error. If
    fewer than ``FIT_EVAL_BATCHES + 1`` batches are available there is nothing to hold
    out, so the fit runs on and rel_mse is measured on every available batch. Returns one
    ``{layer, rel_mse_before, rel_mse_after}`` record per layer, with
    rel_mse = MSE / mean(target**2) measured before and after the fit.
    """
    if src.cfg.hidden != dst.cfg.hidden or src.cfg.n_layers != dst.cfg.n_layers:
        raise ValueError(
            "source and destination must share hidden and n_layers, got "
            f"{src.cfg.hidden}/{src.cfg.n_layers} and {dst.cfg.hidden}/{dst.cfg.n_layers}"
        )
    if steps <= 0:
        raise ValueError(f"steps must be positive, got {steps}")
    if not math.isfinite(lr) or lr <= 0.0:
        raise ValueError(f"lr must be finite and > 0, got {lr}")

    iterator = iter(batches)
    eval_batches: list[torch.Tensor] = []
    for batch in iterator:
        eval_batches.append(torch.as_tensor(batch, dtype=torch.long))
        if len(eval_batches) >= FIT_EVAL_BATCHES:
            break

    train_batches: list[torch.Tensor] = []
    for batch in iterator:
        train_batches.append(torch.as_tensor(batch, dtype=torch.long))
        if len(train_batches) >= steps:
            break

    if not train_batches:
        # Fewer than FIT_EVAL_BATCHES + 1 batches: nothing left to hold out.
        train_batches = eval_batches
    if not train_batches:
        raise ValueError("no token batches were given")

    device = next(src.parameters()).device
    n_layers = src.cfg.n_layers
    src.eval()
    dst.eval()

    def measure() -> list[float]:
        sq = [0.0] * n_layers
        sc = [0.0] * n_layers
        for tokens in eval_batches:
            captured = _capture_ffn_io(src, tokens.to(device))
            for layer in range(n_layers):
                hidden, target = captured[layer]
                with torch.no_grad():
                    pred = dst.layers[layer].mlp(hidden)
                layer_sq, layer_sc = _rel_mse(pred, target)
                sq[layer] += layer_sq
                sc[layer] += layer_sc
        return [sq[layer] / max(sc[layer], 1e-12) for layer in range(n_layers)]

    before = measure()

    optimizers = [
        torch.optim.Adam(list(dst.layers[layer].mlp.parameters()), lr=lr)
        for layer in range(n_layers)
    ]
    for step in range(steps):
        tokens = train_batches[step % len(train_batches)].to(device)
        captured = _capture_ffn_io(src, tokens)
        for layer in range(n_layers):
            hidden, target = captured[layer]
            loss = F.mse_loss(dst.layers[layer].mlp(hidden).float(), target.float())
            optimizers[layer].zero_grad(set_to_none=True)
            loss.backward()
            optimizers[layer].step()

    after = measure()
    return [
        {"layer": layer, "rel_mse_before": before[layer], "rel_mse_after": after[layer]}
        for layer in range(n_layers)
    ]


def _parse_overrides(raw_json: str) -> dict[str, object]:
    try:
        parsed = json.loads(raw_json)
    except json.JSONDecodeError as exc:
        raise UsageError(f"Invalid --config JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise UsageError("--config must be a JSON object")
    unknown = sorted(set(parsed) - set(ALLOWED_OVERRIDES))
    if unknown:
        raise UsageError(
            f"Unsupported --config keys {unknown}; allowed keys are "
            f"{', '.join(ALLOWED_OVERRIDES)}"
        )
    return parsed


def _load_source(path: Path) -> dict:
    if not path.is_file():
        raise UsageError(f"Source checkpoint not found: {path}")
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:  # noqa: BLE001 - surfaced as a usage error
        raise UsageError(f"Failed to load source checkpoint {path}: {exc}") from exc
    if not isinstance(ckpt, dict) or not {"config", "state_dict", "vprime"} <= set(ckpt):
        raise UsageError(
            f"Malformed source checkpoint {path}: expected 'config', 'state_dict', 'vprime'"
        )
    if not isinstance(ckpt["config"], dict) or not isinstance(ckpt["state_dict"], dict):
        raise UsageError(f"Malformed source checkpoint {path}: config and state_dict must be dicts")
    return ckpt


def build_init_checkpoint(
    src_ckpt: dict,
    overrides: dict[str, object],
    seed: int,
    src_path: str,
    src_sha256: str,
) -> tuple[dict, Student]:
    """Return (checkpoint dict, fresh Student with copied non-FFN tensors)."""
    try:
        src_cfg = StudentConfig(**src_ckpt["config"])
    except (TypeError, ValueError) as exc:
        raise UsageError(f"Source checkpoint config is not a valid StudentConfig: {exc}") from exc
    new_cfg = StudentConfig(**{**asdict(src_cfg), **overrides})
    if new_cfg.ffn_type not in FFN_TYPES:
        raise UsageError(
            f"Unknown ffn_type {new_cfg.ffn_type!r}; expected one of {', '.join(FFN_TYPES)}"
        )

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        try:
            dst = Student(new_cfg)
        except ValueError as exc:
            raise UsageError(f"Cannot build the requested Student: {exc}") from exc

    # Student's own init draws from the global RNG; the copy below overwrites every
    # non-FFN tensor, and the FFN keeps exactly this freshly initialized state.
    dst_state = {k: v.detach().clone() for k, v in dst.state_dict().items()}
    src_state = src_ckpt["state_dict"]

    # Check the whole non-FFN tensor set before copying anything, so a source that does
    # not fit the new geometry fails cleanly instead of being partially copied.
    missing = sorted(
        name for name in dst_state if FFN_MARKER not in name and name not in src_state
    )
    if missing:
        raise UsageError(
            "Source checkpoint is missing non-FFN tensors required by the new model: "
            + ", ".join(missing)
        )
    unexpected = sorted(
        name for name in src_state if FFN_MARKER not in name and name not in dst_state
    )
    if unexpected:
        raise UsageError(
            "Source checkpoint has non-FFN tensors with no counterpart in the new model: "
            + ", ".join(unexpected)
        )

    copied = 0
    for name, tensor in src_state.items():
        if FFN_MARKER in name:
            continue
        if tuple(tensor.shape) != tuple(dst_state[name].shape):
            raise UsageError(
                f"Shape mismatch for {name!r}: source {tuple(tensor.shape)} != "
                f"new model {tuple(dst_state[name].shape)}"
            )
        dst_state[name] = tensor.detach().to(torch.float32).cpu().clone()
        copied += 1
    if copied == 0:
        raise UsageError("Source checkpoint has no non-FFN tensors to copy")
    dst.load_state_dict(dst_state, strict=True)

    out = dict(src_ckpt)
    out["config"] = asdict(new_cfg)
    out["state_dict"] = {k: v.detach().to(torch.float32).cpu() for k, v in dst.state_dict().items()}
    out["vprime"] = [int(token) for token in src_ckpt["vprime"]]
    out["step"] = 0
    out["ffn_init"] = {
        "src": src_path,
        "src_sha256": src_sha256,
        "overrides": dict(overrides),
        "seed": int(seed),
        "fit": None,
    }
    return out, dst


def _save_checkpoint(obj: dict, out_path: Path) -> None:
    """Write ``obj`` to ``out_path`` atomically: ``--out`` never exists half-written."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_name(out_path.name + ".tmp")
    try:
        torch.save(obj, tmp_path)
        os.replace(tmp_path, out_path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a b10 FFN-variant init checkpoint from a trained Student checkpoint."
    )
    parser.add_argument("--src", required=True, help="Trained source checkpoint (.pt)")
    parser.add_argument("--config", required=True, help="JSON object of FFN overrides")
    parser.add_argument("--out", required=True, help="Output checkpoint (.pt, must not exist)")
    parser.add_argument("--seed", type=int, default=0, help="Seed for the fresh FFN init")
    parser.add_argument("--fit", action="store_true", help="Fit each new FFN to the source FFN")
    parser.add_argument(
        "--fit-data",
        default=None,
        help="Teacher shard dirs as DIR:WEIGHT,... (same syntax as train.py --data)",
    )
    parser.add_argument(
        "--fit-tokens",
        type=int,
        default=DEFAULT_FIT_TOKENS,
        help=f"Token budget for the fit data (default {DEFAULT_FIT_TOKENS})",
    )
    parser.add_argument(
        "--fit-steps",
        type=int,
        default=DEFAULT_FIT_STEPS,
        help=f"Adam steps per layer for the fit (default {DEFAULT_FIT_STEPS})",
    )
    parser.add_argument(
        "--fit-lr",
        type=float,
        default=DEFAULT_FIT_LR,
        help=f"Adam learning rate for the fit (default {DEFAULT_FIT_LR})",
    )
    parser.add_argument(
        "--fit-batch-seqs",
        type=int,
        default=DEFAULT_FIT_BATCH_SEQS,
        help=f"Sequences per fit batch (default {DEFAULT_FIT_BATCH_SEQS})",
    )
    parser.add_argument("--device", default="cpu", help="Torch device for the fit (default: cpu)")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        src_path = Path(args.src)
        out_path = Path(args.out)
        if out_path.exists():
            raise UsageError(f"Output checkpoint already exists: {out_path}")
        ckpt = _load_source(src_path)
        overrides = _parse_overrides(args.config)
        if args.fit_tokens <= 0:
            raise UsageError(f"--fit-tokens must be positive, got {args.fit_tokens}")
        if args.fit_steps <= 0:
            raise UsageError(f"--fit-steps must be positive, got {args.fit_steps}")
        if not math.isfinite(args.fit_lr) or args.fit_lr <= 0.0:
            raise UsageError(f"--fit-lr must be finite and > 0, got {args.fit_lr}")
        if args.fit_batch_seqs <= 0:
            raise UsageError(f"--fit-batch-seqs must be positive, got {args.fit_batch_seqs}")
        if args.fit and not args.fit_data:
            raise UsageError("--fit requires --fit-data")

        out, dst = build_init_checkpoint(
            ckpt,
            overrides,
            args.seed,
            str(src_path),
            _sha256_file(src_path),
        )

        if args.fit:
            data_specs = _parse_data_spec(args.fit_data)
            device = torch.device(args.device)
            src_model = Student(StudentConfig(**ckpt["config"]))
            src_model.load_state_dict(ckpt["state_dict"])
            src_model.to(device)
            dst.to(device)

            vmap_np = vprime_map([int(t) for t in ckpt["vprime"]], src_model.cfg.n_vocab_in)
            batcher = TeacherBatcher(
                data_specs, vmap_np, batch_seqs=args.fit_batch_seqs, seed=int(args.seed)
            )

            def token_batches() -> Iterable[torch.Tensor]:
                seen = 0
                for batch in batcher:
                    tokens = batch["tokens"]
                    seen += int(tokens.shape[0] * tokens.shape[1])
                    yield tokens
                    if seen >= args.fit_tokens:
                        return

            report = fit_ffn_layerwise(
                src_model, dst, token_batches(), args.fit_steps, args.fit_lr
            )
            out["state_dict"] = {
                k: v.detach().to(torch.float32).cpu() for k, v in dst.state_dict().items()
            }
            out["ffn_init"]["fit"] = report

        _save_checkpoint(out, out_path)
        print(
            f"wrote {out_path} (ffn_type={out['config']['ffn_type']}, ffn={out['config']['ffn']}, "
            f"fit={'yes' if args.fit else 'no'})"
        )
        return 0
    except UsageError as exc:
        print(f"error: {exc}", file=_sys.stderr)
        return 2
    except Exception as exc:  # noqa: BLE001 - mirrors train.py's exit-code contract
        print(f"runtime error: {exc}", file=_sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
