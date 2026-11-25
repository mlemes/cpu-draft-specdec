# Spec: FFN variants for experiment b10

Status 2026-11-24. Design: `experiments/stage-b-distillation/b10-kan-ffn/AGENTS.md` and `configs/arms.json`; proposal section 8, "Design change 2026-11-24".

## 1. `src/train/student.py`

New `StudentConfig` fields, defaults keep the current D-S model unchanged:

| Field | Default | Meaning |
|---|---|---|
| `ffn_type` | `"swiglu"` | one of `swiglu`, `mlp`, `mlp_rational`, `gr_kan`, `bspline_kan` |
| `rational_groups` | 8 | channel groups sharing one rational function |
| `rational_m` | 5 | numerator degree |
| `rational_n` | 4 | denominator degree |
| `kan_grid` | 5 | B-spline intervals G |
| `kan_order` | 3 | B-spline order k |
| `kan_range` | 2.0 | grid covers [-kan_range, kan_range] |

`ffn` is the hidden width for every type. An unknown `ffn_type` raises `ValueError` when `Student` is built. A config dict saved before this change (no new keys) still builds the same model, and SwiGLU parameter names stay `layers.{i}.mlp.{gate,up,down}_proj.weight`.

`build_ffn(cfg) -> nn.Module` returns the block placed at `StudentBlock.mlp`; input and output shape `(..., hidden)`. Parameter names under `mlp.`:

| Type | Forward | Parameters |
|---|---|---|
| `swiglu` | `down(silu(gate(x)) * up(x))` | unchanged |
| `mlp` | `down(silu(up(x)))` | `up_proj.weight` [ffn, hidden], `down_proj.weight` [hidden, ffn] |
| `mlp_rational` | `down(act(up(x)))` | as `mlp`, plus `act` = `RationalGroup(ffn, init="silu")` |
| `gr_kan` | `down(act2(up(act1(x))))` | as `mlp`, plus `act1` = `RationalGroup(hidden, init="identity")`, `act2` = `RationalGroup(ffn, init="silu")` |
| `bspline_kan` | `kan2(norm(kan1(x)))` | `kan1` = `KANLinear(hidden, ffn)`, `norm` = `nn.LayerNorm(ffn, elementwise_affine=False)`, `kan2` = `KANLinear(ffn, hidden)` |

No biases anywhere.

**`RationalGroup(channels, groups, m, n, init)`.** Parameters `numerator` [groups, m+1] (coefficients a_0..a_m) and `denominator` [groups, n] (b_1..b_n). Elementwise f(x) = Σ_{i=0..m} a_i x^i / (1 + |Σ_{j=1..n} b_j x^j|) (safe Padé, KAT). Channel c (last dimension) uses group c // (channels / groups). `channels % groups != 0` raises `ValueError`. `init="identity"`: a_1 = 1, all else 0. `init="silu"`: coefficients such that max |f(x) − silu(x)| ≤ 0.05 for x in [−4, 4], deterministic (independent of the global RNG). Evaluate in float32.

**`bspline_basis(x, grid, order, rng) -> Tensor`** (module-level function). Output shape `x.shape + (grid + order,)`. Uniform knots with spacing h = 2·rng/grid: t_i = −rng + (i − order)·h for i = 0..grid + 2·order; Cox–de Boor recursion from order 0 (half-open intervals [t_i, t_{i+1})) to `order`. For x in [−rng, rng) the bases are ≥ 0 and sum to 1; for x < t_0 or x ≥ t_last all bases are 0.

**`KANLinear(in_f, out_f, grid, order, rng)`.** Parameters `base_weight` [out_f, in_f] and `spline_weight` [out_f, in_f·(grid+order)]. Forward: `F.linear(silu(x), base_weight) + F.linear(bspline_basis(x).reshape(*x.shape[:-1], in_f*(grid+order)), spline_weight)`; input j owns basis columns j·(grid+order) … (j+1)·(grid+order) − 1. Parameters per layer = in_f·out_f·(grid+order+1). Initialization: finite, `base_weight` with the standard `Student` init scale, `spline_weight` small (std ≤ 0.02).

Parameter counts must equal `experiments/stage-b-distillation/b10-kan-ffn/configs/budget.py` (`ffn_params`) for every arm in `arms.json`.

## 2. `src/train/ffn_init.py`

Builds the init checkpoint for one arm from a trained source checkpoint (the format `_save_checkpoint` in `src/train/train.py` writes).

`python -m train.ffn_init --src SRC.pt --config JSON --out OUT.pt [--seed 0] [--fit --fit-data SPEC --fit-tokens N --fit-steps S --fit-lr LR --device cpu|cuda]`

- New config = source config updated with `--config` (a JSON object). Only `ffn_type`, `ffn`, `rational_*`, `kan_*` keys are allowed; others, invalid JSON, a missing source, or an existing `--out` exit 2 with a message on stderr and write nothing.
- Builds `Student(new_cfg)` with `torch.manual_seed(seed)`, copies every source tensor whose name has no `.mlp.` segment (a shape mismatch exits 2), and always keeps the freshly initialized FFN (so `--config '{}'` gives D-S with a re-initialized FFN).
- Output: same keys as the source checkpoint, with `config` and `state_dict` replaced (float32, CPU), `vprime` copied, `step` 0, plus `ffn_init` = {`src`, `src_sha256`, `overrides`, `seed`, `fit`}. `train.py --init OUT.pt` must load it unchanged.
- `--fit`: `SPEC` uses the `--data` syntax of `train.py` (`dir:weight,...`). Runs the source model on teacher-data token batches (no gradients); for each layer l, the input is the output of the source's `post_attention_layernorm` and the target is the source FFN output. Each new FFN l is trained with Adam on mean squared error for S steps. Writes `fit` = list of per-layer {`layer`, `rel_mse_before`, `rel_mse_after`} where rel_mse = MSE / mean(target²).
- Importable core: `fit_ffn_layerwise(src: Student, dst: Student, batches: Iterable[Tensor[B, T] int64], steps: int, lr: float) -> list[dict]` with the same report, no file I/O. Requires equal `hidden` and `n_layers`.

## 3. `src/models/export_student_gguf.py`

A checkpoint whose `config.ffn_type` is present and not `swiglu` exits 2 with a message naming `ffn_type`, and writes no output. (llama.cpp has no graph for these FFNs; b10 builds one only if the gate passes.)

## 4. `experiments/stage-b-distillation/b10-kan-ffn/configs/run_b10.sh`

`run_b10.sh <arm> [...]` for the arm IDs in `arms.json` except `ds-ref`. Same variables and training flags as `../../b6-deeper-student/configs/run_b6.sh` (b5 data mix, 96M tokens, seed 0), with `DSREF` defaulting to `$HDD/runs/b2/kd-pruned-prose45json23-20261122T105044Z/final.pt`. `ds-cont` trains from `DSREF` directly; other arms first run `ffn_init` into `$HDD/runs/b10/init/<arm>.pt` (reused when present) with the arm's config (`kan-bspline-fit` adds `--fit` over the same data, 4M tokens). Run directories `$HDD/runs/b10/<arm>-<UTC>`.

## 5. Constraints

- Python 3, torch, numpy; no new dependencies. Tests use pytest with `pytest.importorskip("torch")`.
- Run as `PYTHONPATH=src python -m ...`; the training venv is `/mnt/hdd1/cpu-draft-specdec/venv-train`.
