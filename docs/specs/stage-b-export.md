# Stage B: student export to llama.cpp and training fixes

Status: specification, 2026-11-21. Goal: run a trained student as a llama.cpp draft for Qwen3-8B, so end-to-end speculative speedups can be measured, and fix two trainer defects before longer runs.

## 1. llama.cpp V′ output head (patch)

Work happens in a standalone clone of the pinned llama.cpp (`29b52d398`), branch `vprime`. The Architect exports the result to `src/patches/llama.cpp-vprime.patch` in this repository and applies it to `third_party/llama.cpp`.

Scope: Qwen3 architecture only (`src/models/qwen3.cpp`), plus the tensor-name tables in `src/llama-arch.{h,cpp}` and the tensor member in `src/llama-model.h`.

Behavior:

- New optional tensor `output_vmap.weight`, type I32, shape `{n_vocab}`. Entry `t` holds the row of `output.weight` that produces token `t`'s logit, or the value `n_vprime` when token `t` is not in V′.
- When `output_vmap.weight` is present, `output.weight` has shape `{n_embd, n_vprime}`, with `n_vprime` read from that tensor's metadata in the file (`n_vprime < n_vocab`). Its absence keeps today's behavior exactly (shape `{n_embd, n_vocab}`, tied fallback to `token_embd`).
- The graph computes the `n_vprime` small logits with the existing `build_lora_mm`, then produces the full `{n_vocab, n_tokens}` logits tensor: `logit[t] = small[vmap[t]]` when `vmap[t] < n_vprime`, and `-INFINITY` otherwise. One way: append one `-INFINITY` row to the small logits (for example `ggml_scale_bias` of a one-row view with scale 0 and bias `-INFINITY`, or `ggml_fill`), transpose so each vocabulary entry is a row, `ggml_get_rows` with `output_vmap`, transpose back, `ggml_cont`. Any equivalent graph is acceptable if it works on the CPU and CUDA backends.
- `res->t_logits` is the full-vocabulary tensor, so the server, samplers and speculative loop need no change.
- `llama-quantize` must keep `output_vmap.weight` as I32 (it is one-dimensional; verify it is not converted).

## 2. Student exporter (`src/models/export_student_gguf.py`)

```
python3 src/models/export_student_gguf.py --ckpt CKPT.pt --ref-gguf REF.gguf --out OUT.gguf [--dtype f32|f16]
```

- `CKPT.pt` is a trainer checkpoint (`torch.load(..., weights_only=False)`), a dict with `config` (the `StudentConfig` fields), `state_dict` (keys `embed_tokens.weight`, `layers.{i}.input_layernorm.weight`, `layers.{i}.self_attn.{q,k,v,o}_proj.weight`, `layers.{i}.self_attn.{q,k}_norm.weight`, `layers.{i}.post_attention_layernorm.weight`, `layers.{i}.mlp.{gate,up,down}_proj.weight`, `norm.weight`, `lm_head.weight`) and `vprime` (list of `n_vprime` token IDs; row `j` of `lm_head.weight` is token `vprime[j]`).
- `REF.gguf` is a Qwen3 GGUF with the same tokenizer (for example Qwen3-0.6B-Q8_0). Copy every `tokenizer.*` key from it. Exit 2 if its token count differs from `config.n_vocab_in`.
- Write `general.architecture = qwen3`, `general.name`, and `qwen3.context_length` (= `max_seq`), `qwen3.embedding_length`, `qwen3.block_count`, `qwen3.feed_forward_length`, `qwen3.attention.head_count`, `qwen3.attention.head_count_kv`, `qwen3.attention.key_length`, `qwen3.attention.value_length` (= `head_dim`), `qwen3.rope.freq_base`, `qwen3.attention.layer_norm_rms_epsilon`.
- Tensor names: `token_embd.weight`, `blk.{i}.attn_norm.weight`, `blk.{i}.attn_{q,k,v,output}.weight`, `blk.{i}.attn_{q,k}_norm.weight`, `blk.{i}.ffn_norm.weight`, `blk.{i}.ffn_{gate,up,down}.weight`, `output_norm.weight`, `output.weight`, `output_vmap.weight`. Weight arrays keep the PyTorch `[out, in]` layout, as `convert_hf_to_gguf.py` does for Qwen3 (no Q/K permutation). Norm weights are always F32; matrices use `--dtype` (default f16); `output_vmap.weight` is I32 built from `vprime`.
- Use the `gguf` package from `third_party/llama.cpp/gguf-py` (insert it on `sys.path`).
- Exit codes: 0 success; 1 runtime failure; 2 usage or input error (missing file, malformed checkpoint, vocabulary mismatch, duplicate or out-of-range V′ IDs). The output file is written to `OUT.gguf.tmp` and renamed on success.

## 3. Trainer fixes

### 3.1 Gradient accumulation weighting (`src/train/train.py`)

Today each microbatch's loss is a mean over its own positions and is divided by `grad_accum`, so microbatches with fewer counted positions get too much weight. New public function:

```
step_backward(model, microbatches: list[dict], loss: str, vmap_t: torch.Tensor | None = None) -> float
```

- `microbatches` are batcher dicts (`tokens`, `topk_vidx`, `topk_p`, `loss_mask`) already on the model's device; `loss` is `kd`, `hard` or `tv`.
- It runs forward and backward on each microbatch in turn (never all at once) and leaves in `.grad` exactly the gradient of the loss pooled over every counted position of all microbatches, that is, `sum_i (n_i / N) * loss_i` where `n_i` is the number of positions the loss counts in microbatch `i` (`kd`, `tv`: `loss_mask`; `hard`: mask and next-token in V′, excluding the last position) and `N = sum n_i`. A step with `N = 0` contributes zero gradient.
- Returns the pooled loss value. The training loop uses it; logged `loss` is the pooled value.

### 3.2 Sequence length (`src/train/data.py`)

`TeacherBatcher` raises `ValueError` at construction when the teacher directories do not all share one `seq_len`.

## 4. Acceptance

Run by the Architect on this machine (GPU, built llama.cpp). Paths come from environment variables with defaults under `/mnt/hdd1/cpu-draft-specdec`.

1. Logit parity: the exported f32 student served by the patched `llama-server` on the CPU (`-ngl 0`) returns, for the next position after each of several prompts (`n_predict 1`, `n_probs 20`, `temperature -1`), a top-1 token equal to the PyTorch student's and log-probabilities within 0.02 of PyTorch's softmax over V′ for shared IDs, with at least 18 of 20 IDs shared.
2. Every ID returned with `n_probs 100` is in V′.
3. An unmodified Qwen3 GGUF still loads and generates with the patched build.
4. `llama-quantize` to Q8_0 succeeds and keeps `output_vmap.weight` as I32.
5. Speculative correctness: Qwen3-8B on the GPU with the Q8_0 student as a CPU draft (`--spec-type draft-simple`, greedy) produces exactly the same tokens as the target alone for at least three prompts of 64 tokens, with drafted and accepted counts above zero.
6. Exporter usage errors exit 2 without leaving `OUT.gguf` behind.
7. `step_backward` gradients match a single pooled computation over the concatenated microbatches (equal sequence lengths, differing mask counts) for `kd` and `hard`; `TeacherBatcher` rejects mixed `seq_len`.
