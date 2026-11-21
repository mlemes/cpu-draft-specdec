# Stage B tooling specification

Status: draft 2026-11-21. Covers the pilot ("short Stage B check"): teacher data (b1), dense student D-S training and offline acceptance (b2). Decisions taken after Stage A: distillation first; the CPU/GPU overlap loop is deferred (Stage A summary, a7).

Large files live under `$HDD = /mnt/hdd1/cpu-draft-specdec`. Evidence labels as in `AGENTS.md`.

## 1. Training corpus (`src/data/build_train_corpus.py`)

Runs locally (pyarrow). Writes one JSONL file per domain to `$HDD/corpora/train/<domain>.jsonl`, one document per line: `{"id": str, "domain": str, "text": str}`, plus `manifest.json` (sources, licenses, document counts, character counts, SHA-256).

| Domain | Source | Exclusions |
|---|---|---|
| `general` | WikiText-103 train (CC-BY-SA-3.0), detokenized as in `prepare_prompts.py` | none needed: prompts use the test split |
| `pt_general` | Portuguese Wikipedia 20231101 (`wikimedia/wikipedia`, CC-BY-SA-4.0), first two shards | none |
| `pt_fin` | Copom minutes 232–261 (BCB open data) | 262–282 are held out for evaluation |
| `json` | Hermes json-mode rows after the first 100 (seeded shuffle as in `prepare_prompts.py`), rendered with the Qwen3 chat template plus the answer | first 100 rows are evaluation prompts |

Character budgets per domain are CLI options; documents are kept whole and in a seeded random order.

## 2. Teacher scorer (`src/teacher/teacher_score.cpp`)

C++17 against the pinned libllama (same include/link flags as `vocab_coverage` in `src/measure/Makefile`; new Makefile target `teacher_score` in `src/teacher/Makefile`).

### CLI

```
teacher_score --model GGUF --input DOCS.jsonl --out DIR
              [--seq-len 1024] [--k 32] [--max-tokens N] [--shard-seqs 1024]
              [--ngl 99] [--threads 6] [--parse-special] [--dump-full N]
```

- `--input`: JSONL documents; only the `text` field is used (string with JSON escapes, including `\uXXXX` and surrogate pairs). A malformed line is an error (exit 1) naming the line number.
- Tokenize each document with `llama_tokenize(vocab, text, add_special=false, parse_special=<flag>)`. Append the vocabulary EOS token after each document. Concatenate all documents in file order and cut the stream into consecutive sequences of exactly `seq-len` tokens; drop the final partial sequence. Stop after `max-tokens` tokens (rounded down to whole sequences) if given.
- Score each sequence from an empty KV cache (`llama_memory_clear`), requesting logits at every position. Position `t`'s distribution is the model's prediction for token `t+1`, after consuming tokens `0..t`.
- For each position compute log-softmax over the **full** vocabulary (numerically stable: subtract the max; float exponentials summed in blocks of ≤ 4096 into a double — relative error of the sum ≤ ~2.5e-4, below the fp16 storage step; accepted 2026-11-21), then the top-`k` tokens by logit, sorted by descending log-probability; ties broken by smaller token ID.
- Parallelize the per-position top-k/log-softmax across `--threads` CPU threads.

### Output layout in `DIR` (must not exist, or must hold a resumable run)

Per shard `i` (zero-padded to 5 digits), with `n` sequences (≤ `shard-seqs`):

| File | dtype | shape |
|---|---|---|
| `shard-iiiii.tokens.u32` | uint32 little-endian | `[n, seq_len]` |
| `shard-iiiii.ids.u32` | uint32 | `[n, seq_len, k]` |
| `shard-iiiii.logp.f16` | IEEE half | `[n, seq_len, k]` |

`index.json`, rewritten atomically (write temp file, rename) after every completed shard:

```json
{"format": "teacher-topk-v1", "model": "<path>", "model_sha256": "<hex>",
 "input": "<path>", "input_sha256": "<hex>", "seq_len": 1024, "k": 32,
 "n_vocab": 151936, "eos_id": 151645, "parse_special": false,
 "shards": [{"name": "shard-00000", "n_seq": 1024,
             "sha256": {"tokens": "...", "ids": "...", "logp": "..."}}],
 "n_seq": 1024, "n_tokens": 1048576, "complete": true,
 "elapsed_s": 0.0, "tokens_per_s": 0.0}
```

`complete` is false while running and true after a normal finish. Tail mass at a position is `1 − Σ exp(logp)`; it is not stored.

- **Resume:** if `DIR/index.json` exists with the same `model_sha256`, `input_sha256`, `seq_len`, `k` and `parse_special`, skip the sequences already in its shards and continue with the next shard number; files of an incomplete trailing shard not listed in the index are overwritten. Any mismatch is an error (exit 1).
- `--dump-full N`: also write `full.logp.f32`, the full-vocabulary log-softmax for the first `N` positions of sequence 0, shape `[N, n_vocab]` (for spot checks).
- Progress to stderr every shard: sequences, tokens, tokens/s.
- Exit codes: 0 success; 2 usage error (unknown option, missing required option, bad number, `k` < 1 or > n_vocab, `seq-len` < 2); 1 runtime error (model load, input parse, decode failure, I/O, resume mismatch). On SIGINT/SIGTERM finish the current sequence, do not write a partial shard, leave `complete: false`, exit 130/143.

## 3. Student, initialization, trainer and offline acceptance (b2)

Python 3.12 with PyTorch (CUDA) and numpy, run from the local venv `$HDD/venv-train` (`$HDD/venv-train/bin/python`). Package `src/train/`. No `transformers` dependency.

### 3.1 V′ (`src/train/vprime.py`)

`load_vprime(ranking_json, size=32768, forced=(151643, 151645)) -> list[int]`: the first `size` token IDs of the a5 frequency ranking (`experiments/stage-a-feasibility/a5-vocab-coverage/results/a5-20261121T011928Z/ranking.json`, a JSON list of token IDs, most frequent first); for the m forced IDs absent from that prefix, drop the m lowest-ranked kept IDs and append the missing forced IDs in the given order (forced IDs already present stay in place). Returns exactly `size` unique IDs. `vprime_map(vprime, n_vocab) -> np.ndarray[int32]` maps target ID → V′ index, −1 if outside.

### 3.2 Student (`src/train/student.py`)

`StudentConfig` dataclass: `n_vocab_in=151936, n_vprime=32768, hidden=512, n_layers=12, n_heads=4, n_kv_heads=1, head_dim=128, ffn=1536, rope_theta=1e6, rms_eps=1e-6, max_seq=2048`.

`Student(cfg)` (torch.nn.Module), Qwen3 layout: token embedding [n_vocab_in, hidden]; per layer: RMSNorm → attention (q/k/v/o projections without bias, per-head RMSNorm on q and k over head_dim (Qwen3 QK-norm), RoPE (rotate-half convention as in HF Qwen3) on q and k, causal GQA via `torch.nn.functional.scaled_dot_product_attention(is_causal=True)`) → residual; RMSNorm → SwiGLU FFN (`down(silu(gate(x)) * up(x))`) → residual; final RMSNorm; untied head [n_vprime, hidden] without bias. `forward(tokens[B, T]) -> logits[B, T, n_vprime]` (float32 or the autocast dtype). Parameter names follow HF Qwen3 (`embed_tokens`, `layers.{i}.self_attn.{q,k,v,o}_proj`, `self_attn.{q,k}_norm`, `input_layernorm`, `post_attention_layernorm`, `mlp.{gate,up,down}_proj`, `norm`, `lm_head`) so export maps 1:1.

### 3.3 Pruned initialization (`src/train/prune_init.py`)

CLI `prune_init.py --src $HDD/hf/Qwen3-0.6B --vprime-ranking RANKING --out CKPT.pt`. Reads `model.safetensors` (bf16) and builds a `Student` state dict:
- **Layers:** source layers `round(i * 27 / 11)` for i = 0..11 (keeps first and last).
- **Hidden dims (1024 → 512):** keep the 512 dims with the largest importance `mean over layers of |input_layernorm.weight|` × column L2 norm of `embed_tokens`; ties by smaller index; kept dims are ordered by ascending source index; the same dim set is applied to every hidden-sized axis (embedding columns, all norms over hidden, input dims of q/k/v/gate/up, output dims of o/down).
- **Heads:** source has 16 query heads, 8 KV heads (query head h uses KV head h // 2). Per layer pick the KV head with the largest `‖k_proj rows‖_F + ‖v_proj rows‖_F`; keep its two query heads plus the two query heads of the next KV head (cyclic); student K/V weights = mean of those two source KV heads' rows. q_norm/k_norm copied (k_norm from the chosen head is shared, it is per-dim only).
- **FFN (3072 → 1536):** per layer keep neurons with the largest `‖down_proj[:, j]‖ · ‖up_proj[j, :]‖`.
- **Head:** `lm_head` = embedding rows of the V′ IDs (source ties embeddings), restricted to the kept hidden dims.
- Saves `{"config": asdict(cfg), "state_dict": ..., "vprime": [...], "source_sha256": ...}`.

### 3.4 Data (`src/train/data.py`)

`TeacherShards(dir)` reads a teacher_score output directory via `index.json` and numpy memmaps. `TeacherBatcher(dirs_with_weights: list[tuple[str, float]], vprime_map: np.ndarray, batch_seqs: int, seed: int)` is an infinite iterable of dicts of torch tensors sampled across domains with the given mixture weights: `tokens[B, L] int64`, `topk_vidx[B, L, k] int64` (V′ index, −1 if outside V′), `topk_p[B, L, k] float32` (teacher probabilities `exp(logp)`), and `loss_mask[B, L] bool` (false at the last position of each sequence). Deterministic for a seed; epoch-style without replacement within a domain.

### 3.5 Losses (`src/train/losses.py`)

Signatures (tensors of any float dtype; reductions in float32 or higher): `kd_forward_kl(logits[B,T,V′], topk_vidx[B,T,k], topk_p[B,T,k], mask[B,T]) -> scalar`, `hard_ce(logits, tokens[B,T], vprime_map[n_vocab] (tensor, −1 outside), mask) -> scalar`, `tv_overlap(logits, topk_vidx, topk_p) -> (lower[B,T], upper[B,T])`. Entries with `topk_vidx = −1` are ignored.

With student log-probs `s = log_softmax(logits)` over V′ and teacher top-k restricted to V′ (`p_i`, `vidx_i ≥ 0`):
- `kd_forward_kl`: Σ_i p_i (log p_i − s[vidx_i]) + p_rest (log p_rest − log q_rest), where p_rest = 1 − Σ_i p_i (teacher mass outside top-k∩V′, clamped ≥ 1e-8) and q_rest = 1 − Σ_i exp(s[vidx_i]) (clamped ≥ 1e-8). Mean over masked positions.
- `hard_ce`: cross-entropy against the next token (`tokens[:, t+1]` mapped to V′); positions whose next token is outside V′ are excluded.
- `tv_overlap`: Σ_i min(p_i, exp(s[vidx_i])) — the exact top-k part of expected acceptance (lower bound); upper bound adds min(p_k, q_rest) where p_k is the smallest top-k teacher probability. Used for evaluation and for the TV arm (loss = 1 − lower bound).

### 3.6 Trainer (`src/train/train.py`)

CLI:
```
train.py --init CKPT.pt|scratch [--config JSON] --data DIR:WEIGHT[,DIR:WEIGHT...] --out RUN_DIR
         --loss kd|hard|tv [--tokens N] [--batch-seqs 16] [--grad-accum 1] [--seq-len L]
         [--lr 6e-4] [--min-lr-frac 0.1] [--warmup 200] [--weight-decay 0.1] [--seed 0]
         [--eval-dir DIR] [--eval-every-tokens N] [--ckpt-every-tokens N] [--device cuda] [--compile]
```
- `--init scratch` builds `Student(StudentConfig(**config))` with fresh init and V′ from `--vprime-ranking` (default: the a5 ranking path of 3.1); `--config` is a JSON object of `StudentConfig` overrides, given inline (starting with `{`) or as a path to a JSON file (also allowed with a checkpoint only if it matches the checkpoint's config, else exit 2). A checkpoint init loads config, weights and V′ from the file.
- `--seq-len` defaults to the teacher data's `seq_len`; if smaller, each teacher sequence is cut into its first `seq_len` positions (the teacher distribution at position t depends only on tokens ≤ t, so truncation is exact).
- Tokens processed per optimizer step = batch_seqs × grad_accum × seq_len; the run stops after `--tokens` (rounded up to whole steps). LR: linear warmup over `--warmup` steps, cosine to `min-lr-frac` × peak at the last step. AdamW (β 0.9/0.95, eps 1e-8), weight decay on 2-D weights except `embed_tokens` and norms; gradient clip 1.0; bf16 autocast on CUDA (fp32 on CPU); losses per 3.5 (`kd` = kd_forward_kl, `hard` = hard_ce, `tv` = 1 − mean tv lower bound over masked positions).
- Outputs in `RUN_DIR` (must not exist): `metrics.jsonl`, one line per step `{"step", "tokens", "loss", "lr", "grad_norm", "tokens_per_s"}` plus eval lines `{"step", "tokens", "eval": <eval_alpha summary>}` every `--eval-every-tokens` (and at the end) when `--eval-dir` is given; checkpoints `ckpt-<tokens>.pt` every `--ckpt-every-tokens` and `final.pt`, in the 3.3 checkpoint format (`config`, `state_dict` float32, `vprime`, plus `train_args`, `step`, `tokens`); `manifest.json` (args, git commit, teacher `index.json` SHA-256 per data dir, start/end time).
- Exit codes: 0 success, 2 usage error (bad `--data` spec, unknown loss, existing `RUN_DIR`, config mismatch), 1 runtime error. Deterministic given seed on CPU.

### 3.7 Offline acceptance (`src/train/eval_alpha.py`)

Evaluation set (`src/data/gen_eval_set.py`, run once on the GPU): for each of the 300 a4 prompts the deployed target (llama-server, greedy, 128 tokens) generates a continuation, returning for every generated position the target's top-32 log-probabilities from a plain softmax of the logits. Files `$HDD/eval/<target_tag>/eval_<domain>.jsonl`, one line per prompt: `{"id", "domain", "prompt_tokens": [int], "gen_tokens": [int], "top_ids": [[int]*32]*n, "top_logp": [[float]*32]*n}`; `top_ids[j]` is the target's distribution for `gen_tokens[j]` given `prompt_tokens + gen_tokens[:j]`.

CLI `eval_alpha.py --ckpt CKPT.pt --eval-dir DIR [--domains general,pt_fin,json] [--out JSON] [--device cuda] [--batch 16]`; also importable as `evaluate(model, vprime, eval_dir, domains, device) -> dict` (same summary) for the trainer. Domains whose file is absent are skipped. For each row, run the student on `prompt_tokens + gen_tokens[:-1]` (truncate from the left to the student's `max_seq`) and read its distribution at the positions that predict each generated token. Per domain, over generated positions:
- `greedy_agree`: fraction where the student argmax over V′, mapped to a target ID, equals `top_ids[j][0]` — per-position α for greedy speculative decoding;
- `tv_lower` / `tv_upper`: Σ min(p, q) over the target top-32 ∩ V′ (lower), plus min(p_32, q_rest) (upper), with p = exp(top_logp) — α estimates for sampling at temperature 1;
- 95 % CI by percentile bootstrap over prompts (per-prompt means, 2000 resamples, seed 0).
Writes a JSON summary `{domain: {"positions", "prompts", "greedy_agree": [mean, lo, hi], "tv_lower": [...], "tv_upper": [...]}}` to `--out` (default stdout).
