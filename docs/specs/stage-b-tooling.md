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

CLI: `--init CKPT.pt|scratch --data DIR:WEIGHT[,DIR:WEIGHT...] --eval-data DIR[,DIR] --out RUN_DIR --loss kd|hard|tv --tokens N --batch-seqs B --seq-len L --lr 6e-4 --warmup 500 --seed 0 --eval-every STEPS`. AdamW (β 0.9/0.95, weight decay 0.1 except norms and embeddings), cosine decay to 10 % of peak, gradient clip 1.0, bf16 autocast, fused/torch.compile optional. Writes `RUN_DIR/metrics.jsonl` (step, tokens, loss, lr, tokens/s, eval metrics), checkpoints `ckpt-<tokens>.pt` and `final.pt`, and `manifest.json` (args, data index SHA-256s, git commit). Never shares the GPU with teacher scoring.

### 3.7 Offline acceptance (`src/train/eval_alpha.py`)

Evaluation set: target-generated greedy continuations (Qwen3-8B Q4_K_M, temperature 0, 128 tokens) of the 300 a4 prompts, scored by `teacher_score` (`$HDD/teacher/eval-<domain>/`, one document per prompt+continuation; positions inside the generated part are marked by a sidecar `positions.json`). Per domain report, over generated positions only: greedy agreement (student argmax over V′ mapped to target ID == teacher argmax) — this is per-position α for greedy speculative decoding — and the TV lower/upper bounds of 3.5 as α estimates for sampling at temperature 1. Bootstrap 95 % CIs over prompts.
