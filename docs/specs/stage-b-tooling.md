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
- For each position compute log-softmax over the **full** vocabulary (numerically stable: subtract the max, accumulate in double), then the top-`k` tokens by logit, sorted by descending log-probability; ties broken by smaller token ID.
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

## 3. Student and trainer (b2) — to be specified after b1 lands

Pilot shape of D-S (deviation from the proposal recorded here): hidden 512, 12 layers, SwiGLU FFN 1536, **4 query heads and 1 KV head of dimension 128** (same KV bytes per token as 2 × 64; keeps Qwen3-0.6B's head dimension so pruned initialization copies whole heads), full 151,936-token input embedding, untied output head over V′ = 32,768 frequency-ranked tokens (a5 ranking). Offline acceptance metrics on target-generated continuations: greedy agreement with the teacher argmax, and Σ min(p, q) over V′ with the tail bound.
