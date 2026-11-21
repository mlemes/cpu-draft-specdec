# Stage A tooling: interface specification

Status: specified 2026-11-20, not implemented. Standard library only (see `src/AGENTS.md`). Import root `src/`.

## 1. GGUF I/O: `src/models/gguf_io.py`

GGUF version 3, little-endian, per the format in `ggml/docs/gguf.md` of llama.cpp.

- Header: magic `GGUF` (bytes), uint32 version = 3, uint64 tensor_count, uint64 kv_count.
- Strings: uint64 length + UTF-8 bytes.
- Metadata value types: 0 UINT8, 1 INT8, 2 UINT16, 3 INT16, 4 UINT32, 5 INT32, 6 FLOAT32, 7 BOOL, 8 STRING, 9 ARRAY (uint32 element type + uint64 count + elements), 10 UINT64, 11 INT64, 12 FLOAT64.
- Tensor info: name (string), uint32 n_dims, n_dims × uint64 dims (ggml order: ne0 first, the contiguous row length), uint32 ggml type, uint64 offset relative to the start of the data section.
- Data section starts at the next multiple of `general.alignment` (default 32) after the tensor infos; every tensor offset is a multiple of the alignment.
- ggml types used: F32 = 0, F16 = 1, Q4_0 = 2, Q8_0 = 8.
- Byte sizes: F32 4 per element, F16 2 per element, Q8_0 34 bytes per 32 elements (fp16 scale `d` followed by 32 int8), Q4_0 18 bytes per 32 elements (fp16 scale `d` followed by 16 bytes of packed 4-bit values). For quantized types ne0 must be a multiple of 32.

API:

- `read_gguf(path) -> GGUFFile` with attributes `version: int`, `alignment: int`, `metadata: dict[str, object]` (arrays as Python lists, strings as `str`, bools as `bool`), `tensors: list[TensorInfo]`, `data_offset: int` (absolute file offset of the data section).
- `TensorInfo` has `name: str`, `shape: tuple[int, ...]` (ggml order), `ggml_type: str` (one of `"F32"`, `"F16"`, `"Q4_0"`, `"Q8_0"`; unknown type IDs give `"UNKNOWN_<id>"`), `offset: int` (absolute file offset), `nbytes: int`.
- `tensor_nbytes(shape, ggml_type) -> int`. Raises `ValueError` for a quantized type whose ne0 is not a multiple of 32.
- Bad magic or a version other than 3 raises `ValueError`.

CLI: `PYTHONPATH=src python3 -m models.gguf_io inspect FILE` prints one JSON object to stdout: `version`, `alignment`, `metadata` (each array value replaced by `{"array_len": N}`), `tensors` (list of `{name, shape, type, offset, nbytes}`), `total_tensor_bytes`. Exit code 0; exit code 2 with a message on stderr for an unreadable or invalid file.

## 2. Random-weight model generator: `src/models/random_gguf.py`

CLI: `PYTHONPATH=src python3 -m models.random_gguf --config CONFIG.json --quant {q8_0,q4_0} --seed N --out OUT.gguf`

Config JSON keys: `arch` (`"qwen3"` or `"qwen3moe"`), `hidden`, `layers`, `ffn` (dense FFN width; ignored for `qwen3moe`), `heads`, `kv_heads`, `head_dim`, `vocab`, `context` (default 4096), and for `qwen3moe` also `experts`, `experts_used`, `expert_ffn`.

Output tensors, llama.cpp naming, shapes in ggml order:

| Tensor | Shape | Type |
|---|---|---|
| `token_embd.weight` | (hidden, vocab) | quant |
| `output_norm.weight` | (hidden,) | F32 |
| `output.weight` | (hidden, vocab) | quant |
| `blk.{i}.attn_norm.weight` | (hidden,) | F32 |
| `blk.{i}.attn_q.weight` | (hidden, heads·head_dim) | quant |
| `blk.{i}.attn_k.weight` | (hidden, kv_heads·head_dim) | quant |
| `blk.{i}.attn_v.weight` | (hidden, kv_heads·head_dim) | quant |
| `blk.{i}.attn_output.weight` | (heads·head_dim, hidden) | quant |
| `blk.{i}.attn_q_norm.weight` | (head_dim,) | F32 |
| `blk.{i}.attn_k_norm.weight` | (head_dim,) | F32 |
| `blk.{i}.ffn_norm.weight` | (hidden,) | F32 |
| dense: `blk.{i}.ffn_gate.weight`, `blk.{i}.ffn_up.weight` | (hidden, ffn) | quant |
| dense: `blk.{i}.ffn_down.weight` | (ffn, hidden) | quant |
| moe: `blk.{i}.ffn_gate_inp.weight` | (hidden, experts) | F32 |
| moe: `blk.{i}.ffn_gate_exps.weight`, `blk.{i}.ffn_up_exps.weight` | (hidden, expert_ffn, experts) | quant |
| moe: `blk.{i}.ffn_down_exps.weight` | (expert_ffn, hidden, experts) | quant |

Metadata (prefix `A` = the `arch` value):

- `general.architecture` = arch, `general.name` = `"random-<arch>"`, `general.alignment` = 32 (UINT32).
- UINT32: `A.block_count`, `A.context_length`, `A.embedding_length`, `A.feed_forward_length` (dense only), `A.attention.head_count`, `A.attention.head_count_kv`, `A.attention.key_length` = head_dim, `A.attention.value_length` = head_dim; moe only: `A.expert_count`, `A.expert_used_count`, `A.expert_feed_forward_length`.
- FLOAT32: `A.attention.layer_norm_rms_epsilon` = 1e-6, `A.rope.freq_base` = 1000000.0.
- Tokenizer: `tokenizer.ggml.model` = `"llama"`; `tokenizer.ggml.tokens` (array of `vocab` strings), `tokenizer.ggml.scores` (FLOAT32 array), `tokenizer.ggml.token_type` (INT32 array). Token 0 `<unk>` type 2, token 1 `<s>` type 3, token 2 `</s>` type 3, tokens 3–258 `<0x00>`…`<0xFF>` type 6, the rest `▁t<id>` type 1. UINT32 `tokenizer.ggml.bos_token_id` = 1, `tokenizer.ggml.eos_token_id` = 2, `tokenizer.ggml.unknown_token_id` = 0. Requires vocab ≥ 259.

Weights:

- Quantized blocks: every fp16 scale is finite and non-zero; quant bytes come from `random.Random(seed)`. The same config, quant and seed produce a byte-identical file; a different seed changes tensor data.
- F32 norm weights are all 1.0. `ffn_gate_inp` (router) values are finite random floats from the same generator.
- Invalid configs (unknown arch, missing key, vocab < 259, any quantized ne0 not a multiple of 32) exit with code 2 and a message on stderr, writing no output file.
- Performance: a file of about 64 MB is written in under 30 seconds on this machine.

## 3. Measurement helpers: `src/measure/`

### `resctrl.py`

`class Resctrl(root="/sys/fs/resctrl")`. All paths are relative to `root`, so tests can use a fake directory tree.

- `info() -> dict` with `cbm_mask: int` (from `info/L3/cbm_mask`, hex), `min_cbm_bits: int` (`info/L3/min_cbm_bits`), `num_closids: int` (`info/L3/num_closids`), `mon_features: list[str]` (lines of `info/L3_MON/mon_features`, empty list if the file is absent).
- `ways_mask(n_ways, offset=0) -> int`: contiguous mask of `n_ways` bits starting at bit `offset`. `ValueError` if it exceeds `cbm_mask` or `n_ways < min_cbm_bits`.
- `create_group(name) -> pathlib.Path`: makes `root/name` (no error if it exists). Name must match `[A-Za-z0-9_-]+`, else `ValueError`.
- `set_l3_mask(name, mask, cache_id=0)`: writes `L3:<cache_id>=<mask as lowercase hex without 0x>\n` to `root/name/schemata`. `ValueError` if the mask is zero, not contiguous, has fewer than `min_cbm_bits` bits, or has bits outside `cbm_mask`.
- `assign(name, pid)`: writes `<pid>\n` to `root/name/tasks`.
- `read_mbm_total(name, domain="mon_L3_00") -> int` and `read_occupancy(name, domain="mon_L3_00") -> int`: read `root/name/mon_data/<domain>/mbm_total_bytes` and `.../llc_occupancy`. Name `""` means the root group (files directly under `root/mon_data`).
- `remove_group(name)`: removes `root/name` with `rmdir` semantics.

### `manifest.py`

- `collect_system(sysroot="/") -> dict` with keys `kernel` (`proc/sys/kernel/osrelease`), `cpu_model` (first `model name` in `proc/cpuinfo`), `nproc` (count of `processor` lines), `smt_control` (`sys/devices/system/cpu/smt/control`), `governor` (`sys/devices/system/cpu/cpufreq/policy0/scaling_governor`), `boost` (`sys/devices/system/cpu/cpufreq/boost`), `thp` (`sys/kernel/mm/transparent_hugepage/enabled`), `mem_total_kib` and `swap_used_kib` (`proc/meminfo`: `MemTotal`, and `SwapTotal − SwapFree`, as int). Paths are under `sysroot`. Missing or unreadable values are `None`; strings are stripped.
- `write_manifest(run_dir, *, run_id, extra=None, files=(), sysroot="/") -> dict`: creates `run_dir` if needed and writes `run_dir/manifest.json` with keys `run_id`, `utc_timestamp` (ISO 8601 ending in `Z`), `system` (from `collect_system`), `git_commit` (current commit of the git repository containing the current working directory, or `None`), `files` (list of `{path, bytes, sha256}` for each path in `files`), `extra` (dict, default `{}`). Returns the same dict.

### `stats.py`

- `bootstrap_ci(x, n_boot=10000, alpha=0.05, seed=0) -> (mean, lo, hi)`: percentile bootstrap of the mean.
- `paired_bootstrap_ratio_ci(a, b, n_boot=10000, alpha=0.05, seed=0) -> (ratio, lo, hi)`: ratio = mean(a) / mean(b); each replicate resamples paired indices. `ValueError` for empty input, unequal lengths, or a replicate with a non-positive denominator mean.
- Results are deterministic for a given `seed`; `lo ≤ point ≤ hi` for well-behaved data.
- `expected_tokens(alpha, gamma) -> float`: (1 − α^(γ+1)) / (1 − α), and γ + 1 when α = 1. `ValueError` outside 0 ≤ α ≤ 1 or γ < 1.
- `speedup(alpha, gamma, t_d, t_v, t_t, t_overhead=0.0) -> float`: expected_tokens · t_t / (γ·t_d + t_v + t_overhead).
- `break_even_alpha(gamma, t_d, t_v, t_t, t_overhead=0.0, target=1.0, tol=1e-4) -> float | None`: smallest α in [0, 1] with speedup ≥ target, or `None` if unreachable even at α = 1.

## 4. Streaming benchmark: `src/measure/stream_bench.c`

Build: `make -C src/measure stream_bench` produces `src/measure/stream_bench` (gcc, `-O2`, no AVX-512 flags).

CLI: `stream_bench --bytes N --passes P [--cpu C] [--warmup W] [--wait-signal]`

- Allocates N bytes aligned to 2 MiB, requests transparent huge pages with `madvise(MADV_HUGEPAGE)` (failure is not an error), and fills the buffer with a deterministic pattern: byte i = (i × 131 + 7) mod 256.
- `--cpu C` pins the process to CPU C with `sched_setaffinity`; failure exits with code 1.
- Runs W warm-up passes (default 1), not reported, then P timed passes. Each pass reads every 8-byte word of the buffer and adds it to a 64-bit accumulator.
- With `--wait-signal`, after allocation, fill and warm-up it prints `{"ready": <pid>}` and blocks until SIGUSR1, then runs the timed passes. This lets a driver assign the process to a resctrl group first.
- Output: one JSON object per line on stdout. Each timed pass prints `{"pass": i, "bytes": N, "ns": <int>, "gbps": <float>}` with i from 0, `ns` from `CLOCK_MONOTONIC`. The last line is `{"summary": true, "bytes": N, "passes": P, "median_ns": <int>, "checksum": <uint64 sum of all 8-byte little-endian words of the buffer, wrapping, for one pass>}`.
- N must be a positive multiple of 8 and P ≥ 1; otherwise, or on unknown options, print usage to stderr and exit with code 2.

## 5. Counter-validation driver (experiment a0): `src/measure/a0_counter_validation.py`

Added 2026-11-21. Uses `measure.resctrl`, `measure.manifest` and `stream_bench`. Standard library only. Runs as root on the real machine (`sudo env PYTHONPATH=src python3 -m measure.a0_counter_validation ...`); tests run it against a fake resctrl tree and a fake benchmark.

CLI options (defaults in brackets):

- `--out DIR` (required): run directory.
- `--resctrl-root PATH` [`/sys/fs/resctrl`]
- `--bench CMD` [`src/measure/stream_bench`]: benchmark executable; it must follow the section 4 protocol (`--bytes`, `--passes`, `--cpu`, `--wait-signal`).
- `--sizes-mb LIST` [`4,8,16,32,48,64,96,128,192,384`]: buffer sizes in MiB.
- `--ways LIST` [`1,2,4,8,14`]
- `--reserve-ways N` [2]: the highest N ways are given to the root (default) group for the whole run, so other processes never share the benchmark's ways.
- `--passes N` [5], `--cpu N` [2]
- `--l3-bytes N` [read from `/sys/devices/system/cpu/cpu0/cache/index3/size`, for example `98304K`; 100663296 if unreadable]
- `--perf PATH` [none] and `--perf-events LIST` [none]: when both are given, `PATH stat -x, -e LIST -p PID` is attached after the benchmark reports ready and is stopped with SIGINT after the benchmark exits; counts are parsed from its CSV stderr.

Behavior:

1. Read `Resctrl.info()`. W = number of set bits in `cbm_mask`. Every value in `--ways` must be ≥ 1 and ≤ W − reserve; otherwise exit 2 before changing anything.
2. Save the root group's `L3:` schemata line, then write `L3:0=<mask of the top reserve bits>` to the root group's schemata.
3. Create group `a0bench`. For each ways value w (outer loop) and size s (inner loop):
   a. Set the group's mask to `ways_mask(w, 0)`.
   b. Start the benchmark with `--bytes s --passes P --cpu C --wait-signal`, read its `{"ready": pid}` line, and assign that pid to `a0bench`.
   c. Read the group's `mbm_total_bytes` (before), attach perf if configured, send SIGUSR1, wait for exit (60 s timeout per run, then kill and record an error), read `mbm_total_bytes` (after) and `llc_occupancy`.
   d. Append to `metrics.jsonl`: `ways`, `cat_bytes` (= w × l3_bytes / W), `size_bytes`, `passes`, `median_ns`, `gbps` (size_bytes / median_ns), `mbm_bytes_per_pass` ((after − before) / P), `mbm_ratio` (mbm_bytes_per_pass / size_bytes), `occupancy_bytes`, `perf` (dict event → count, empty without perf), `error` (null or a message).
4. Always (normal exit, exception, SIGINT or SIGTERM): restore the saved root `L3:` line, remove `a0bench`, and kill a still-running benchmark.
5. Write `manifest.json` with `measure.manifest.write_manifest` (`extra` holds the parsed options, `info()`, W and l3_bytes), and `summary.json`. Revised criterion (2026-11-21, after the first run showed no LRU cliff and a residual floor): (a) accuracy: at the smallest allocation, every run with size_bytes ≥ 2 × cat_bytes has 0.9 ≤ mbm_ratio ≤ 1.1; (b) monotone: for each size, mbm_ratio does not rise by more than 0.05 as ways increase (needs ≥ 2 ways); (c) separation: at the largest allocation, the smallest size has mbm_ratio ≤ 0.3 and the largest size ≥ 0.7. `passed` = a and b and c. Runs with an error are excluded. `floor_ratio_median` (median mbm_ratio of runs with size_bytes ≤ 0.5 × cat_bytes) is reported but not judged.
6. When run under sudo (`SUDO_UID` and `SUDO_GID` set), chown the run directory and its files to that user at the end.
7. Exit 0 when the run completes (even if `passed` is false), 2 for invalid options, 1 with the message `needs root: run with sudo` on a PermissionError from resctrl.

## 6. Touched-byte accounting and group launcher

### `src/measure/touched.py`

- `touched_bytes(path) -> dict` for a GGUF written by `models.random_gguf` or a llama.cpp model, using `models.gguf_io.read_gguf`. Returns `stored_bytes` (sum of all tensor nbytes), `embedding_row_bytes` (nbytes of one row of `token_embd.weight`, i.e. `tensor_nbytes((ne0,), type)`), `expert_bytes` (sum over tensors whose name contains `_exps.`), `expert_count` and `expert_used` (from `<arch>.expert_count` / `<arch>.expert_used_count`, 0 when absent), and `touched_bytes_per_token` = stored − token_embd nbytes + embedding_row_bytes − expert_bytes + expert_bytes × expert_used / expert_count (dense models: no expert term).
- `config_for_touched_bytes(target_bytes, quant, vocab=32768, layers=12) -> dict`: a `qwen3` config for `models.random_gguf` with hidden size d a multiple of 128 (minimum 256), ffn = 3d, head_dim 64, heads = d / 64, kv_heads = the largest divisor of heads that is ≤ heads / 4 (at least 1), chosen so the predicted touched bytes per token are closest to `target_bytes`. Prediction: (12 layers × (2.5 d² + 9 d²) + vocab × d) elements × bytes per element (Q8_0 34/32, Q4_0 18/32), plus norms. Ties go to the smaller d.

### `src/measure/exec_in_group.py`

`python3 -m measure.exec_in_group TASKS_FILE -- CMD [ARGS...]`: writes its own pid followed by a newline to TASKS_FILE, then replaces itself with CMD via `os.execvp`. Every thread the command later creates inherits the resctrl group. Exit code 2 for bad usage and 1 if the tasks write fails (with the error on stderr).

## 7. CPU latency and cache sweep (experiment a2): `src/measure/a2_cpu_sweep.py`

Uses resctrl group `a2bench` (same reuse rule as `a0bench`: an existing group with a non-empty tasks file is in use, exit 1). Reuses the safety pattern of section 5 (validate before changing anything; `--out` must not exist; flag-only SIGINT/SIGTERM handlers; root `L3:` line saved, top `--reserve-ways` given to the root group, restored in cleanup; exit codes 0 / 1 / 2 / 3 / 128+signum with the same precedence; MBM warm-up and `Unavailable` retries; chown to SUDO_UID/SUDO_GID).

Options (defaults): `--out DIR` (required); `--resctrl-root` [/sys/fs/resctrl]; `--llama-bench` [third_party/llama.cpp/build/bin/llama-bench]; `--models-dir` [/mnt/hdd1/cpu-draft-specdec/gguf/random]; `--sizes-mb` [16,32,48,64,96,128,256,512]; `--quants` [q8_0,q4_0]; `--threads` [1,3,6]; `--cat-sizes-mb` [48,96,192]; `--cat-ways` [1,2,4,8,14]; `--cat-threads` [6]; `--reserve-ways` [2]; `--reps` [5]; `--n-timing` [128]; `--n-short` [64]; `--n-long` [320]; `--seed` [1]; `--l3-bytes` [from sysfs as in section 5]; `--phases` [grid,cat].

For each configuration:

1. Build the model config with `config_for_touched_bytes`, generate it with `models.random_gguf` into `--models-dir` if a file with the same config, quant and seed does not exist (file name includes a short hash of those), and compute `touched_bytes`.
2. Set the benchmark group's L3 mask: phase `grid` uses all non-reserved ways; phase `cat` uses `ways_mask(w, 0)`.
3. Timing: run `llama-bench -m MODEL -ngl 0 -dev none -t T -p 0 -n N_TIMING -r REPS -o jsonl` (`-dev none` keeps every operation on the CPU; the CUDA build otherwise may offload operations even with `-ngl 0`) through `measure.exec_in_group` with CPU affinity limited to CPUs 0..T−1 (`taskset -c`). Parse `avg_ts` and `stddev_ts` from the tg record.
4. DRAM traffic per token: two more runs with the same flags and `-r 1`, `-n N_SHORT` and `-n N_LONG`, each recording the group's MBM delta over the whole process. `mbm_bytes_per_token` = (Δlong − Δshort) / (N_LONG − N_SHORT), which cancels model loading and warm-up. Read `llc_occupancy` after the long run.
5. Append to `metrics.jsonl`: `phase`, `target_mb`, `quant`, `threads`, `ways`, `cat_bytes`, `model`, `config`, `stored_bytes`, `touched_bytes`, `tg_tps`, `tg_tps_sd`, `ms_per_token` (1000 / tg_tps), `mbm_bytes_per_token`, `mbm_ratio` (mbm_bytes_per_token / touched_bytes), `occupancy_bytes`, `error`.

Phase `grid` covers sizes × quants × threads at the full allocation. Phase `cat` covers cat sizes × quants × cat ways at `--cat-threads`. A failed configuration records `error` and the sweep continues. Write `manifest.json` (options, `info()`, llama-bench path and version line from `llama-bench --version` if available) at the end.

## 8. PCIe microbenchmark (experiment a3): `src/measure/pcie_bench.cu`

Build: `make -C src/measure pcie_bench` with `nvcc -ccbin /usr/bin/g++-13 -O2 -arch=sm_86`.

CLI: `pcie_bench [--sizes B1,B2,...] [--iters N] [--warmup W] [--device D]`, defaults sizes 1024,4096,65536,262144,1048576,3145728, iters 200, warmup 20, device 0.

For each size, direction (`h2d`, `d2h`) and host memory kind (`pinned` via cudaMallocHost, `pageable` via malloc), time each synchronous `cudaMemcpy` with `std::chrono::steady_clock` and print one JSON line: `{"kind": "...", "dir": "...", "bytes": N, "iters": N, "median_us": x, "p95_us": x, "gbps": bytes / median}`. Then time an empty kernel launch followed by `cudaDeviceSynchronize` and print `{"kind": "launch_sync", "iters": N, "median_us": x, "p95_us": x}`. CUDA errors print a message to stderr and exit 1; bad arguments exit 2. No CUDA device: exit 1 with a message.

## 9. GPU target benchmark driver (experiment a1): `src/measure/a1_gpu_bench.py`

`python3 -m measure.a1_gpu_bench --out DIR --model GGUF [--llama-bench PATH] [--depths 0,512,2048] [--verify-n 1,2,3,4,5,6,7,8] [--reps 5]`. `--out` must not exist.

1. Decode: `llama-bench -m MODEL -ngl 99 -p 0 -n 128 -d D -r REPS -o jsonl` for each depth D. Record `avg_ts`, `stddev_ts` and `t_t_ms` = 1000 / avg_ts.
2. Verification cost: `llama-bench -m MODEL -ngl 99 -p N -n 0 -d D -r REPS -o jsonl` for each N in `--verify-n` and each depth D. Record `t_v_ms` = 1000 × N / avg_ts (time for one batch of N tokens).
3. Before and after, record `nvidia-smi --query-gpu=memory.used,memory.total,pcie.link.gen.current,pcie.link.width.current --format=csv,noheader,nounits`.
4. Write `metrics.jsonl` (one line per llama-bench record with `kind` = `decode` or `verify`, `depth`, `n`, timing fields, `error`) and `manifest.json` (model sha256 via `measure.manifest`). A failed llama-bench call records `error` and the driver continues; exit code 0 unless options are invalid (2).

## 10. Reference-draft acceptance driver (experiment a4): `src/measure/a4_reference_draft.py`

`python3 -m measure.a4_reference_draft --out DIR --target GGUF --draft GGUF --prompts F1,F2,... [--llama-server PATH] [--plan PLAN] [--n-predict 64] [--port 18080] [--draft-threads 6] [--seed 1234]`. `--out` must not exist (exit 2). Default `--llama-server` is `third_party/llama.cpp/build/bin/llama-server` relative to the repository root. Prompt files are JSONL with fields `id`, `domain`, `prompt`.

`--plan` is a comma-separated list of `device:gamma:subset` items; default `none:0:all,cpu:4:all,gpu:4:all,cpu:2:30,cpu:6:30,gpu:2:30,gpu:6:30`. `device` is `none` (target alone), `cpu` or `gpu`; `subset` is `all` or the first N prompts of each file.

For each plan item (revised 2026-11-21 after the first run: this llama.cpp build defaults to `--spec-type none`, which loads the draft but never uses it; `-c 4096` left no VRAM for a GPU draft):

1. Start `llama-server -m TARGET -ngl 99 -c 2048 -np 1 --port PORT --no-webui` plus, unless `device` is `none`: `-md DRAFT --spec-type draft-simple --spec-draft-n-max G --spec-draft-n-min G --spec-draft-p-min 0`, and for `cpu`: `-ngld 0 -devd none -td DRAFT_THREADS -tbd DRAFT_THREADS`, for `gpu`: `-ngld 99`. Server stdout/stderr go to `DIR/server-<item>.log`. Poll `GET /health` every 0.2 s until HTTP 200 (timeout 180 s; on timeout record an error row for the item, stop the server and continue).
2. For each prompt and each mode (`greedy`: temperature 0; `sampled`: temperature 0.7, top_k 20, top_p 0.8, min_p 0), `POST /completion` with JSON `{"prompt", "n_predict", "temperature", "top_k", "top_p", "min_p", "seed": SEED + index of the prompt within its file, "cache_prompt": false}`. Timeout 300 s per request.
3. Append one row per request to `metrics.jsonl`: `device`, `gamma`, `mode`, `domain`, `prompt_id`, `predicted_n`, `predicted_ms`, `tokens_per_s` (predicted_n / predicted_ms × 1000), `prompt_n`, `prompt_ms`, `draft_n`, `draft_n_accepted` (from the response's `timings`; null when absent), `acceptance` (draft_n_accepted / draft_n, null when draft_n is 0 or absent), `error`.
4. Stop the server (SIGTERM, then SIGKILL after 10 s) before the next item, and in cleanup on any exit path (flag-only signal handlers as in section 5; exit 128 + signum after cleanup).

Write `summary.json` grouped by (device, gamma, mode, domain): request count, mean acceptance, mean tokens_per_s, and speedup = mean tokens_per_s / mean tokens_per_s of the `none` item for the same mode and domain (null if missing). Write `manifest.json` with sha256 of target and draft (via `measure.manifest`). Standard library only (`urllib.request`, `http.client`).

## 11. Vocabulary coverage tool (experiment a5): `src/measure/vocab_coverage.cpp`

Build: `make -C src/measure vocab_coverage`, C++17 with `g++`, including `third_party/llama.cpp/include` and `third_party/llama.cpp/ggml/include`, linking `-lllama -lggml -lggml-base` from `third_party/llama.cpp/build/bin` with an rpath to that directory. The Makefile target may assume the llama.cpp build exists.

CLI: `vocab_coverage --model GGUF --freq-text F1[,F2...] --eval PROMPTS1[,PROMPTS2...] --out OUT.json [--sizes 16384,32768,65536] [--ctx 1024] [--ngl 99] [--max-rows N] [--ranking-out RANK.json]`.

1. Frequency ranking: tokenize each frequency text file with `llama_tokenize` (no BOS, special tokens parsed). Count occurrences per token id. Rank by count descending, ties by id ascending; tokens never seen follow in id order. V′_k is the first k ranked ids. `--ranking-out` writes the first max(sizes) ids as a JSON array.
2. Evaluation: each eval file is JSONL with `prompt` and `heldout` strings (minimal JSON string parsing is enough: these two fields, with standard escapes including `\uXXXX` and surrogate pairs). Text = prompt + heldout (json domain prompts already end with the assistant header; join without a separator when the prompt ends with a newline, else with one space). Tokenize with BOS as the model's vocab requests and special tokens parsed. Truncate to `--ctx` tokens.
3. Run the model over the tokens in one batch with logits for every position. For each position t < n − 1, with p = softmax(logits_t): `mass_k` = Σ p over V′_k; `greedy_in_k` = 1 if argmax p ∈ V′_k; `actual_in_k` = 1 if token t + 1 ∈ V′_k. Average over the row's positions.
4. Output `OUT.json`: `{"model", "sizes", "ctx", "freq_files", "rows": [{"file", "row", "n_positions", "mass": {k: x}, "greedy_in": {k: x}, "actual_in": {k: x}}]}` with k as decimal strings. Errors (missing file, model load failure, bad arguments) print to stderr and exit 1 (2 for bad arguments).

A small Python wrapper is not required; Python post-processing computes per-domain means and bootstrap CIs with `measure.stats`.
