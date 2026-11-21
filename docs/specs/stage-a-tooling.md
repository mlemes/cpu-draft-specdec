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
5. Write `manifest.json` with `measure.manifest.write_manifest` (`extra` holds the parsed options, `info()`, W and l3_bytes), and `summary.json`: for each w, `above` = runs with size_bytes ≥ 2 × cat_bytes and `below` = runs with size_bytes ≤ 0.5 × cat_bytes; `pass_w` is true when every `above` run has 0.9 ≤ mbm_ratio ≤ 1.1 and every `below` run has mbm_ratio ≤ 0.1 and both lists are non-empty. Top-level `passed` is true when every w passes. `summary.json` also lists per-w the run counts used.
6. When run under sudo (`SUDO_UID` and `SUDO_GID` set), chown the run directory and its files to that user at the end.
7. Exit 0 when the run completes (even if `passed` is false), 2 for invalid options, 1 with the message `needs root: run with sudo` on a PermissionError from resctrl.
