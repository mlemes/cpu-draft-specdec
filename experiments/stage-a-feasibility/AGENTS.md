# Stage A: hardware and runtime feasibility

Short benchmarks, no training. Decides whether the project can work on this machine before any expensive step. Proposal sections 6, 8 and 10.

Order: `a0` first (it validates the counters every later cache claim depends on), then `a1`–`a5` in any order.

| Experiment | Measures |
|---|---|
| `a0-counter-validation` | Whether MBM, occupancy and core fill counters report known traffic correctly |
| `a1-gpu-target-bench` | Target decode, prefill and block-verification cost |
| `a2-cpu-latency-cache-sweep` | CPU draft latency versus touched bytes, precision, threads and CAT allocation |
| `a3-pcie-transfer` | Host–device copy and synchronization latency |
| `a4-reference-draft` | Acceptance of the off-the-shelf Qwen3-0.6B draft, on CPU and GPU |
| `a5-vocab-coverage` | Target probability mass inside candidate draft vocabularies V′ |

## Stage output

`results/stage-a-summary.md`: the measured break-even table (replacing the estimates in proposal section 6) and a go / narrow / stop decision.

## Stop if

- t_d for a model touching ≤ 60 MB is at least 5 ms at γ = 4.
- Latency and MBM bytes per token show no knee as CAT ways shrink. Then drop the cache claim and reframe the project as "small CPU draft".
- p(V′ = 32k) < 0.9.

## Prerequisites (operator, needs sudo)

`perf_event_paranoid` ≤ 0, `modprobe amd_uncore`, resctrl mounted, SMT off, VRAM freed. Download Qwen3-8B, Qwen3-4B and Qwen3-0.6B GGUFs to `/mnt/hdd1/cpu-draft-specdec/gguf/`.
