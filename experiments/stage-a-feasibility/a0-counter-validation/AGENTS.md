# a0: counter validation

**Goal.** Show that the counters used for cache claims report known traffic correctly on this Zen 3 machine. Every later cache claim depends on this.

**Method.**
1. A pinned single-thread microbenchmark streams repeatedly over a buffer of size S, from 4 MB to 384 MB, read-only.
2. Run it in a resctrl group with CAT masks of 1, 2, 4, 8 and 16 ways (6 MiB per way). Keep all other processes in a disjoint mask.
3. Record per pass: `mbm_total_bytes`, `llc_occupancy`, `ls_dmnd_fills_from_sys.{int_cache,mem_io_local}`, hardware-prefetch fill events, `amd_l3` misses, `amd_df` DRAM events, and time.
4. Repeat with hardware prefetchers at their default setting only; record which prefetch events are available.

**Outputs.** `results/<run_id>/`: bytes per pass by counter versus S and CAT allocation, and a short note on each counter's agreement and limits.

**Pass** (revised 2026-11-21 after the first run). (a) MBM accuracy: at the smallest allocation, MBM bytes per pass are within 10% of the buffer size for buffers at least twice the allocation. (b) For each buffer size, MBM traffic does not rise as ways are added. (c) At the largest allocation, a small buffer shows ≤ 0.3 and the largest ≥ 0.7 of its bytes as DRAM traffic. The original criterion assumed an LRU cliff and a zero floor; this L3 shows neither (see `results/a0-20261121T004032Z/notes.md`). Core fill-source events are not usable as DRAM evidence on this part.

**If it fails.** No cache claim is allowed. Consult the operator before continuing Stage A.
