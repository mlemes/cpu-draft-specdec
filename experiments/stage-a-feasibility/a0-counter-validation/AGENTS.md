# a0: counter validation

**Goal.** Show that the counters used for cache claims report known traffic correctly on this Zen 3 machine. Every later cache claim depends on this.

**Method.**
1. A pinned single-thread microbenchmark streams repeatedly over a buffer of size S, from 4 MB to 384 MB, read-only.
2. Run it in a resctrl group with CAT masks of 1, 2, 4, 8 and 16 ways (6 MiB per way). Keep all other processes in a disjoint mask.
3. Record per pass: `mbm_total_bytes`, `llc_occupancy`, `ls_dmnd_fills_from_sys.{int_cache,mem_io_local}`, hardware-prefetch fill events, `amd_l3` misses, `amd_df` DRAM events, and time.
4. Repeat with hardware prefetchers at their default setting only; record which prefetch events are available.

**Outputs.** `results/<run_id>/`: bytes per pass by counter versus S and CAT allocation, and a short note on each counter's agreement and limits.

**Pass.** MBM bytes per pass ≈ S (within 10%) when S clearly exceeds the allocation, and close to 0 when S fits well inside it, with a visible transition near the allocation. Record which other counters agree. `amd_df` byte events may read 0 on this model; report that rather than treating it as a pass.

**If it fails.** No cache claim is allowed. Consult the operator before continuing Stage A.
