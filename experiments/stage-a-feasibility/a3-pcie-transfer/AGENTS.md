# a3: PCIe transfer and synchronization

**Goal.** Measure the communication cost of one speculative round on this PCIe 3.0 x16 link.

**Method.**
- Pinned and pageable H2D and D2H copies of 1 KB, 4 KB, 64 KB, 256 KB, 1 MB and 3 MB (the sizes of token IDs, q̃ over V′, and full target logits for γ + 1 positions).
- Stream synchronization and kernel-launch round trip with an empty kernel.
- Report the link generation and width during the run (it idles at Gen1).

**Outputs.** Latency per size with p50, p95 and CI; t_comm + t_sync for each verification placement in `docs/architecture.md`.
