# a6 rerun with `-tbd 4` (CPU draft γ = 4 and target alone only)

Same setup as the first a6 run. The only change: draft batch threads equal draft threads (`-td 4 -tbd 4`). See a7 for why.

| γ=4 | CPU-draft speedup, first run (`-tbd` default) | rerun (`-tbd 4`) |
|---|---|---|
| greedy general | 1.30 | 1.40 (1.34–1.45) |
| greedy pt_fin | 1.03 | 1.12 (1.08–1.17) |
| sampled general | 1.14 | 1.22 (1.19–1.26) |
| sampled pt_fin | 0.95 | 1.01 (0.97–1.04) |

Acceptance unchanged (0.31 / 0.19 greedy). The GPU draft (1.65× / 1.32×, first run) is still faster. [V]
