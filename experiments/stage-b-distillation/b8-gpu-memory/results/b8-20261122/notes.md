# b8: GPU memory freed by the CPU draft (2026-11-22)

`configs/vram.py` (run from the repo root) starts `llama-server` with the benchmark command (`build_server_cmd`, -c 2048,
γ 4) for target alone, GPU draft and CPU draft (the b4 draft, Q8_0), and reads the server's own VRAM from
`nvidia-smi --query-compute-apps`.

| setup | server VRAM (MiB) |
|---|---|
| target alone | 5,034 |
| target + draft on GPU | 5,120 |
| target + draft on CPU | 5,034 |

- The GPU draft costs 86 MiB: its 146 MB file includes the 82 MB input embedding table, which llama.cpp keeps in host
  memory; the rest is the layers, the V′ head, its small KV cache and buffers.
- 86 MiB is about 600 tokens of Qwen3-8B's KV cache (36 layers x 8 KV heads x 128 x 2 x 2 bytes = 144 KiB per token).
  The CPU draft's memory saving is negligible on an 8 GB card.
- Runs at -c 8192 and 12288 failed for all three setups (KV allocation out of memory) because about 1.7 GB of VRAM
  was held by other processes; that does not change the difference above.
- The CPU draft's real resource saving is GPU time (about 1 ms per drafted token at batch 1), which matters only when
  the GPU is shared with other work.
