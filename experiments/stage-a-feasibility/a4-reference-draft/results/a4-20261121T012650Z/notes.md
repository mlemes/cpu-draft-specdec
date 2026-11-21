# a4 run 2026-11-21T01:26:50Z: INVALID

Do not use these speculative-decoding numbers.

- CPU-draft items: the server loaded the draft model but never drafted. This llama.cpp build defaults to `--spec-type none`; without `--spec-type draft-simple` the `-md` model is unused. Responses carry no draft statistics and throughput equals the target alone.
- GPU-draft items: the server failed to start (`cudaMalloc failed: out of memory` allocating the draft's 448 MiB KV cache at `-c 4096`).
- Helper tests ran for about 15 s during the baseline phase, which may have slowed a few baseline requests.

The `none` baseline rows remain a valid measurement of the target alone (about 77–78 tokens/s, matching a1). The driver now passes `--spec-type draft-simple` and uses `-c 2048` (longest prompt 573 tokens); see the rerun.
