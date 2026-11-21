# Architecture

Status: planned. Nothing here is built yet. Update this file whenever a component's behavior changes.

## Components

| Component | Location | Role |
|---|---|---|
| Target runtime | `third_party/llama.cpp`, CUDA backend | Runs Qwen3-8B Q4_K_M on the RTX 3070 for decode, block verification and teacher scoring |
| Draft runtime | `third_party/llama.cpp`, CPU backend | Runs the student on the Ryzen CPU with AVX2 kernels; MoE layers use `mul_mat_id`, which skips unrouted experts |
| Draft-head patch | `src/patches/` | Restricts the draft's output projection to the vocabulary subset V′ and maps indices back to target token IDs |
| Speculative loop | `src/specdec/` | One process holding both models: drafts γ tokens on the CPU, verifies on the GPU (greedy or stochastic), rolls back both KV caches, logs per-round timings |
| Measurement | `src/measure/` | resctrl (CAT masks, MBM, occupancy), `perf` core and uncore events, IBS sampling, manifest writer, touched-byte calculator from tensor layouts and routing traces |
| Training | `src/train/` | PyTorch distillation: hard-label, forward-KL on top-k teacher logits, TV objective, MoE upcycling, sticky-routing penalty |
| Model building | `src/models/` | Student configurations, structured pruning of Qwen3-0.6B, Hugging Face to GGUF conversion, quantization, random-weight GGUFs for latency sweeps |

## Pinned runtime

- llama.cpp commit `29b52d398b294509746ec0a3f28948a3107c0213`, cloned into `third_party/llama.cpp` on 2026-11-20.
- Build: `cmake -S . -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86 -DGGML_NATIVE=ON -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-13 -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF`, then `cmake --build build -j 10 --target llama-bench llama-cli llama-server llama-speculative llama-speculative-simple llama-lookup llama-perplexity llama-quantize`. CUDA 12.4 does not support gcc 15, so gcc 13 is the CUDA host compiler.
- Changing the commit or flags is a recorded design change.

## Placement

- GPU: target weights (about 5 GB), target KV cache (147 KB per token), verification and, optionally, the stochastic acceptance kernel.
- CPU L3 (goal): the draft's touched weights per token (about 56 MB at Q8_0) and, when it fits, the draft's KV cache (6 KB per token).
- RAM: everything the draft stores but does not touch for the current token: unrouted experts, the full input-embedding table, and teacher-data buffers.

## PCIe traffic per round

| Mode | CPU → GPU | GPU → CPU |
|---|---|---|
| Greedy | γ draft token IDs | Accepted count and the correction or bonus token |
| Stochastic, CPU-side acceptance | γ draft token IDs | Target logits for γ + 1 positions (about 3 MB at γ = 4) |
| Stochastic, GPU-side acceptance | Draft token IDs and q̃ over V′ (about 256 KB at γ = 4) | Accepted count and sampled token |

## Invariants

- Target file, quantization and sampler chain are fixed across compared runs.
- Draft and target share token IDs. V′ only restricts what the draft can propose.
- After every round both KV caches hold exactly the committed prefix.
