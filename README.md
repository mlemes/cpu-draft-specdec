# CPU draft, GPU target: cache-resident speculative decoding

> **Status: ongoing research (master's project).** Results are preliminary and will change as experiments continue. Last updated 2026-11-25.

Can a small model distilled from a GPU-resident LLM, running on an otherwise idle CPU, speed up the LLM's generation through speculative decoding? The draft's active weights are sized to stay resident in the 96 MB L3 cache of a Ryzen 5 5500X3D.

## Setup

- **Target:** Qwen3-8B Q4_K_M on an RTX 3070 (8 GB).
- **Draft:** a dense transformer with about 53M parameters in its compute path (12 layers, 32,768-token pruned output head), initialized by pruning Qwen3-0.6B and distilled from the target's top-32 logits with forward KL. Trained in PyTorch on the same GPU.
- **Runtime:** llama.cpp, with the target on CUDA and the draft on CPU, plus a patch for the pruned draft head.
- **Domains:** English Wikipedia, Portuguese financial text (Brazilian central bank committee minutes; evaluation documents are later than the training ones), and JSON structured output. All data is public.

## Preliminary results

- Distillation beats ordinary next-token training for draft quality, and pruned initialization helps further. Greedy agreement with the target is 0.56 on English, 0.65 on financial text, and 0.89 on JSON after adding more structured training data.
- End-to-end speedup over the target alone (greedy decoding, 6 draft threads): **2.07× on JSON** (2.26× with 6 drafted tokens), **1.31× on Portuguese financial text**, and **1.09–1.22× on English**, depending on draft length.
- A draft-free n-gram lookup gives about 1.13–1.17×, so the trained draft pays off mainly on structured output.
- **Open issue:** speculative greedy output matches target-only greedy output on only 30 of 60 prompts. The likely cause is batched-verification numerics flipping near-tied logits, but this is not yet isolated (article section 7.9).

## Next steps

Adaptive draft length, overlapping drafting with verification, exactness and sampling tests, a mixture-of-experts student at the same active-byte budget, and cache-residency evidence from L3 allocation control (CAT) and memory-bandwidth (MBM) counters.

## Read more

- [`docs/article.md`](docs/article.md): working article with methods, results and caveats.
- [`docs/proposal.md`](docs/proposal.md): research proposal, hypotheses and stage plan.
- [`docs/architecture.md`](docs/architecture.md), [`docs/data_flow.md`](docs/data_flow.md): system design.
