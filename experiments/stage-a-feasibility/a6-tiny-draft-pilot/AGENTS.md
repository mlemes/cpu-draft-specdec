# a6: tiny off-the-shelf draft pilot (Llama-2-7B + llama-68m)

Added 2026-11-21 at the operator's request, after a4 showed that the smallest official Qwen3 draft (0.6B) is DRAM-bound on the CPU.

**Goal.** Get an early, cheap data point for the project's largest unknown: the acceptance rate and CPU-draft speedup of a draft of about 50–70M parameters, the size that fits in L3 (a2), before any distillation work. This is a bound, not evidence for H1: different target, tokenizer and training data, and the draft was not distilled from the target.

**Models.**
- Target: Llama-2-7B base, `llama-2-7b.Q4_K_M.gguf` from `TheBloke/Llama-2-7B-GGUF` (Llama 2 Community License), on the GPU.
- Draft: `JackFram/llama-68m` (Apache-2.0; the SpecInfer draft for Llama-2-7B), weights from the repository's safetensors conversion branch `refs/pr/1` (never the pickle `pytorch_model.bin`), converted with the pinned llama.cpp `convert_hf_to_gguf.py` to F16, then quantized to Q8_0. Vocabulary 32,000, shared with Llama-2.
- Both under `/mnt/hdd1/cpu-draft-specdec/` with SHA-256 recorded.

**Method.** `measure.a4_reference_draft` unchanged, with the general and pt_fin prompt sets of a4 (the json set uses Qwen chat tokens and is skipped). Plan as a4: target alone, CPU draft and GPU draft at γ = 4 on all prompts, γ = 2 and 6 on 30 prompts per domain. Also record the draft's touched bytes per token (`measure.touched`) and its CPU ms/token (llama-bench, 3 threads).

**Decision rule** (recorded before the run):

| Acceptance (γ = 4, greedy, general) | Reading |
|---|---|
| ≥ 0.5 and CPU-draft speedup > 1 | Approach is viable; proceed to Stage B with confidence |
| 0.3–0.5 | Thin margin; Stage B must show distillation lifts acceptance |
| < 0.3 | Undistilled tiny drafts are far off; Stage B still decides, plan for a negative result |

**Outputs.** `results/<run_id>/` as in a4, plus `notes.md` applying the decision rule.
