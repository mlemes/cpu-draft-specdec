# Experiments

One folder per stage, run in order. Each experiment subfolder has its own `AGENTS.md` (goal, inputs, method, outputs, gate), a `configs/` folder and a `results/` folder. Shared code lives in `../src/`; the design is in `../docs/proposal.md`.

| Stage | Question | Gate to continue |
|---|---|---|
| `stage-a-feasibility` | Can acceleration happen at all on this machine, and can L3 residency be measured? | Counters validated; some (γ, t_d, α) region reaches 1.3× on paper; p(V′ = 32k) ≥ 0.9 |
| `stage-b-distillation` | How much acceptance do small dense and MoE students reach, and do they stay cache-resident? | D-S α ≥ break-even α for its measured t_d minus 0.05 |
| `stage-c-end-to-end` | Does the measured end-to-end speedup beat every baseline? | Speedup CI lower bound > 1.0 for at least one draft |
| `stage-d-adaptation` | When does refreshing the student pay back its cost? | Final stage; report N* and cumulative net savings |

A failed gate is a result. Write it up in the stage's results and stop or narrow as `../docs/proposal.md` section 8 says; do not start the next stage.

Results go in `<experiment>/results/<run_id>/` with `manifest.json` and `metrics.jsonl` (format in the root `AGENTS.md`).
