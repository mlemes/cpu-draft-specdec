"""b9: bound the gain from overlapping CPU drafting with GPU verification.

Per-step time is recovered from the measured b3 speedup (step = E * t_base / speedup,
E = 1 + gamma * acc). Overlap can hide at most min(gamma * t_draft, t_verify) per step.
"Draft-ahead" hides it only when all gamma tokens and the bonus token match the draft
(probability alpha^(gamma+1), iid approximation); "oracle" hides it on every step.
"""
import csv, sys
res = sys.argv[1] if len(sys.argv) > 1 else "experiments/stage-b-distillation/b9-overlap-bound/results/b9-20261122"
tgt = {(int(r["n_prompt"]), int(r["n_gen"])): float(r["avg_ts"]) for r in csv.DictReader(open(f"{res}/tgt.csv"))}
dft = {(int(r["n_prompt"]), int(r["n_gen"])): float(r["avg_ts"]) for r in csv.DictReader(open(f"{res}/dft.csv"))}
t_base = 1e3 / tgt[(0, 64)]
t_draft = 1e3 / dft[(0, 64)]
t_verify = {k: 1e3 * k / tgt[(k, 0)] for k in (3, 5, 7)}
# (gamma, domain, acceptance, implied alpha, measured speedup) from b3-json23-cpu-20261122T154807Z
rows = [(2, "general", .393, .52, 1.14), (2, "json", .845, .89, 1.68), (2, "pt_fin", .508, .63, 1.28),
        (4, "general", .243, .51, 1.03), (4, "json", .750, .89, 1.89), (4, "pt_fin", .365, .64, 1.24),
        (6, "general", .170, .51, .90), (6, "json", .683, .89, 2.04), (6, "pt_fin", .269, .63, 1.11)]
print(f"t_base {t_base:.2f} ms  t_draft {t_draft:.2f} ms/token  t_verify {t_verify}")
for g, dom, acc, alpha, meas in rows:
    E = 1 + g * acc
    step = E * t_base / meas
    hide = min(g * t_draft, t_verify[g + 1])
    hit = alpha ** (g + 1)
    print(f"{g} {dom:8s} step {step:5.1f} ms  draft {g*t_draft:4.1f} ms ({g*t_draft/step:.0%})  "
          f"measured {meas:.2f}  draft-ahead {E*t_base/(step-hit*hide):.2f} (hit {hit:.2f})  oracle {E*t_base/(step-hide):.2f}")
