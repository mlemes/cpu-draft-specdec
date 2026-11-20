# d3: update policies

**Goal.** Compare update policies on the d1 streams.

**Policies.**
1. Frozen student.
2. Periodic update every N tokens.
3. Change-triggered: CUSUM on per-round acceptance.
4. Amortization rule: update when expected remaining segment tokens × predicted Δ exceeds C_u.
5. For recurring workloads: reload a stored earlier student version instead of retraining.

**Outputs.** Cumulative net seconds saved over time per policy, the measured N* per update size, and the policy ranking with CIs across streams.
