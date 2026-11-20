# d1: workload streams

**Goal.** Build held-out streams that exercise adaptation.

**Domains.** General text, Portuguese financial text (public central bank and securities-regulator documents), structured JSON outputs. Public data only.

**Stream shapes.** Stable periods, abrupt switches between domains, and recurrences of an earlier domain. Segment lengths 20k–200k tokens. Fix and hash each stream; none of its text may appear in training data.

**Outputs.** Stream definitions in `configs/` and token files under `$HDD/streams/`.
