# b2: dense student D-S

**Goal.** Train the dense draft (hidden 512, 12 layers, FFN 1536, 8 query / 2 KV heads, head dim 64, head over V′) and measure how much distillation helps acceptance.

**Arms** (300M tokens each):
1. Hard labels only (no teacher soft targets).
2. Forward KL on top-32 teacher records, with a tail bin.
3. TV objective on student-generated data (draft-distribution-aware).
4. Initialization control: arm 2 from scratch instead of pruned from Qwen3-0.6B.

**Outputs.** Checkpoints, GGUF exports at Q4_0 and Q8_0 with touched and stored bytes, greedy and T = 0.7 acceptance per domain, and the TV-objective upper and lower bounds on acceptance.

**Gate.** Compare α with the break-even α from Stage A for the measured t_d.
