# d2: update cost

**Goal.** Measure each component of C_u for one student update.

**Components.** Teacher scoring (zero extra generation when target distributions are captured during verification), training, GGUF export, quantization and repacking, reload, cache warm-up, and serving throughput lost while the GPU trains.

**Method.** Time each component at 2–3 update sizes (for example 1M, 5M and 20M tokens). Measure serving tokens/s while training runs on the GPU to get the interference cost.

**Outputs.** C_u per update size, with GPU-seconds converted to lost target tokens.
