# b7: training for acceptance (total-variation loss)

Question: the per-position acceptance rate is alpha = 1 - TV(p, q), yet the student is trained with forward KL. Does
fine-tuning on 1 - overlap (`--loss tv`, the lower bound from the teacher's top-32) raise acceptance beyond what the
same extra training with forward KL gives? (DistillSpec, Zhou et al. 2024, reports objective choice matters.)

Design: both arms start from the b5 final checkpoint (12 layers, b5 mix) and train 24M more tokens on the b5 mix at peak
LR 1e-4 (warmup 100 steps, cosine to 10%), seed 1, batch 4 x 4. Arms: `kd` (control: extra training alone) and `tv`.
Compare offline greedy agreement (greedy acceptance) and `tv_lower` (sampled acceptance at temperature 1) per domain.

Run: `configs/run_b7.sh`.
