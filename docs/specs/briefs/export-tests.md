# Brief: hidden acceptance tests for docs/specs/stage-b-export.md section 4

Write black-box pytest tests before the implementation exists. Read the spec fully, plus `src/train/student.py`, `src/train/data.py`, `src/train/losses.py`, `src/train/vprime.py`, and `src/data/gen_eval_set.py` (copy its llama-server pattern: `/tokenize`, `/completion` with `n_probs`, `temperature -1`, `return_tokens`, health wait, process-group teardown).

## Files (only under `tests_hidden_export/`)

1. `conftest.py`
   - Paths from env vars with defaults: `HDD=/mnt/hdd1/cpu-draft-specdec`; `CKPT=$HDD/runs/b2/kd-pruned-20261121T193214Z/final.pt`; `REF_GGUF=$HDD/gguf/Qwen3-0.6B-Q8_0.gguf`; `TARGET=$HDD/gguf/Qwen3-8B-Q4_K_M.gguf`; `LLAMA_BIN=<repo>/third_party/llama.cpp/build/bin`.
   - `pytest.skip` llama/GPU tests only when a required binary or file is missing.
   - Session fixtures: export f32 and f16 GGUFs into `tmp_path_factory` by running `src/models/export_student_gguf.py` with `sys.executable`; quantize the f16 file to Q8_0 with `llama-quantize`; a server context manager (free port, `start_new_session=True`, kill the process group in `finally`).
2. `test_export_llama.py`, spec items 1-6.
   - Item 1 reference: `Student(StudentConfig(**ckpt["config"]))` loaded from `ckpt["state_dict"]`, float32 on CPU, `log_softmax` over V′ at the last prompt position, mapped to token IDs via `ckpt["vprime"]`. Compare with the server's top-20 at `n_predict 1` (`-ngl 0`, f32 GGUF). Four short plain-text prompts, tokenized by the server's `/tokenize` with `add_special false`; send token lists as the prompt.
   - Item 2: `n_probs 100`, every returned ID is in V′.
   - Item 3: `REF_GGUF` served with the patched build generates 16 tokens without error.
   - Item 4: gguf-py `GGUFReader` (insert `third_party/llama.cpp/gguf-py` on `sys.path`) on the Q8_0 file: `output_vmap.weight` is I32 with `n_vocab` entries, values in `[0, n_vprime]`, exactly `n_vprime` entries below `n_vprime`, and `vmap[vprime[j]] == j` for all `j`.
   - Item 5: speculative server (target `-ngl 99`, `-md` Q8_0 draft, `-ngld 0 -devd none --spec-type draft-simple --draft-max 4 -c 2048`) against a baseline server with the target alone; 3 prompts x 64 tokens, greedy (`temperature -1`), `cache_prompt false`; token lists identical; the speculative response's `timings` report `draft_n > 0` and `draft_n_accepted > 0`.
   - Item 6: exporter exits 2, without leaving the output path, for: missing `--ckpt`; a checkpoint whose `vprime` has a duplicate ID; a checkpoint with an out-of-range V′ ID; a reference GGUF whose vocabulary size differs (build a tiny fake checkpoint with `n_vocab_in` different from the reference's instead of crafting a GGUF). Use small saved dict checkpoints in `tmp_path`.
3. `test_train_fixes.py`, spec item 7 (CPU only, no llama).
   - Tiny `StudentConfig` (for example `n_vocab_in=300, n_vprime=64, hidden=32, n_layers=2, n_heads=2, n_kv_heads=1, head_dim=16, ffn=48, max_seq=32`), seeded.
   - Two microbatches of shape (2, 12) with different numbers of `True` positions in `loss_mask`; `topk_vidx` (k=4, some `-1`), `topk_p` summing below 1.
   - Reference: one model copy, loss computed once on the concatenated batch with the functions in `src/train/losses.py`, `.backward()`. Candidate: an identical model copy, `train.step_backward(model, [mb1, mb2], loss, vmap_t)`. Compare every parameter's `.grad` (`torch.allclose`, `atol=1e-6`, `rtol=1e-4`) and the returned value against the reference loss, for `kd` and `hard` (for `hard`, include next-tokens outside V′ so the counts differ from the mask counts).
   - A step whose masks are all `False` returns 0 and leaves zero gradients.
   - `TeacherBatcher` raises `ValueError` for two teacher dirs with different `seq_len`. Build minimal valid teacher dirs following `docs/specs/stage-b-tooling.md` section 2 (index.json `teacher-topk-v1` plus shard files); reuse the fixture style in `tests/` if one exists.
   - Import as `from train.train import step_backward` with `src` on `sys.path`.

## Rules

- Tests must fail now only because the behavior is missing (exporter absent, `step_backward` absent), not because of broken fixtures. Check that fixtures follow the real checkpoint and teacher-data formats in the source.
- No GPU or llama.cpp is available in your sandbox; keep llama tests correct by construction and make the CPU tests in `test_train_fixes.py` runnable if torch is present.
- Report: files created, what each test checks, and any spec ambiguity.
