# Brief: student exporter and trainer fixes (developer)

Implement `docs/specs/stage-b-export.md` sections 2 and 3. Section 1 (the llama.cpp patch) is done elsewhere; do not touch `third_party/`.

## Files you may change

- New `src/models/export_student_gguf.py` (section 2)
- `src/train/train.py`: add `step_backward` (section 3.1) and use it in the training loop
- `src/train/data.py`: `seq_len` check (section 3.2)
- New focused tests under `tests/` for your helpers (for example `tests/test_export_student.py`, `tests/test_step_backward.py`)
- `docs/architecture.md`, `docs/data_flow.md`: one line each for the exporter

## Notes

- The checkpoint format is produced by `_save_checkpoint` in `src/train/train.py`; read it.
- gguf-py lives in `third_party/llama.cpp/gguf-py`, which is not in your sandbox. Write against its public API (`gguf.GGUFWriter(path, arch)`, `add_*` methods, `add_tensor(name, ndarray)`, `write_header_to_file`, `write_kv_data_to_file`, `write_tensors_to_file`, `close`; `gguf.GGUFReader(path)` with `.fields` and `.tensors`). For copying `tokenizer.*` keys from the reference, read each field's value and type from `GGUFReader.fields` and re-add it with the matching writer call (strings, string arrays, integer arrays, float arrays, scalars). Keep this logic in small functions that can be unit-tested with a fake reader object. The `gguf` import must happen lazily inside the functions that need it, so `--help` and argument errors work without gguf-py.
- `step_backward`: count positions per microbatch exactly as the loss functions in `src/train/losses.py` do, compute `N` before any backward, scale each microbatch loss by `n_i / N`. Keep autocast behavior identical to the current loop.
- torch may be absent in your sandbox; if so, make the code correct by careful reading and say what you could not run.

## Acceptance

The Architect runs hidden acceptance tests (llama.cpp and GPU parts locally). Through `verify`, `python3 -m pytest -q tests_hidden_export/test_train_fixes.py` and the existing suite `python3 -m pytest -q tests` must pass where torch is available.

## Report

Files changed, how tokenizer keys are copied, what you ran, what you could not run.
