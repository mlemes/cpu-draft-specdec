"""Unit tests for Stage A2 tooling: touched, exec_in_group, and a2_cpu_sweep."""

from __future__ import annotations

import contextlib
import io
import json
import os
import pathlib
import signal
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from measure import a2_cpu_sweep as a2
from measure import exec_in_group
from measure.touched import config_for_touched_bytes, touched_bytes
from models.gguf_io import read_gguf, tensor_nbytes
from models.random_gguf import generate_random_gguf


FAKE_LLAMA_BENCH = """#!/usr/bin/env python3
import argparse
import json
import os
import pathlib
import sys

if "--version" in sys.argv:
    print("version: 1234 (fake-llama-bench)")
    sys.exit(0)

parser = argparse.ArgumentParser()
parser.add_argument("-m", required=True)
parser.add_argument("-ngl", type=int, required=True)
parser.add_argument("-dev", required=True)
parser.add_argument("-t", type=int, required=True)
parser.add_argument("-p", type=int, required=True)
parser.add_argument("-n", type=int, required=True)
parser.add_argument("-r", type=int, required=True)
parser.add_argument("-o", required=True)
args = parser.parse_args()

if args.dev != "none":
    sys.stderr.write(f"expected -dev none, got {args.dev}\\n")
    sys.exit(2)

if os.environ.get("FAKE_BENCH_FAIL") == "1":
    sys.stderr.write("simulated llama-bench failure\\n")
    sys.exit(1)

mon_dir_env = os.environ.get("FAKE_MON_DIR")
if mon_dir_env:
    mon_dir = pathlib.Path(mon_dir_env)
    mbm_path = mon_dir / "mbm_total_bytes"
    cur = int(mbm_path.read_text(encoding="utf-8").strip())
    # Fixed model-load overhead + per-token traffic proportional to n_gen * reps.
    delta = 100_000 + args.n * args.r * 50_000
    mbm_path.write_text(f"{cur + delta}\\n", encoding="utf-8")
    (mon_dir / "llc_occupancy").write_text("4194304\\n", encoding="utf-8")

# Emit a prompt line (n_gen == 0) and a tg line (n_gen > 0) to verify filtering.
print(json.dumps({"n_prompt": 16, "n_gen": 0, "avg_ts": 999.0, "stddev_ts": 1.0}))
print(json.dumps({"n_prompt": 0, "n_gen": args.n, "avg_ts": 42.5, "stddev_ts": 0.5}))
"""


class TestTouched(unittest.TestCase):
    def test_touched_bytes_dense(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "dense.gguf"
            cfg = {
                "arch": "qwen3",
                "hidden": 256,
                "layers": 2,
                "ffn": 768,
                "heads": 4,
                "kv_heads": 1,
                "head_dim": 64,
                "vocab": 320,
                "context": 512,
            }
            generate_random_gguf(cfg, "q8_0", 1, path)
            res = touched_bytes(path)
            gguf = read_gguf(path)
            embd = next(t for t in gguf.tensors if t.name == "token_embd.weight")
            expected_row = tensor_nbytes((256,), embd.ggml_type)
            self.assertEqual(res["expert_bytes"], 0)
            self.assertEqual(res["expert_count"], 0)
            self.assertEqual(res["expert_used"], 0)
            self.assertEqual(res["embedding_row_bytes"], expected_row)
            self.assertEqual(
                res["touched_bytes_per_token"],
                res["stored_bytes"] - embd.nbytes + expected_row,
            )

    def test_touched_bytes_moe(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "moe.gguf"
            cfg = {
                "arch": "qwen3moe",
                "hidden": 256,
                "layers": 2,
                "heads": 4,
                "kv_heads": 1,
                "head_dim": 64,
                "vocab": 320,
                "context": 512,
                "experts": 8,
                "experts_used": 2,
                "expert_ffn": 512,
            }
            generate_random_gguf(cfg, "q4_0", 2, path)
            res = touched_bytes(path)
            gguf = read_gguf(path)
            embd = next(t for t in gguf.tensors if t.name == "token_embd.weight")
            exp_bytes = sum(t.nbytes for t in gguf.tensors if "_exps." in t.name)
            self.assertEqual(res["expert_count"], 8)
            self.assertEqual(res["expert_used"], 2)
            self.assertEqual(res["expert_bytes"], exp_bytes)
            expected_touched = (
                res["stored_bytes"]
                - embd.nbytes
                + res["embedding_row_bytes"]
                - exp_bytes
                + (exp_bytes * 2) / 8
            )
            self.assertAlmostEqual(res["touched_bytes_per_token"], expected_touched)

    def test_config_for_touched_bytes_and_tie_break(self) -> None:
        cfg_small = config_for_touched_bytes(1024, "q8_0")
        self.assertEqual(cfg_small["hidden"], 256)
        self.assertEqual(cfg_small["ffn"], 768)
        self.assertEqual(cfg_small["heads"], 4)
        self.assertEqual(cfg_small["kv_heads"], 1)
        self.assertEqual(cfg_small["layers"], 12)

        # Midpoint tie between d=256 and d=384 should pick smaller d=256.
        bpe = 34.0 / 32.0
        p256 = (6 * (11.5 * 256 * 256) + 320 * 256) * bpe + (6 * 3 * 256 + 256) * 4
        p384 = (6 * (11.5 * 384 * 384) + 320 * 384) * bpe + (6 * 3 * 384 + 384) * 4
        mid = (p256 + p384) / 2.0
        cfg_tie = config_for_touched_bytes(mid, "q8_0", vocab=320, layers=6)
        self.assertEqual(cfg_tie["hidden"], 256)
        self.assertEqual(cfg_tie["layers"], 6)


class TestExecInGroup(unittest.TestCase):
    def test_bad_usage_returns_2(self) -> None:
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            rc = exec_in_group.main(["/tmp/tasks"])
        self.assertEqual(rc, 2)

    def test_unwritable_tasks_returns_1(self) -> None:
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            rc = exec_in_group.main(["/nonexistent_dir_xyz/tasks", "--", "true"])
        self.assertEqual(rc, 1)

    def test_writes_pid_and_execs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tasks_file = pathlib.Path(tmp) / "tasks"
            out_file = pathlib.Path(tmp) / "pid.txt"
            src_dir = str(pathlib.Path(__file__).resolve().parent.parent / "src")
            env = os.environ.copy()
            env["PYTHONPATH"] = src_dir
            proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "measure.exec_in_group",
                    str(tasks_file),
                    "--",
                    sys.executable,
                    "-c",
                    f"import os, pathlib; pathlib.Path({str(out_file)!r}).write_text(str(os.getpid()))",
                ],
                env=env,
                capture_output=True,
                text=True,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertEqual(
                tasks_file.read_text(encoding="utf-8").strip(),
                out_file.read_text(encoding="utf-8").strip(),
            )


class TestA2CpuSweep(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self.tmp.name)
        self.resctrl = self.root / "resctrl"
        (self.resctrl / "info" / "L3").mkdir(parents=True)
        (self.resctrl / "info" / "L3" / "cbm_mask").write_text("ffff\n", encoding="utf-8")
        (self.resctrl / "info" / "L3" / "min_cbm_bits").write_text("1\n", encoding="utf-8")
        (self.resctrl / "info" / "L3" / "num_closids").write_text("16\n", encoding="utf-8")
        (self.resctrl / "info" / "L3_MON").mkdir(parents=True)
        (self.resctrl / "info" / "L3_MON" / "mon_features").write_text(
            "llc_occupancy\nmbm_total_bytes\n", encoding="utf-8"
        )
        self.original_schemata = "    MB:0=2048\n    L3:0=ffff\n"
        (self.resctrl / "schemata").write_text(self.original_schemata, encoding="utf-8")

        self.mon_dir = self.resctrl / "a2bench" / "mon_data" / "mon_L3_00"
        self.mon_dir.mkdir(parents=True)
        (self.resctrl / "a2bench" / "tasks").write_text("", encoding="utf-8")
        (self.resctrl / "a2bench" / "schemata").write_text("L3:0=ffff\n", encoding="utf-8")
        (self.mon_dir / "mbm_total_bytes").write_text("0\n", encoding="utf-8")
        (self.mon_dir / "llc_occupancy").write_text("0\n", encoding="utf-8")

        self.bench = self.root / "fake_llama_bench.py"
        self.bench.write_text(FAKE_LLAMA_BENCH, encoding="utf-8")
        self.bench.chmod(0o755)

        self.models_dir = self.root / "models"
        self.out_dir = self.root / "run_out"

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _base_argv(self, extra: list[str] | None = None) -> list[str]:
        argv = [
            "--out",
            str(self.out_dir),
            "--resctrl-root",
            str(self.resctrl),
            "--llama-bench",
            str(self.bench),
            "--models-dir",
            str(self.models_dir),
            "--sizes-mb",
            "1",
            "--quants",
            "q8_0",
            "--threads",
            "1",
            "--cat-sizes-mb",
            "1",
            "--cat-ways",
            "1,2",
            "--cat-threads",
            "1",
            "--reserve-ways",
            "2",
            "--reps",
            "2",
            "--n-timing",
            "16",
            "--n-short",
            "8",
            "--n-long",
            "24",
            "--l3-bytes",
            "100663296",
        ]
        if extra:
            argv.extend(extra)
        return argv

    def test_end_to_end_grid_and_cat(self) -> None:
        with mock.patch.dict(os.environ, {"FAKE_MON_DIR": str(self.mon_dir)}):
            rc = a2.main(self._base_argv())
        self.assertEqual(rc, 0)

        # Root L3 schemata restored.
        self.assertEqual(
            (self.resctrl / "schemata").read_text(encoding="utf-8").strip(),
            "L3:0=ffff",
        )

        rows = [
            json.loads(line)
            for line in (self.out_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.assertEqual(len(rows), 3)  # 1 grid + 2 cat
        self.assertEqual([r["phase"] for r in rows], ["grid", "cat", "cat"])
        for row in rows:
            self.assertIsNone(row["error"])
            self.assertAlmostEqual(row["tg_tps"], 42.5)
            self.assertAlmostEqual(row["tg_tps_sd"], 0.5)
            self.assertAlmostEqual(row["ms_per_token"], 1000.0 / 42.5)
            # delta_long - delta_short = (24 - 8) * 50_000; divided by (24 - 8) == 50_000
            self.assertAlmostEqual(row["mbm_bytes_per_token"], 50_000.0)
            self.assertEqual(row["occupancy_bytes"], 4194304)

        # Only one cached model file created and reused across all 3 runs.
        cached = list(self.models_dir.glob("*.gguf"))
        self.assertEqual(len(cached), 1)

        manifest = json.loads((self.out_dir / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["extra"]["llama_bench_version"], "version: 1234 (fake-llama-bench)")
        self.assertEqual(manifest["extra"]["W"], 16)

    def test_existing_out_returns_2(self) -> None:
        self.out_dir.mkdir()
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            rc = a2.main(self._base_argv())
        self.assertEqual(rc, 2)
        self.assertEqual((self.resctrl / "schemata").read_text(encoding="utf-8"), self.original_schemata)

    def test_group_in_use_returns_1(self) -> None:
        (self.resctrl / "a2bench" / "tasks").write_text(f"{os.getpid()}\n", encoding="utf-8")  # a live pid marks the group in use
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            rc = a2.main(self._base_argv())
        self.assertEqual(rc, 1)
        self.assertEqual((self.resctrl / "schemata").read_text(encoding="utf-8"), self.original_schemata)

    def test_failed_bench_records_error_and_continues(self) -> None:
        with mock.patch.dict(
            os.environ,
            {"FAKE_MON_DIR": str(self.mon_dir), "FAKE_BENCH_FAIL": "1"},
        ):
            rc = a2.main(self._base_argv(["--phases", "grid"]))
        self.assertEqual(rc, 0)
        rows = [
            json.loads(line)
            for line in (self.out_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.assertEqual(len(rows), 1)
        self.assertIsNotNone(rows[0]["error"])
        self.assertIn("exited with code 1", rows[0]["error"])
        self.assertIsNone(rows[0]["tg_tps"])


if __name__ == "__main__":
    unittest.main()
