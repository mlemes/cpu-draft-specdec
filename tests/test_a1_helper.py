"""Unit tests for src/measure/a1_gpu_bench.py."""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from measure import a1_gpu_bench as a1


FAKE_LLAMA_BENCH = """#!/usr/bin/env python3
import argparse
import json
import os
import sys

parser = argparse.ArgumentParser()
parser.add_argument("-m", required=True)
parser.add_argument("-ngl", type=int, required=True)
parser.add_argument("-p", type=int, required=True)
parser.add_argument("-n", type=int, required=True)
parser.add_argument("-d", type=int, required=True)
parser.add_argument("-r", type=int, required=True)
parser.add_argument("-o", required=True)
args = parser.parse_args()

log_path = os.environ.get("FAKE_BENCH_LOG")
if log_path:
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(sys.argv[1:]) + "\\n")

mode = os.environ.get("FAKE_BENCH_MODE", "ok")
if mode == "fail_decode" and args.n > 0:
    sys.stderr.write("simulated cuda oom\\n")
    sys.exit(1)
if mode == "bad_json" and args.p == 2:
    print("not valid json")
    print(json.dumps({"n_gen": 0, "n_prompt": 0, "avg_ts": 10.0}))
    sys.exit(0)
if mode == "zero_ts" and args.n > 0:
    print(json.dumps({"n_gen": args.n, "n_prompt": args.p, "avg_ts": 0.0, "stddev_ts": 0.0}))
    sys.exit(0)

# Emit a non-matching header/noise line first to ensure filtering works.
print("build: fake llama-bench")
if args.n > 0:
    avg_ts = 100.0 + args.d * 0.01
    stddev_ts = 1.5
else:
    avg_ts = 250.0 * args.p
    stddev_ts = 5.0

print(json.dumps({
    "model_filename": args.m,
    "n_gpu_layers": args.ngl,
    "n_prompt": args.p,
    "n_gen": args.n,
    "n_depth": args.d,
    "n_rep": args.r,
    "avg_ts": avg_ts,
    "stddev_ts": stddev_ts,
}))
"""

FAKE_NVIDIA_SMI = """#!/usr/bin/env python3
import os
import sys

state_file = os.environ.get("FAKE_SMI_STATE")
count = 0
if state_file and os.path.exists(state_file):
    with open(state_file, "r", encoding="utf-8") as f:
        count = int(f.read().strip())
if state_file:
    with open(state_file, "w", encoding="utf-8") as f:
        f.write(str(count + 1))

if os.environ.get("FAKE_SMI_FAIL") == "1":
    sys.stderr.write("NVIDIA-SMI has failed\\n")
    sys.exit(1)

used = 4096 + count * 128
print(f"{used}, 8192, 3, 16")
"""


class TestA1GpuBench(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self.tmp.name)
        self.bin_dir = self.root / "bin"
        self.bin_dir.mkdir()

        self.fake_bench = self.bin_dir / "llama-bench"
        self.fake_bench.write_text(FAKE_LLAMA_BENCH, encoding="utf-8")
        self.fake_bench.chmod(0o755)

        self.fake_smi = self.bin_dir / "nvidia-smi"
        self.fake_smi.write_text(FAKE_NVIDIA_SMI, encoding="utf-8")
        self.fake_smi.chmod(0o755)

        self.model = self.root / "target.gguf"
        self.model.write_bytes(b"GGUF-fake-model-bytes")
        self.model_sha256 = hashlib.sha256(b"GGUF-fake-model-bytes").hexdigest()

        self.out_dir = self.root / "run_a1"
        self.log_path = self.root / "bench_calls.jsonl"
        self.smi_state = self.root / "smi_count.txt"

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _env(self, extra: dict[str, str] | None = None) -> dict[str, str]:
        env = {
            "PATH": f"{self.bin_dir}:{os.environ.get('PATH', '')}",
            "FAKE_BENCH_LOG": str(self.log_path),
            "FAKE_SMI_STATE": str(self.smi_state),
        }
        if extra:
            env.update(extra)
        return env

    def test_happy_path_metrics_and_manifest(self) -> None:
        with mock.patch.dict(os.environ, self._env()):
            rc = a1.main([
                "--out", str(self.out_dir),
                "--model", str(self.model),
                "--llama-bench", str(self.fake_bench),
                "--depths", "0,512",
                "--verify-n", "1,4",
                "--reps", "3",
            ])
        self.assertEqual(rc, 0)

        rows = [
            json.loads(line)
            for line in (self.out_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        # 2 decode + 2 * 2 verify = 6 rows
        self.assertEqual(len(rows), 6)

        decode_rows = [r for r in rows if r["kind"] == "decode"]
        verify_rows = [r for r in rows if r["kind"] == "verify"]
        self.assertEqual([(r["depth"], r["n"]) for r in decode_rows], [(0, 128), (512, 128)])
        self.assertEqual(
            [(r["depth"], r["n"]) for r in verify_rows],
            [(0, 1), (512, 1), (0, 4), (512, 4)],
        )

        self.assertAlmostEqual(decode_rows[0]["avg_ts"], 100.0)
        self.assertAlmostEqual(decode_rows[0]["t_t_ms"], 1000.0 / 100.0)
        self.assertIsNone(decode_rows[0]["error"])

        self.assertAlmostEqual(decode_rows[1]["avg_ts"], 105.12)
        self.assertAlmostEqual(decode_rows[1]["t_t_ms"], 1000.0 / 105.12)

        # verify avg_ts = 250.0 * n -> t_v_ms = 1000 * n / (250 * n) = 4.0
        for vr in verify_rows:
            self.assertAlmostEqual(vr["avg_ts"], 250.0 * vr["n"])
            self.assertAlmostEqual(vr["t_v_ms"], 4.0)
            self.assertIsNone(vr["error"])

        # Check exact llama-bench CLI invocations
        calls = [json.loads(ln) for ln in self.log_path.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(calls), 6)
        self.assertEqual(
            calls[0],
            ["-m", str(self.model), "-ngl", "99", "-p", "0", "-n", "128", "-d", "0", "-r", "3", "-o", "jsonl"],
        )
        self.assertEqual(
            calls[4],
            ["-m", str(self.model), "-ngl", "99", "-p", "4", "-n", "0", "-d", "0", "-r", "3", "-o", "jsonl"],
        )

        manifest = json.loads((self.out_dir / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["run_id"], "run_a1")
        self.assertEqual(len(manifest["files"]), 1)
        self.assertEqual(manifest["files"][0]["path"], str(self.model))
        self.assertEqual(manifest["files"][0]["sha256"], self.model_sha256)
        self.assertEqual(manifest["extra"]["nvidia_smi_before"], "4096, 8192, 3, 16")
        self.assertEqual(manifest["extra"]["nvidia_smi_after"], "4224, 8192, 3, 16")
        self.assertEqual(manifest["extra"]["options"]["depths"], [0, 512])
        self.assertEqual(manifest["extra"]["options"]["verify_n"], [1, 4])
        self.assertEqual(manifest["extra"]["options"]["reps"], 3)

    def test_nvidia_smi_missing_or_failing_records_none(self) -> None:
        # Missing from PATH
        out1 = self.root / "out_no_smi"
        with mock.patch.dict(os.environ, {"PATH": "/nonexistent_dir_xyz"}):
            rc = a1.main([
                "--out", str(out1),
                "--model", str(self.model),
                "--llama-bench", str(self.fake_bench),
                "--depths", "0",
                "--verify-n", "1",
                "--reps", "1",
            ])
        self.assertEqual(rc, 0)
        m1 = json.loads((out1 / "manifest.json").read_text(encoding="utf-8"))
        self.assertIsNone(m1["extra"]["nvidia_smi_before"])
        self.assertIsNone(m1["extra"]["nvidia_smi_after"])

        # Present on PATH but exiting non-zero
        out2 = self.root / "out_fail_smi"
        with mock.patch.dict(os.environ, self._env({"FAKE_SMI_FAIL": "1"})):
            rc2 = a1.main([
                "--out", str(out2),
                "--model", str(self.model),
                "--llama-bench", str(self.fake_bench),
                "--depths", "0",
                "--verify-n", "1",
                "--reps", "1",
            ])
        self.assertEqual(rc2, 0)
        m2 = json.loads((out2 / "manifest.json").read_text(encoding="utf-8"))
        self.assertIsNone(m2["extra"]["nvidia_smi_before"])
        self.assertIsNone(m2["extra"]["nvidia_smi_after"])

    def test_failed_llama_bench_records_error_and_continues(self) -> None:
        # Non-zero exit on decode, succeeds on verify
        with mock.patch.dict(os.environ, self._env({"FAKE_BENCH_MODE": "fail_decode"})):
            rc = a1.main([
                "--out", str(self.out_dir),
                "--model", str(self.model),
                "--llama-bench", str(self.fake_bench),
                "--depths", "0",
                "--verify-n", "1,2",
                "--reps", "1",
            ])
        self.assertEqual(rc, 0)
        rows = [
            json.loads(ln)
            for ln in (self.out_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[0]["kind"], "decode")
        self.assertIsNotNone(rows[0]["error"])
        self.assertIn("exited with 1", rows[0]["error"])
        self.assertIsNone(rows[0]["t_t_ms"])
        self.assertIsNone(rows[1]["error"])
        self.assertIsNone(rows[2]["error"])

        # Bad JSON / missing matching record
        out_bad = self.root / "out_bad_json"
        with mock.patch.dict(os.environ, self._env({"FAKE_BENCH_MODE": "bad_json"})):
            rc2 = a1.main([
                "--out", str(out_bad),
                "--model", str(self.model),
                "--llama-bench", str(self.fake_bench),
                "--depths", "0",
                "--verify-n", "2",
                "--reps", "1",
            ])
        self.assertEqual(rc2, 0)
        rows2 = [
            json.loads(ln)
            for ln in (out_bad / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        self.assertIsNone(rows2[0]["error"])
        self.assertIn("no matching verify jsonl record", rows2[1]["error"])
        self.assertIsNone(rows2[1]["t_v_ms"])

        # Missing llama-bench binary
        out_missing = self.root / "out_missing_bench"
        with mock.patch.dict(os.environ, self._env()):
            rc3 = a1.main([
                "--out", str(out_missing),
                "--model", str(self.model),
                "--llama-bench", str(self.root / "no_such_llama_bench"),
                "--depths", "0",
                "--verify-n", "1",
                "--reps", "1",
            ])
        self.assertEqual(rc3, 0)
        rows3 = [
            json.loads(ln)
            for ln in (out_missing / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        self.assertEqual(len(rows3), 2)
        self.assertIn("failed to run llama-bench", rows3[0]["error"])
        self.assertIn("failed to run llama-bench", rows3[1]["error"])

    def test_invalid_options_exit_2(self) -> None:
        existing = self.root / "already_exists"
        existing.mkdir()

        cases = [
            ["--out", str(existing), "--model", str(self.model)],
            ["--out", str(self.out_dir), "--model", str(self.root / "missing.gguf")],
            ["--out", str(self.out_dir), "--model", str(self.model), "--depths", "-1"],
            ["--out", str(self.out_dir), "--model", str(self.model), "--verify-n", "0"],
            ["--out", str(self.out_dir), "--model", str(self.model), "--reps", "0"],
            ["--out", str(self.out_dir), "--model", str(self.model), "--unknown-flag"],
        ]
        for argv in cases:
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                rc = a1.main(argv)
            self.assertEqual(rc, 2, f"expected exit 2 for {argv}")
            self.assertFalse(self.out_dir.exists(), f"--out should not be created for {argv}")

    def test_default_llama_bench_path_and_module_entrypoint(self) -> None:
        expected_bench = pathlib.Path(a1.__file__).resolve().parents[2] / "third_party/llama.cpp/build/bin/llama-bench"
        self.assertEqual(a1.DEFAULT_LLAMA_BENCH, expected_bench)

        repo_root = pathlib.Path(__file__).resolve().parents[1]
        env = dict(os.environ)
        env["PYTHONPATH"] = str(repo_root / "src")
        env.update(self._env())
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "measure.a1_gpu_bench",
                "--out",
                str(self.out_dir),
                "--model",
                str(self.model),
                "--llama-bench",
                str(self.fake_bench),
                "--depths",
                "0",
                "--verify-n",
                "1",
                "--reps",
                "1",
            ],
            cwd=str(repo_root),
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertTrue((self.out_dir / "metrics.jsonl").is_file())
        self.assertTrue((self.out_dir / "manifest.json").is_file())


if __name__ == "__main__":
    unittest.main()
