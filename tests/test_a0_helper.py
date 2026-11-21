"""Unit tests for src/measure/a0_counter_validation.py."""

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
import time
import unittest
from unittest import mock

from measure import a0_counter_validation as a0


FAKE_BENCH_SCRIPT = """#!/usr/bin/env python3
import argparse
import json
import os
import pathlib
import signal
import sys
import time

parser = argparse.ArgumentParser()
parser.add_argument("--bytes", type=int, required=True)
parser.add_argument("--passes", type=int, required=True)
parser.add_argument("--cpu", type=int, required=True)
parser.add_argument("--wait-signal", action="store_true")
args = parser.parse_args()

mode = os.environ.get("FAKE_BENCH_MODE", "normal")
if mode == "early_exit":
    sys.exit(3)
if mode == "no_ready_hang":
    while True:
        time.sleep(0.1)

got_sig = False
def _on_usr1(_sig, _frame):
    global got_sig
    got_sig = True

signal.signal(signal.SIGUSR1, _on_usr1)
print(json.dumps({"ready": os.getpid()}), flush=True)

if mode == "hang":
    while True:
        time.sleep(0.1)

if args.wait_signal:
    while not got_sig:
        time.sleep(0.005)

delay = float(os.environ.get("FAKE_BENCH_DELAY", "0"))
if delay > 0:
    time.sleep(delay)

mon_dir_env = os.environ.get("FAKE_MON_DIR")
if mon_dir_env:
    mon_dir = pathlib.Path(mon_dir_env)
    cur = int((mon_dir / "mbm_total_bytes").read_text(encoding="utf-8").strip())
    cat_bytes = int(os.environ.get("FAKE_CAT_BYTES", "16777216"))
    if args.bytes >= 2 * cat_bytes:
        delta = args.bytes * args.passes
    elif args.bytes <= 0.5 * cat_bytes:
        delta = int(0.05 * args.bytes * args.passes)
    else:
        delta = int(0.5 * args.bytes * args.passes)
    (mon_dir / "mbm_total_bytes").write_text(f"{cur + delta}\\n", encoding="utf-8")
    (mon_dir / "llc_occupancy").write_text(f"{min(args.bytes, cat_bytes)}\\n", encoding="utf-8")

for i in range(args.passes):
    print(json.dumps({"pass": i, "bytes": args.bytes, "ns": 1000000, "gbps": args.bytes / 1e6}), flush=True)
print(json.dumps({"summary": True, "bytes": args.bytes, "passes": args.passes, "median_ns": 1000000, "checksum": 42}), flush=True)
"""

FAKE_PERF_SCRIPT = """#!/usr/bin/env python3
import signal
import sys
import time

running = True
def _stop(_sig, _frame):
    global running
    running = False

signal.signal(signal.SIGINT, _stop)
while running:
    time.sleep(0.01)

sys.stderr.write("12345,,LLC-loads,1000,100.00,,\\n")
sys.stderr.write("678,,LLC-load-misses,1000,100.00,,\\n")
sys.stderr.write("<not counted>,,cache-misses,0,0.00,,\\n")
sys.stderr.flush()
"""


class TestA0CounterValidation(unittest.TestCase):
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

        self.mon_dir = self.resctrl / "a0bench" / "mon_data" / "mon_L3_00"
        self.mon_dir.mkdir(parents=True)
        (self.mon_dir / "mbm_total_bytes").write_text("1000\n", encoding="utf-8")
        (self.mon_dir / "llc_occupancy").write_text("0\n", encoding="utf-8")

        self.bench = self.root / "fake_bench.py"
        self.bench.write_text(FAKE_BENCH_SCRIPT, encoding="utf-8")
        self.bench.chmod(0o755)

        self.perf = self.root / "fake_perf.py"
        self.perf.write_text(FAKE_PERF_SCRIPT, encoding="utf-8")
        self.perf.chmod(0o755)

        self.out_dir = self.root / "out"

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_happy_path_with_perf_and_summary(self) -> None:
        # W=16, l3_bytes=64 MiB -> w=4 gives cat_bytes=16 MiB.
        # sizes_mb=4,8,32,64 -> below=(4,8), above=(32,64)
        env_patch = {
            "FAKE_MON_DIR": str(self.mon_dir),
            "FAKE_CAT_BYTES": str(16 * 1024 * 1024),
            "FAKE_BENCH_DELAY": "0.15",
        }
        with mock.patch.dict(os.environ, env_patch):
            rc = a0.main([
                "--out", str(self.out_dir),
                "--resctrl-root", str(self.resctrl),
                "--bench", str(self.bench),
                "--sizes-mb", "4,8,32,64",
                "--ways", "4",
                "--reserve-ways", "2",
                "--passes", "3",
                "--cpu", "0",
                "--l3-bytes", str(64 * 1024 * 1024),
                "--perf", str(self.perf),
                "--perf-events", "LLC-loads,LLC-load-misses",
            ])
        self.assertEqual(rc, 0)
        self.assertEqual((self.resctrl / "schemata").read_text(encoding="utf-8"), "L3:0=ffff\n")

        rows = [
            json.loads(line)
            for line in (self.out_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.assertEqual(len(rows), 4)
        for row in rows:
            self.assertIsNone(row["error"])
            self.assertEqual(row["ways"], 4)
            self.assertEqual(
                row["perf"],
                {"LLC-loads": 12345, "LLC-load-misses": 678, "cache-misses": None},
            )

        summary = json.loads((self.out_dir / "summary.json").read_text(encoding="utf-8"))
        # One allocation cannot show monotone behaviour across ways, so the revised criterion fails overall.
        self.assertFalse(summary["passed"])
        self.assertTrue(summary["criteria"]["accuracy"]["passed"])
        self.assertEqual(summary["criteria"]["accuracy"]["runs"], 2)
        self.assertTrue(summary["criteria"]["separation"]["passed"])
        self.assertFalse(summary["criteria"]["monotone"]["passed"])
        self.assertEqual(summary["ways"]["4"]["total_count"], 4)

        manifest = json.loads((self.out_dir / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["extra"]["W"], 16)
        self.assertEqual(manifest["extra"]["l3_bytes"], 64 * 1024 * 1024)
        self.assertIn("options", manifest["extra"])
        self.assertIn("info", manifest["extra"])

    def test_invalid_options_exit_2_without_touching_resctrl(self) -> None:
        # Remove a0bench to verify it is not created on invalid input
        (self.mon_dir / "mbm_total_bytes").unlink()
        (self.mon_dir / "llc_occupancy").unlink()
        self.mon_dir.rmdir()
        (self.resctrl / "a0bench" / "mon_data").rmdir()
        (self.resctrl / "a0bench").rmdir()

        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            rc = a0.main([
                "--out", str(self.out_dir),
                "--resctrl-root", str(self.resctrl),
                "--bench", str(self.bench),
                "--ways", "15",
                "--reserve-ways", "2",
            ])
        self.assertEqual(rc, 2)
        self.assertEqual((self.resctrl / "schemata").read_text(encoding="utf-8"), self.original_schemata)
        self.assertFalse((self.resctrl / "a0bench").exists())
        self.assertFalse(self.out_dir.exists())

    def test_permission_error_exits_1_and_restores_root_l3(self) -> None:
        stderr = io.StringIO()
        with mock.patch("measure.resctrl.Resctrl.assign", side_effect=PermissionError("denied")):
            with contextlib.redirect_stderr(stderr):
                rc = a0.main([
                    "--out", str(self.out_dir),
                    "--resctrl-root", str(self.resctrl),
                    "--bench", str(self.bench),
                    "--sizes-mb", "4",
                    "--ways", "4",
                    "--reserve-ways", "2",
                    "--passes", "1",
                ])
        self.assertEqual(rc, 1)
        self.assertIn("needs root: run with sudo", stderr.getvalue())
        self.assertEqual((self.resctrl / "schemata").read_text(encoding="utf-8"), "L3:0=ffff\n")

    def test_early_exit_bench_records_error_and_continues(self) -> None:
        with mock.patch.dict(os.environ, {"FAKE_BENCH_MODE": "early_exit"}):
            rc = a0.main([
                "--out", str(self.out_dir),
                "--resctrl-root", str(self.resctrl),
                "--bench", str(self.bench),
                "--sizes-mb", "4",
                "--ways", "4",
                "--reserve-ways", "2",
                "--passes", "1",
            ])
        self.assertEqual(rc, 0)
        rows = [
            json.loads(line)
            for line in (self.out_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.assertEqual(len(rows), 1)
        self.assertIsNotNone(rows[0]["error"])
        summary = json.loads((self.out_dir / "summary.json").read_text(encoding="utf-8"))
        self.assertFalse(summary["passed"])
        self.assertEqual((self.resctrl / "schemata").read_text(encoding="utf-8"), "L3:0=ffff\n")

    def test_no_ready_timeout_restores_root_l3(self) -> None:
        with mock.patch.object(a0, "READY_TIMEOUT_S", 0.1):
            with mock.patch.dict(os.environ, {"FAKE_BENCH_MODE": "no_ready_hang"}):
                rc = a0.main([
                    "--out", str(self.out_dir),
                    "--resctrl-root", str(self.resctrl),
                    "--bench", str(self.bench),
                    "--sizes-mb", "4",
                    "--ways", "4",
                    "--reserve-ways", "2",
                    "--passes", "1",
                ])
        self.assertEqual(rc, 0)
        rows = [
            json.loads(line)
            for line in (self.out_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.assertEqual(len(rows), 1)
        self.assertIn("timeout waiting for ready line", rows[0]["error"])
        summary = json.loads((self.out_dir / "summary.json").read_text(encoding="utf-8"))
        self.assertFalse(summary["passed"])
        self.assertEqual((self.resctrl / "schemata").read_text(encoding="utf-8"), "L3:0=ffff\n")

    def _run_signal_test(self, sig: signal.Signals) -> None:
        env = dict(os.environ)
        src_dir = str(pathlib.Path(__file__).resolve().parent.parent / "src")
        env["PYTHONPATH"] = src_dir
        env["FAKE_BENCH_MODE"] = "hang"

        proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "measure.a0_counter_validation",
                "--out",
                str(self.out_dir),
                "--resctrl-root",
                str(self.resctrl),
                "--bench",
                str(self.bench),
                "--sizes-mb",
                "4",
                "--ways",
                "4",
                "--reserve-ways",
                "2",
                "--passes",
                "1",
            ],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            tasks_file = self.resctrl / "a0bench" / "tasks"
            deadline = time.monotonic() + 5.0
            while not tasks_file.exists() and time.monotonic() < deadline:
                time.sleep(0.02)
            self.assertTrue(tasks_file.exists(), "bench never assigned to a0bench")
            bench_pid = int(tasks_file.read_text(encoding="utf-8").strip())

            proc.send_signal(sig)
            rc = proc.wait(timeout=5)
            self.assertEqual(rc, 128 + sig.value)
            self.assertEqual((self.resctrl / "schemata").read_text(encoding="utf-8"), "L3:0=ffff\n")

            # Verify the hanging bench process was killed and reaped
            dead = False
            for _ in range(50):
                try:
                    os.kill(bench_pid, 0)
                except ProcessLookupError:
                    dead = True
                    break
                time.sleep(0.02)
            self.assertTrue(dead, f"bench pid {bench_pid} still alive after {sig.name}")
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()
            if proc.stdout is not None:
                proc.stdout.close()
            if proc.stderr is not None:
                proc.stderr.close()

    def test_sigint_restores_root_l3_and_kills_bench(self) -> None:
        self._run_signal_test(signal.SIGINT)

    def test_sigterm_restores_root_l3_and_kills_bench(self) -> None:
        self._run_signal_test(signal.SIGTERM)

    def test_existing_out_rejected_with_exit_2(self) -> None:
        target_dir = self.root / "real_target"
        target_dir.mkdir()
        sym_out = self.root / "sym_out"
        sym_out.symlink_to(target_dir)

        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            rc = a0.main([
                "--out", str(sym_out),
                "--resctrl-root", str(self.resctrl),
                "--bench", str(self.bench),
                "--sizes-mb", "4",
                "--ways", "4",
                "--reserve-ways", "2",
                "--passes", "1",
            ])
        self.assertEqual(rc, 2)
        self.assertFalse((target_dir / "metrics.jsonl").exists())

        nonempty_out = self.root / "nonempty_out"
        nonempty_out.mkdir()
        (nonempty_out / "existing.txt").write_text("hi\n", encoding="utf-8")
        stderr2 = io.StringIO()
        with contextlib.redirect_stderr(stderr2):
            rc2 = a0.main([
                "--out", str(nonempty_out),
                "--resctrl-root", str(self.resctrl),
                "--bench", str(self.bench),
                "--sizes-mb", "4",
                "--ways", "4",
                "--reserve-ways", "2",
                "--passes", "1",
            ])
        self.assertEqual(rc2, 2)

        empty_out = self.root / "empty_out"
        empty_out.mkdir()
        stderr3 = io.StringIO()
        with contextlib.redirect_stderr(stderr3):
            rc3 = a0.main([
                "--out", str(empty_out),
                "--resctrl-root", str(self.resctrl),
                "--bench", str(self.bench),
                "--sizes-mb", "4",
                "--ways", "4",
                "--reserve-ways", "2",
                "--passes", "1",
            ])
        self.assertEqual(rc3, 2)
        self.assertFalse((empty_out / "metrics.jsonl").exists())
        self.assertEqual((self.resctrl / "schemata").read_text(encoding="utf-8"), self.original_schemata)

    def test_group_in_use_exits_1_before_changing_anything(self) -> None:
        (self.resctrl / "a0bench" / "tasks").write_text("12345\n", encoding="utf-8")
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            rc = a0.main([
                "--out", str(self.out_dir),
                "--resctrl-root", str(self.resctrl),
                "--bench", str(self.bench),
                "--sizes-mb", "4",
                "--ways", "4",
                "--reserve-ways", "2",
                "--passes", "1",
            ])
        self.assertEqual(rc, 1)
        self.assertIn("group a0bench in use", stderr.getvalue())
        self.assertEqual((self.resctrl / "schemata").read_text(encoding="utf-8"), self.original_schemata)
        self.assertFalse(self.out_dir.exists())

    def test_stale_empty_group_continues(self) -> None:
        (self.resctrl / "a0bench" / "tasks").write_text("", encoding="utf-8")
        rc = a0.main([
            "--out", str(self.out_dir),
            "--resctrl-root", str(self.resctrl),
            "--bench", str(self.bench),
            "--sizes-mb", "4",
            "--ways", "4",
            "--reserve-ways", "2",
            "--passes", "1",
        ])
        self.assertEqual(rc, 0)

    def test_unexpected_exception_exits_1_and_restores_root_l3(self) -> None:
        stderr = io.StringIO()
        with mock.patch.object(a0, "_run_single", side_effect=RuntimeError("boom")):
            with contextlib.redirect_stderr(stderr):
                rc = a0.main([
                    "--out", str(self.out_dir),
                    "--resctrl-root", str(self.resctrl),
                    "--bench", str(self.bench),
                    "--sizes-mb", "4",
                    "--ways", "4",
                    "--reserve-ways", "2",
                    "--passes", "1",
                ])
        self.assertEqual(rc, 1)
        self.assertIn("error: boom", stderr.getvalue())
        self.assertEqual((self.resctrl / "schemata").read_text(encoding="utf-8"), "L3:0=ffff\n")

    def test_restore_failure_exits_3_precedence(self) -> None:
        stderr = io.StringIO()
        with mock.patch.object(a0, "_restore_root_l3_line", side_effect=OSError("io error")):
            with contextlib.redirect_stderr(stderr):
                rc = a0.main([
                    "--out", str(self.out_dir),
                    "--resctrl-root", str(self.resctrl),
                    "--bench", str(self.bench),
                    "--sizes-mb", "4",
                    "--ways", "4",
                    "--reserve-ways", "2",
                    "--passes", "1",
                ])
        self.assertEqual(rc, 3)
        self.assertIn("ERROR: failed to restore root L3 schemata to L3:0=ffff: io error", stderr.getvalue())

        (self.resctrl / "a0bench" / "tasks").write_text("", encoding="utf-8")
        (self.resctrl / "schemata").write_text(self.original_schemata, encoding="utf-8")
        out_dir2 = self.root / "out2"
        stderr2 = io.StringIO()
        with mock.patch("measure.resctrl.Resctrl.assign", side_effect=PermissionError("denied")):
            with mock.patch.object(a0, "_restore_root_l3_line", side_effect=OSError("io error")):
                with contextlib.redirect_stderr(stderr2):
                    rc2 = a0.main([
                        "--out", str(out_dir2),
                        "--resctrl-root", str(self.resctrl),
                        "--bench", str(self.bench),
                        "--sizes-mb", "4",
                        "--ways", "4",
                        "--reserve-ways", "2",
                        "--passes", "1",
                    ])
        self.assertEqual(rc2, 3)
        self.assertIn("ERROR: failed to restore root L3 schemata to L3:0=ffff: io error", stderr2.getvalue())

    def test_perf_early_exit_nonzero_and_no_counts(self) -> None:
        early_perf = self.root / "early_perf.py"
        early_perf.write_text(
            "#!/usr/bin/env python3\n"
            "import sys\n"
            "sys.stderr.write('perf_event_paranoid too high\\n')\n"
            "sys.exit(2)\n",
            encoding="utf-8",
        )
        early_perf.chmod(0o755)
        rc = a0.main([
            "--out", str(self.out_dir),
            "--resctrl-root", str(self.resctrl),
            "--bench", str(self.bench),
            "--sizes-mb", "4",
            "--ways", "4",
            "--reserve-ways", "2",
            "--passes", "1",
            "--perf", str(early_perf),
            "--perf-events", "LLC-loads",
        ])
        self.assertEqual(rc, 0)
        row = json.loads((self.out_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()[0])
        self.assertIsNotNone(row["error"])
        self.assertIn("perf_event_paranoid too high", row["error"])

        (self.resctrl / "a0bench" / "tasks").write_text("", encoding="utf-8")
        # Perf exits with non-zero code after SIGINT
        bad_exit_perf = self.root / "bad_exit_perf.py"
        bad_exit_perf.write_text(
            "#!/usr/bin/env python3\n"
            "import signal, sys, time\n"
            "running = True\n"
            "def _s(_a, _b):\n"
            "    global running\n"
            "    running = False\n"
            "signal.signal(signal.SIGINT, _s)\n"
            "while running:\n"
            "    time.sleep(0.01)\n"
            "sys.stderr.write('100,,LLC-loads,1000,100.00,,\\n')\n"
            "sys.exit(2)\n",
            encoding="utf-8",
        )
        bad_exit_perf.chmod(0o755)
        out2 = self.root / "out_bad_perf"
        rc2 = a0.main([
            "--out", str(out2),
            "--resctrl-root", str(self.resctrl),
            "--bench", str(self.bench),
            "--sizes-mb", "4",
            "--ways", "4",
            "--reserve-ways", "2",
            "--passes", "1",
            "--perf", str(bad_exit_perf),
            "--perf-events", "LLC-loads",
        ])
        self.assertEqual(rc2, 0)
        row2 = json.loads((out2 / "metrics.jsonl").read_text(encoding="utf-8").splitlines()[0])
        self.assertIsNotNone(row2["error"])
        self.assertIn("perf exited with code 2", row2["error"])

        (self.resctrl / "a0bench" / "tasks").write_text("", encoding="utf-8")
        # Perf produces only <not supported> / <not counted> -> mapped to None and error set
        nocount_perf = self.root / "nocount_perf.py"
        nocount_perf.write_text(
            "#!/usr/bin/env python3\n"
            "import signal, sys, time\n"
            "running = True\n"
            "def _s(_a, _b):\n"
            "    global running\n"
            "    running = False\n"
            "signal.signal(signal.SIGINT, _s)\n"
            "while running:\n"
            "    time.sleep(0.01)\n"
            "sys.stderr.write('<not supported>,,LLC-loads,0,0.00,,\\n')\n"
            "sys.stderr.write('<not counted>,,cache-misses,0,0.00,,\\n')\n",
            encoding="utf-8",
        )
        nocount_perf.chmod(0o755)
        out3 = self.root / "out_nocount_perf"
        rc3 = a0.main([
            "--out", str(out3),
            "--resctrl-root", str(self.resctrl),
            "--bench", str(self.bench),
            "--sizes-mb", "4",
            "--ways", "4",
            "--reserve-ways", "2",
            "--passes", "1",
            "--perf", str(nocount_perf),
            "--perf-events", "LLC-loads,cache-misses",
        ])
        self.assertEqual(rc3, 0)
        row3 = json.loads((out3 / "metrics.jsonl").read_text(encoding="utf-8").splitlines()[0])
        self.assertEqual(row3["perf"], {"LLC-loads": None, "cache-misses": None})
        self.assertIsNotNone(row3["error"])
        self.assertIn("no parsed counts", row3["error"])


if __name__ == "__main__":
    unittest.main()
