import json
import os
import pathlib
import signal
import subprocess
import unittest


REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
MEASURE_DIR = REPO_ROOT / "src" / "measure"
BINARY = MEASURE_DIR / "stream_bench"


def expected_checksum(n_bytes: int) -> int:
    data = bytes((i * 131 + 7) % 256 for i in range(n_bytes))
    total = 0
    for off in range(0, n_bytes, 8):
        total = (total + int.from_bytes(data[off : off + 8], "little")) & ((1 << 64) - 1)
    return total


class TestStreamBench(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        res = subprocess.run(
            ["make", "-C", str(MEASURE_DIR), "stream_bench"],
            capture_output=True,
            text=True,
            check=False,
        )
        if res.returncode != 0:
            raise RuntimeError(f"make failed: {res.stderr}")
        if not BINARY.exists():
            raise RuntimeError(f"Expected binary at {BINARY}")

    @classmethod
    def tearDownClass(cls) -> None:
        subprocess.run(
            ["make", "-C", str(MEASURE_DIR), "clean"],
            capture_output=True,
            text=True,
            check=False,
        )

    def test_small_buffer_json_and_checksum(self) -> None:
        n_bytes = 4096
        passes = 3
        res = subprocess.run(
            [str(BINARY), "--bytes", str(n_bytes), "--passes", str(passes), "--warmup", "1"],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(res.returncode, 0, msg=res.stderr)
        lines = [line for line in res.stdout.strip().splitlines() if line]
        self.assertEqual(len(lines), passes + 1)

        pass_objs = [json.loads(line) for line in lines[:passes]]
        for idx, obj in enumerate(pass_objs):
            self.assertEqual(obj["pass"], idx)
            self.assertEqual(obj["bytes"], n_bytes)
            self.assertIsInstance(obj["ns"], int)
            self.assertGreaterEqual(obj["ns"], 0)
            self.assertIsInstance(obj["gbps"], (int, float))

        summary = json.loads(lines[-1])
        self.assertTrue(summary["summary"])
        self.assertEqual(summary["bytes"], n_bytes)
        self.assertEqual(summary["passes"], passes)
        expected_med = sorted(obj["ns"] for obj in pass_objs)[passes // 2]
        self.assertEqual(summary["median_ns"], expected_med)
        self.assertEqual(summary["checksum"], expected_checksum(n_bytes))

    def test_cpu_affinity(self) -> None:
        ok = subprocess.run(
            [str(BINARY), "--bytes", "64", "--passes", "1", "--cpu", "0"],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(ok.returncode, 0, msg=ok.stderr)

        bad = subprocess.run(
            [str(BINARY), "--bytes", "64", "--passes", "1", "--cpu", "999999"],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(bad.returncode, 1)

    def test_wait_signal(self) -> None:
        proc = subprocess.Popen(
            [str(BINARY), "--bytes", "1024", "--passes", "2", "--wait-signal"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            assert proc.stdout is not None
            ready_line = proc.stdout.readline()
            ready_obj = json.loads(ready_line)
            self.assertEqual(ready_obj["ready"], proc.pid)

            os.kill(proc.pid, signal.SIGUSR1)
            out, err = proc.communicate(timeout=5)
            self.assertEqual(proc.returncode, 0, msg=err)
            lines = [json.loads(line) for line in out.strip().splitlines() if line]
            self.assertEqual(len(lines), 3)
            self.assertTrue(lines[-1]["summary"])
            self.assertEqual(lines[-1]["checksum"], expected_checksum(1024))
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()

    def test_argument_errors_exit_2(self) -> None:
        bad_args = [
            ["--bytes", "10", "--passes", "1"],
            ["--bytes", "0", "--passes", "1"],
            ["--bytes", "64", "--passes", "0"],
            ["--bytes", "64"],
            ["--bytes", "64", "--passes", "1", "--unknown"],
            ["--bytes"],
        ]
        for args in bad_args:
            res = subprocess.run(
                [str(BINARY), *args],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(res.returncode, 2, msg=f"args={args} stderr={res.stderr}")
            self.assertIn("Usage:", res.stderr)


if __name__ == "__main__":
    unittest.main()
