import json
import pathlib
import shutil
import subprocess
import unittest


REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
MEASURE_DIR = REPO_ROOT / "src" / "measure"
BINARY = MEASURE_DIR / "pcie_bench"
HAS_NVCC = shutil.which("nvcc") is not None
HAS_GPU = pathlib.Path("/dev/nvidia0").exists()


@unittest.skipUnless(HAS_NVCC, "nvcc not installed")
class TestPcieBench(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        res = subprocess.run(
            ["make", "-C", str(MEASURE_DIR), "pcie_bench"],
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

    def test_bad_args_exit_2(self) -> None:
        bad_args = [
            ["--unknown"],
            ["--sizes", "abc"],
            ["--sizes", "1024,0"],
            ["--sizes", "1024,"],
            ["--sizes"],
            ["--iters", "0"],
            ["--iters", "-5"],
            ["--iters", "xyz"],
            ["--iters"],
            ["--warmup", "-1"],
            ["--device", "-1"],
            ["positional"],
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

    def test_execution_with_or_without_gpu(self) -> None:
        res = subprocess.run(
            [str(BINARY), "--sizes", "1024,4096", "--iters", "5", "--warmup", "1"],
            capture_output=True,
            text=True,
            check=False,
        )
        if not HAS_GPU:
            self.assertEqual(res.returncode, 1, msg=res.stderr)
            self.assertTrue(len(res.stderr.strip()) > 0)
            return

        self.assertEqual(res.returncode, 0, msg=res.stderr)
        lines = [line for line in res.stdout.strip().splitlines() if line]
        # 2 sizes * 2 dirs * 2 kinds + 1 launch_sync = 9 lines
        self.assertEqual(len(lines), 9)
        objs = [json.loads(line) for line in lines]

        for obj in objs[:-1]:
            self.assertIn(obj["kind"], ("pinned", "pageable"))
            self.assertIn(obj["dir"], ("h2d", "d2h"))
            self.assertIn(obj["bytes"], (1024, 4096))
            self.assertEqual(obj["iters"], 5)
            self.assertGreater(obj["median_us"], 0.0)
            self.assertGreaterEqual(obj["p95_us"], obj["median_us"])
            self.assertGreater(obj["gbps"], 0.0)

        launch = objs[-1]
        self.assertEqual(launch["kind"], "launch_sync")
        self.assertEqual(launch["iters"], 5)
        self.assertGreater(launch["median_us"], 0.0)
        self.assertGreaterEqual(launch["p95_us"], launch["median_us"])


if __name__ == "__main__":
    unittest.main()
