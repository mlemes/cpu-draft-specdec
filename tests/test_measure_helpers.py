"""Unit tests for src/measure/ helpers (resctrl, manifest, stats)."""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import tempfile
import unittest

from measure.manifest import collect_system, write_manifest
from measure.resctrl import Resctrl
from measure.stats import (
    bootstrap_ci,
    break_even_alpha,
    expected_tokens,
    paired_bootstrap_ratio_ci,
    speedup,
)


class TestResctrl(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self.tmp.name)
        (self.root / "info" / "L3").mkdir(parents=True)
        (self.root / "info" / "L3" / "cbm_mask").write_text("ffff\n", encoding="utf-8")
        (self.root / "info" / "L3" / "min_cbm_bits").write_text("2\n", encoding="utf-8")
        (self.root / "info" / "L3" / "num_closids").write_text("16\n", encoding="utf-8")
        (self.root / "info" / "L3_MON").mkdir(parents=True)
        (self.root / "info" / "L3_MON" / "mon_features").write_text(
            "llc_occupancy\nmbm_total_bytes\n", encoding="utf-8"
        )
        (self.root / "mon_data" / "mon_L3_00").mkdir(parents=True)
        (self.root / "mon_data" / "mon_L3_00" / "mbm_total_bytes").write_text("12345\n", encoding="utf-8")
        (self.root / "mon_data" / "mon_L3_00" / "llc_occupancy").write_text("67890\n", encoding="utf-8")
        self.rc = Resctrl(self.root)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_info_and_missing_mon_features(self) -> None:
        info = self.rc.info()
        self.assertEqual(info["cbm_mask"], 0xFFFF)
        self.assertEqual(info["min_cbm_bits"], 2)
        self.assertEqual(info["num_closids"], 16)
        self.assertEqual(info["mon_features"], ["llc_occupancy", "mbm_total_bytes"])

        (self.root / "info" / "L3_MON" / "mon_features").unlink()
        self.assertEqual(self.rc.info()["mon_features"], [])

    def test_ways_mask(self) -> None:
        self.assertEqual(self.rc.ways_mask(4), 0x000F)
        self.assertEqual(self.rc.ways_mask(4, offset=4), 0x00F0)
        with self.assertRaises(ValueError):
            self.rc.ways_mask(1)
        with self.assertRaises(ValueError):
            self.rc.ways_mask(4, offset=13)
        with self.assertRaises(ValueError):
            self.rc.ways_mask(2, offset=-1)

    def test_group_lifecycle_and_validation(self) -> None:
        with self.assertRaises(ValueError):
            self.rc.create_group("bad/name")
        with self.assertRaises(ValueError):
            self.rc.create_group("")

        g = self.rc.create_group("exp_a0-1")
        self.assertTrue(g.is_dir())
        # Idempotent creation
        self.assertEqual(self.rc.create_group("exp_a0-1"), g)

        self.rc.set_l3_mask("exp_a0-1", 0x00FF, cache_id=0)
        self.assertEqual((g / "schemata").read_text(encoding="utf-8"), "L3:0=ff\n")

        for bad_mask in (0, 0x0101, 0x0001, 0x10000):
            with self.assertRaises(ValueError):
                self.rc.set_l3_mask("exp_a0-1", bad_mask)

        self.rc.assign("exp_a0-1", 4242)
        self.assertEqual((g / "tasks").read_text(encoding="utf-8"), "4242\n")

        (g / "mon_data" / "mon_L3_00").mkdir(parents=True)
        (g / "mon_data" / "mon_L3_00" / "mbm_total_bytes").write_text("999\n", encoding="utf-8")
        (g / "mon_data" / "mon_L3_00" / "llc_occupancy").write_text("111\n", encoding="utf-8")
        self.assertEqual(self.rc.read_mbm_total("exp_a0-1"), 999)
        self.assertEqual(self.rc.read_occupancy("exp_a0-1"), 111)

        # Root group monitoring
        self.assertEqual(self.rc.read_mbm_total(""), 12345)
        self.assertEqual(self.rc.read_occupancy(""), 67890)

        # remove_group uses rmdir semantics: non-empty raises OSError
        with self.assertRaises(OSError):
            self.rc.remove_group("exp_a0-1")

        # Empty group removes cleanly, missing group raises OSError
        empty = self.rc.create_group("empty_grp")
        self.rc.remove_group("empty_grp")
        self.assertFalse(empty.exists())
        with self.assertRaises(OSError):
            self.rc.remove_group("empty_grp")


class TestManifest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_collect_system_missing_and_populated(self) -> None:
        empty = collect_system(self.root / "missing")
        self.assertEqual(
            set(empty.keys()),
            {
                "kernel",
                "cpu_model",
                "nproc",
                "smt_control",
                "governor",
                "boost",
                "thp",
                "mem_total_kib",
                "swap_used_kib",
            },
        )
        self.assertTrue(all(v is None for v in empty.values()))

        sysroot = self.root / "sysroot"
        (sysroot / "proc" / "sys" / "kernel").mkdir(parents=True)
        (sysroot / "proc" / "sys" / "kernel" / "osrelease").write_text("7.0.0-test\n", encoding="utf-8")
        (sysroot / "proc" / "cpuinfo").write_text(
            "processor\t: 0\nmodel name\t: AMD Ryzen 5 5500X3D\n\n"
            "processor\t: 1\nmodel name\t: AMD Ryzen 5 5500X3D\n",
            encoding="utf-8",
        )
        (sysroot / "proc" / "meminfo").write_text(
            "MemTotal:       32000000 kB\nSwapTotal:       2000000 kB\nSwapFree:        1500000 kB\n",
            encoding="utf-8",
        )
        (sysroot / "sys" / "devices" / "system" / "cpu" / "smt").mkdir(parents=True)
        (sysroot / "sys" / "devices" / "system" / "cpu" / "smt" / "control").write_text("off\n", encoding="utf-8")
        (sysroot / "sys" / "devices" / "system" / "cpu" / "cpufreq" / "policy0").mkdir(parents=True)
        (sysroot / "sys" / "devices" / "system" / "cpu" / "cpufreq" / "policy0" / "scaling_governor").write_text(
            "performance\n", encoding="utf-8"
        )
        (sysroot / "sys" / "devices" / "system" / "cpu" / "cpufreq" / "boost").write_text("1\n", encoding="utf-8")
        (sysroot / "sys" / "kernel" / "mm" / "transparent_hugepage").mkdir(parents=True)
        (sysroot / "sys" / "kernel" / "mm" / "transparent_hugepage" / "enabled").write_text(
            "[always] madvise never\n", encoding="utf-8"
        )

        got = collect_system(sysroot)
        self.assertEqual(got["kernel"], "7.0.0-test")
        self.assertEqual(got["cpu_model"], "AMD Ryzen 5 5500X3D")
        self.assertEqual(got["nproc"], 2)
        self.assertEqual(got["smt_control"], "off")
        self.assertEqual(got["governor"], "performance")
        self.assertEqual(got["boost"], "1")
        self.assertEqual(got["thp"], "[always] madvise never")
        self.assertEqual(got["mem_total_kib"], 32000000)
        self.assertEqual(got["swap_used_kib"], 500000)

    def test_write_manifest(self) -> None:
        payload = b"hello world\n" * 100
        artifact = self.root / "sample.bin"
        artifact.write_bytes(payload)
        run_dir = self.root / "runs" / "r1"
        manifest = write_manifest(
            run_dir,
            run_id="r1",
            extra={"gamma": 4},
            files=[artifact],
            sysroot=self.root / "missing",
        )
        self.assertEqual(manifest["run_id"], "r1")
        self.assertTrue(str(manifest["utc_timestamp"]).endswith("Z"))
        self.assertEqual(manifest["extra"], {"gamma": 4})
        self.assertEqual(len(manifest["files"]), 1)
        self.assertEqual(manifest["files"][0]["bytes"], len(payload))
        self.assertEqual(manifest["files"][0]["sha256"], hashlib.sha256(payload).hexdigest())

        loaded = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(loaded, manifest)


class TestStats(unittest.TestCase):
    def test_bootstrap_ci(self) -> None:
        data = [10.0, 11.0, 12.0, 13.0, 14.0]
        r1 = bootstrap_ci(data, n_boot=500, seed=42)
        r2 = bootstrap_ci(data, n_boot=500, seed=42)
        self.assertEqual(r1, r2)
        mean, lo, hi = r1
        self.assertAlmostEqual(mean, 12.0)
        self.assertLessEqual(lo, mean)
        self.assertLessEqual(mean, hi)
        with self.assertRaises(ValueError):
            bootstrap_ci([])

    def test_paired_bootstrap_ratio_ci(self) -> None:
        a = [20.0, 22.0, 24.0, 26.0, 28.0]
        b = [10.0, 11.0, 12.0, 13.0, 14.0]
        r1 = paired_bootstrap_ratio_ci(a, b, n_boot=500, seed=7)
        r2 = paired_bootstrap_ratio_ci(a, b, n_boot=500, seed=7)
        self.assertEqual(r1, r2)
        ratio, lo, hi = r1
        self.assertAlmostEqual(ratio, 2.0)
        self.assertAlmostEqual(lo, 2.0)
        self.assertAlmostEqual(hi, 2.0)
        with self.assertRaises(ValueError):
            paired_bootstrap_ratio_ci([1.0], [1.0, 2.0])
        with self.assertRaises(ValueError):
            paired_bootstrap_ratio_ci([1.0, 2.0], [0.0, 0.0])

    def test_specdec_formulas(self) -> None:
        self.assertAlmostEqual(expected_tokens(1.0, 4), 5.0)
        self.assertAlmostEqual(expected_tokens(0.0, 4), 1.0)
        self.assertAlmostEqual(expected_tokens(0.5, 3), 1.875)
        with self.assertRaises(ValueError):
            expected_tokens(-0.1, 4)
        with self.assertRaises(ValueError):
            expected_tokens(0.5, 0)

        sp = speedup(0.5, 4, 2.5, 15.0, 18.0, 0.3)
        self.assertAlmostEqual(sp, expected_tokens(0.5, 4) * 18.0 / 25.3)

        be = break_even_alpha(4, 2.5, 15, 18, 0.3)
        self.assertIsNotNone(be)
        assert be is not None
        self.assertGreaterEqual(speedup(be, 4, 2.5, 15, 18, 0.3), 1.0)
        self.assertLess(speedup(be - 2e-4, 4, 2.5, 15, 18, 0.3), 1.0)

        # Unreachable target when speedup(1.0) < target
        self.assertIsNone(break_even_alpha(4, 10.0, 20.0, 5.0, target=1.0))


if __name__ == "__main__":
    unittest.main()
