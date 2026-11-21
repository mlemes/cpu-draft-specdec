"""Helpers for Linux resctrl L3 CAT allocation and MBM/occupancy monitoring."""

from __future__ import annotations

import os
import pathlib
import re

_GROUP_NAME_RE = re.compile(r"^[A-Za-z0-9_-]+$")


def _is_contiguous_mask(mask: int) -> bool:
    if mask <= 0:
        return False
    lsb = mask & -mask
    shifted = mask >> (lsb.bit_length() - 1)
    return (shifted & (shifted + 1)) == 0


class Resctrl:
    """Interface to a resctrl filesystem tree (real or synthetic)."""

    def __init__(self, root: str | os.PathLike[str] = "/sys/fs/resctrl") -> None:
        self.root = pathlib.Path(root)

    def _group_dir(self, name: str, *, allow_root: bool = False) -> pathlib.Path:
        if name == "":
            if allow_root:
                return self.root
            raise ValueError("group name must not be empty")
        if not _GROUP_NAME_RE.fullmatch(name):
            raise ValueError(f"invalid resctrl group name: {name!r}")
        return self.root / name

    def info(self) -> dict[str, object]:
        l3_dir = self.root / "info" / "L3"
        cbm_mask = int((l3_dir / "cbm_mask").read_text(encoding="utf-8").strip(), 16)
        min_cbm_bits = int((l3_dir / "min_cbm_bits").read_text(encoding="utf-8").strip())
        num_closids = int((l3_dir / "num_closids").read_text(encoding="utf-8").strip())

        mon_path = self.root / "info" / "L3_MON" / "mon_features"
        try:
            mon_text = mon_path.read_text(encoding="utf-8")
            mon_features = [line.strip() for line in mon_text.splitlines() if line.strip()]
        except FileNotFoundError:
            mon_features = []

        return {
            "cbm_mask": cbm_mask,
            "min_cbm_bits": min_cbm_bits,
            "num_closids": num_closids,
            "mon_features": mon_features,
        }

    def ways_mask(self, n_ways: int, offset: int = 0) -> int:
        if offset < 0:
            raise ValueError(f"offset must be non-negative, got {offset}")
        cfg = self.info()
        min_cbm_bits = int(cfg["min_cbm_bits"])
        cbm_mask = int(cfg["cbm_mask"])
        if n_ways < min_cbm_bits:
            raise ValueError(f"n_ways ({n_ways}) < min_cbm_bits ({min_cbm_bits})")
        mask = ((1 << n_ways) - 1) << offset
        if (mask & ~cbm_mask) != 0:
            raise ValueError(
                f"mask {mask:#x} exceeds cbm_mask {cbm_mask:#x}"
            )
        return mask

    def create_group(self, name: str) -> pathlib.Path:
        group_dir = self._group_dir(name, allow_root=False)
        group_dir.mkdir(exist_ok=True)
        return group_dir

    def set_l3_mask(self, name: str, mask: int, cache_id: int = 0) -> None:
        if isinstance(mask, bool) or not isinstance(mask, int):
            raise ValueError(f"mask must be an int, got {type(mask).__name__}")
        cfg = self.info()
        cbm_mask = int(cfg["cbm_mask"])
        min_cbm_bits = int(cfg["min_cbm_bits"])
        if mask <= 0:
            raise ValueError(f"mask must be non-zero, got {mask:#x}")
        if (mask & ~cbm_mask) != 0:
            raise ValueError(f"mask {mask:#x} has bits outside cbm_mask {cbm_mask:#x}")
        if not _is_contiguous_mask(mask):
            raise ValueError(f"mask {mask:#x} is not contiguous")
        if mask.bit_count() < min_cbm_bits:
            raise ValueError(
                f"mask {mask:#x} has {mask.bit_count()} bits < min_cbm_bits ({min_cbm_bits})"
            )
        group_dir = self._group_dir(name, allow_root=True)
        (group_dir / "schemata").write_text(f"L3:{cache_id}={mask:x}\n", encoding="utf-8")

    def assign(self, name: str, pid: int) -> None:
        group_dir = self._group_dir(name, allow_root=True)
        (group_dir / "tasks").write_text(f"{int(pid)}\n", encoding="utf-8")

    def read_mbm_total(self, name: str, domain: str = "mon_L3_00") -> int:
        group_dir = self._group_dir(name, allow_root=True)
        path = group_dir / "mon_data" / domain / "mbm_total_bytes"
        return int(path.read_text(encoding="utf-8").strip())

    def read_occupancy(self, name: str, domain: str = "mon_L3_00") -> int:
        group_dir = self._group_dir(name, allow_root=True)
        path = group_dir / "mon_data" / domain / "llc_occupancy"
        return int(path.read_text(encoding="utf-8").strip())

    def remove_group(self, name: str) -> None:
        group_dir = self._group_dir(name, allow_root=False)
        os.rmdir(group_dir)
