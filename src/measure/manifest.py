"""Run manifest writer and system state collector."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
import pathlib
import subprocess
from typing import Iterable


def _read_stripped(path: pathlib.Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeDecodeError):
        return None


def _parse_cpuinfo(path: pathlib.Path) -> tuple[str | None, int | None]:
    text = _read_stripped(path)
    if text is None:
        return None, None
    cpu_model: str | None = None
    nproc = 0
    for line in text.splitlines():
        if ":" not in line:
            continue
        key, val = line.split(":", 1)
        key = key.strip()
        if key == "model name" and cpu_model is None:
            cpu_model = val.strip()
        elif key == "processor":
            nproc += 1
    return cpu_model, nproc


def _parse_meminfo(path: pathlib.Path) -> tuple[int | None, int | None]:
    text = _read_stripped(path)
    if text is None:
        return None, None
    values: dict[str, int] = {}
    for line in text.splitlines():
        if ":" not in line:
            continue
        key, rest = line.split(":", 1)
        parts = rest.strip().split()
        if not parts:
            continue
        try:
            values[key.strip()] = int(parts[0])
        except ValueError:
            continue
    mem_total = values.get("MemTotal")
    swap_total = values.get("SwapTotal")
    swap_free = values.get("SwapFree")
    swap_used = (swap_total - swap_free) if (swap_total is not None and swap_free is not None) else None
    return mem_total, swap_used


def collect_system(sysroot: str | os.PathLike[str] = "/") -> dict[str, object]:
    root = pathlib.Path(sysroot)
    cpu_model, nproc = _parse_cpuinfo(root / "proc" / "cpuinfo")
    mem_total_kib, swap_used_kib = _parse_meminfo(root / "proc" / "meminfo")
    return {
        "kernel": _read_stripped(root / "proc" / "sys" / "kernel" / "osrelease"),
        "cpu_model": cpu_model,
        "nproc": nproc,
        "smt_control": _read_stripped(root / "sys" / "devices" / "system" / "cpu" / "smt" / "control"),
        "governor": _read_stripped(
            root / "sys" / "devices" / "system" / "cpu" / "cpufreq" / "policy0" / "scaling_governor"
        ),
        "boost": _read_stripped(root / "sys" / "devices" / "system" / "cpu" / "cpufreq" / "boost"),
        "thp": _read_stripped(root / "sys" / "kernel" / "mm" / "transparent_hugepage" / "enabled"),
        "mem_total_kib": mem_total_kib,
        "swap_used_kib": swap_used_kib,
    }


def _git_commit() -> str | None:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return None
    if proc.returncode != 0:
        return None
    out = proc.stdout.strip()
    return out or None


def _sha256_file(path: pathlib.Path, chunk_size: int = 1 << 20) -> tuple[int, str]:
    h = hashlib.sha256()
    total = 0
    with path.open("rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
            total += len(chunk)
    return total, h.hexdigest()


def write_manifest(
    run_dir: str | os.PathLike[str],
    *,
    run_id: str,
    extra: dict[str, object] | None = None,
    files: Iterable[str | os.PathLike[str]] = (),
    sysroot: str | os.PathLike[str] = "/",
) -> dict[str, object]:
    out_dir = pathlib.Path(run_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    file_entries: list[dict[str, object]] = []
    for item in files:
        p = pathlib.Path(item)
        nbytes, digest = _sha256_file(p)
        file_entries.append({
            "path": str(item),
            "bytes": nbytes,
            "sha256": digest,
        })

    manifest: dict[str, object] = {
        "run_id": run_id,
        "utc_timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "system": collect_system(sysroot=sysroot),
        "git_commit": _git_commit(),
        "files": file_entries,
        "extra": dict(extra) if extra is not None else {},
    }

    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest
