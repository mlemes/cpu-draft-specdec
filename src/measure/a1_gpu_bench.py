"""Experiment a1: GPU target benchmark driver wrapping llama-bench and nvidia-smi."""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import shutil
import subprocess
import sys
from typing import Any

from measure.manifest import write_manifest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_LLAMA_BENCH = REPO_ROOT / "third_party" / "llama.cpp" / "build" / "bin" / "llama-bench"
DEFAULT_DEPTHS = "0,512,2048"
DEFAULT_VERIFY_N = "1,2,3,4,5,6,7,8"
DEFAULT_REPS = 5
DECODE_TOKENS = 128

NVIDIA_SMI_QUERY = (
    "--query-gpu=memory.used,memory.total,pcie.link.gen.current,pcie.link.width.current"
)


def _parse_int_list(text: str, name: str, *, minimum: int = 0) -> list[int]:
    parts = [p.strip() for p in text.split(",") if p.strip()]
    if not parts:
        raise ValueError(f"{name} must not be empty")
    values: list[int] = []
    for part in parts:
        try:
            val = int(part)
        except ValueError as exc:
            raise ValueError(f"invalid integer {part!r} in {name}") from exc
        if val < minimum:
            raise ValueError(f"{name} value {val} must be >= {minimum}")
        values.append(val)
    return values


def _stderr_tail(text: str, max_lines: int = 5, max_chars: int = 400) -> str:
    lines = [ln for ln in text.strip().splitlines() if ln.strip()]
    tail = "\n".join(lines[-max_lines:])
    if len(tail) > max_chars:
        tail = tail[-max_chars:]
    return tail


def _query_nvidia_smi() -> str | None:
    exe = shutil.which("nvidia-smi")
    if exe is None:
        return None
    try:
        proc = subprocess.run(
            [exe, NVIDIA_SMI_QUERY, "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=False,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    out = proc.stdout.strip()
    return out if out else None


def _find_matching_record(stdout: str, kind: str) -> dict[str, Any] | None:
    key = "n_gen" if kind == "decode" else "n_prompt"
    for line in stdout.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        try:
            rec = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        if not isinstance(rec, dict):
            continue
        val = rec.get(key)
        if isinstance(val, (int, float)) and not isinstance(val, bool) and val > 0:
            return rec
    return None


def _to_finite_float(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    f = float(value)
    if not math.isfinite(f):
        return None
    return f


def _run_llama_bench(
    *,
    llama_bench: str,
    model_path: pathlib.Path,
    kind: str,
    depth: int,
    n_tokens: int,
    reps: int,
) -> dict[str, object]:
    if kind == "decode":
        n_prompt = 0
        n_gen = n_tokens
    else:
        n_prompt = n_tokens
        n_gen = 0

    cmd = [
        llama_bench,
        "-m",
        str(model_path),
        "-ngl",
        "99",
        "-p",
        str(n_prompt),
        "-n",
        str(n_gen),
        "-d",
        str(depth),
        "-r",
        str(reps),
        "-o",
        "jsonl",
    ]

    avg_ts: float | None = None
    stddev_ts: float | None = None
    timing_ms: float | None = None
    error: str | None = None

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    except OSError as exc:
        error = f"failed to run llama-bench: {exc}"
    else:
        if proc.returncode != 0:
            tail = _stderr_tail(proc.stderr or proc.stdout or "")
            error = f"llama-bench exited with {proc.returncode}" + (f": {tail}" if tail else "")
        else:
            rec = _find_matching_record(proc.stdout, kind)
            if rec is None:
                error = f"no matching {kind} jsonl record in llama-bench output"
            else:
                avg_ts = _to_finite_float(rec.get("avg_ts"))
                stddev_ts = _to_finite_float(rec.get("stddev_ts"))
                if avg_ts is None or avg_ts <= 0.0:
                    error = f"invalid avg_ts in llama-bench record: {rec.get('avg_ts')!r}"
                else:
                    timing_ms = (1000.0 / avg_ts) if kind == "decode" else (1000.0 * n_tokens / avg_ts)

    row: dict[str, object] = {
        "kind": kind,
        "depth": depth,
        "n": n_tokens,
        "avg_ts": avg_ts,
        "stddev_ts": stddev_ts,
    }
    if kind == "decode":
        row["t_t_ms"] = timing_ms
    else:
        row["t_v_ms"] = timing_ms
    row["error"] = error
    return row


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="measure.a1_gpu_bench",
        description="Experiment a1: GPU target decode and verification benchmark driver.",
    )
    parser.add_argument("--out", required=True, help="Output run directory (must not exist).")
    parser.add_argument("--model", required=True, help="Path to target GGUF model file.")
    parser.add_argument(
        "--llama-bench",
        default=str(DEFAULT_LLAMA_BENCH),
        help=f"Path to llama-bench binary (default: {DEFAULT_LLAMA_BENCH}).",
    )
    parser.add_argument(
        "--depths",
        default=DEFAULT_DEPTHS,
        help=f"Comma-separated prompt depths >= 0 (default: {DEFAULT_DEPTHS}).",
    )
    parser.add_argument(
        "--verify-n",
        default=DEFAULT_VERIFY_N,
        help=f"Comma-separated verification batch sizes >= 1 (default: {DEFAULT_VERIFY_N}).",
    )
    parser.add_argument(
        "--reps",
        type=int,
        default=DEFAULT_REPS,
        help=f"Repetitions per llama-bench call >= 1 (default: {DEFAULT_REPS}).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return exc.code if isinstance(exc.code, int) else 2

    out_dir = pathlib.Path(args.out)
    model_path = pathlib.Path(args.model)

    if out_dir.exists() or out_dir.is_symlink():
        print(f"error: --out already exists: {out_dir}", file=sys.stderr)
        return 2
    if not model_path.is_file():
        print(f"error: --model does not exist or is not a file: {model_path}", file=sys.stderr)
        return 2
    if args.reps < 1:
        print(f"error: --reps must be >= 1, got {args.reps}", file=sys.stderr)
        return 2

    try:
        depths = _parse_int_list(args.depths, "--depths", minimum=0)
        verify_n = _parse_int_list(args.verify_n, "--verify-n", minimum=1)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    out_dir.mkdir(parents=True, exist_ok=False)
    metrics_path = out_dir / "metrics.jsonl"

    nvidia_smi_before = _query_nvidia_smi()

    with metrics_path.open("w", encoding="utf-8") as mf:
        for depth in depths:
            row = _run_llama_bench(
                llama_bench=args.llama_bench,
                model_path=model_path,
                kind="decode",
                depth=depth,
                n_tokens=DECODE_TOKENS,
                reps=args.reps,
            )
            mf.write(json.dumps(row) + "\n")
            mf.flush()

        for n in verify_n:
            for depth in depths:
                row = _run_llama_bench(
                    llama_bench=args.llama_bench,
                    model_path=model_path,
                    kind="verify",
                    depth=depth,
                    n_tokens=n,
                    reps=args.reps,
                )
                mf.write(json.dumps(row) + "\n")
                mf.flush()

    nvidia_smi_after = _query_nvidia_smi()

    extra: dict[str, object] = {
        "options": {
            "out": str(out_dir),
            "model": str(model_path),
            "llama_bench": str(args.llama_bench),
            "depths": depths,
            "verify_n": verify_n,
            "reps": args.reps,
        },
        "nvidia_smi_before": nvidia_smi_before,
        "nvidia_smi_after": nvidia_smi_after,
    }
    write_manifest(
        out_dir,
        run_id=out_dir.name or "a1",
        extra=extra,
        files=[model_path],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
