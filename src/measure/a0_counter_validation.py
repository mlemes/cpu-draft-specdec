"""Experiment a0: resctrl CAT / MBM / llc_occupancy counter-validation driver.

Note: perf attaches with a 0.2 s settle delay, so its counts may miss the
first instants of the timed passes; MBM is the primary counter.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import queue
import signal
import subprocess
import sys
import threading
import time
from typing import Any, Callable

from measure.manifest import write_manifest
from measure.resctrl import Resctrl

DEFAULT_SIZES_MB = "4,8,16,32,48,64,96,128,192,384"
DEFAULT_WAYS = "1,2,4,8,14"
DEFAULT_L3_BYTES = 100663296
DEFAULT_L3_SIZE_PATH = pathlib.Path("/sys/devices/system/cpu/cpu0/cache/index3/size")
GROUP_NAME = "a0bench"
READY_TIMEOUT_S = 30.0
RUN_TIMEOUT_S = 60.0
POLL_STEP_S = 0.1


def _default_l3_bytes(size_path: pathlib.Path = DEFAULT_L3_SIZE_PATH) -> int:
    try:
        raw = size_path.read_text(encoding="utf-8").strip().upper()
    except OSError:
        return DEFAULT_L3_BYTES
    if not raw:
        return DEFAULT_L3_BYTES
    multipliers = {"K": 1024, "M": 1024**2, "G": 1024**3}
    mult = 1
    if raw[-1] in multipliers:
        mult = multipliers[raw[-1]]
        raw = raw[:-1]
    try:
        val = int(raw) * mult
    except ValueError:
        return DEFAULT_L3_BYTES
    return val if val > 0 else DEFAULT_L3_BYTES


def _parse_int_list(text: str, name: str, *, minimum: int = 1) -> list[int]:
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


def _read_root_l3_line(resctrl_root: pathlib.Path) -> str | None:
    schemata_path = resctrl_root / "schemata"
    if not schemata_path.exists():
        return None
    for line in schemata_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("L3:"):
            return stripped
    return None


def _restore_root_l3_line(resctrl_root: pathlib.Path, l3_line: str | None) -> None:
    if l3_line is None:
        return
    (resctrl_root / "schemata").write_text(l3_line + "\n", encoding="utf-8")


def _parse_perf_csv(stderr_text: str) -> dict[str, int | float | None]:
    counts: dict[str, int | float | None] = {}
    for line in stderr_text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        fields = stripped.split(",")
        if len(fields) < 3:
            continue
        val_str = fields[0].strip()
        event_name = fields[2].strip()
        if not event_name or not val_str:
            continue
        if val_str in ("<not supported>", "<not counted>"):
            counts[event_name] = None
            continue
        if val_str.startswith("<"):
            continue
        try:
            counts[event_name] = int(val_str)
        except ValueError:
            try:
                counts[event_name] = float(val_str)
            except ValueError:
                continue
    return counts


def _stderr_tail(text: str, max_lines: int = 5, max_chars: int = 400) -> str:
    lines = [ln for ln in text.strip().splitlines() if ln.strip()]
    tail = "\n".join(lines[-max_lines:])
    if len(tail) > max_chars:
        tail = tail[-max_chars:]
    return tail


def _kill_and_reap(proc: subprocess.Popen[str] | None) -> None:
    if proc is None:
        return
    if proc.poll() is None:
        try:
            proc.kill()
        except OSError:
            pass
    try:
        proc.wait(timeout=5)
    except (OSError, subprocess.TimeoutExpired):
        pass


def _start_stdout_reader(
    pipe: Any,
) -> tuple[queue.Queue[str | None], threading.Thread]:
    q: queue.Queue[str | None] = queue.Queue()

    def _reader() -> None:
        try:
            for line in iter(pipe.readline, ""):
                q.put(line)
        finally:
            q.put(None)

    t = threading.Thread(target=_reader, daemon=True)
    t.start()
    return q, t


def _error_row(
    *,
    ways: int,
    cat_bytes: float,
    size_bytes: int,
    passes: int,
    message: str,
) -> dict[str, Any]:
    return {
        "ways": ways,
        "cat_bytes": cat_bytes,
        "size_bytes": size_bytes,
        "passes": passes,
        "median_ns": None,
        "gbps": None,
        "mbm_bytes_per_pass": None,
        "mbm_ratio": None,
        "occupancy_bytes": None,
        "perf": {},
        "error": message,
    }


def _run_single(
    rc: Resctrl,
    *,
    bench: str,
    ways: int,
    cat_bytes: float,
    size_bytes: int,
    passes: int,
    cpu: int,
    perf_path: str | None,
    perf_events: str | None,
    active_procs: dict[str, subprocess.Popen[str] | None],
    should_stop: Callable[[], bool],
) -> dict[str, Any] | None:
    cmd = [
        bench,
        "--bytes",
        str(size_bytes),
        "--passes",
        str(passes),
        "--cpu",
        str(cpu),
        "--wait-signal",
    ]
    bench_proc: subprocess.Popen[str] | None = None
    bench_start_err: OSError | None = None
    try:
        bench_proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            text=True,
        )
        active_procs["bench"] = bench_proc
    except OSError as exc:
        bench_start_err = exc

    if bench_start_err is not None or bench_proc is None:
        return _error_row(
            ways=ways,
            cat_bytes=cat_bytes,
            size_bytes=size_bytes,
            passes=passes,
            message=f"failed to start bench: {bench_start_err}",
        )

    perf_proc: subprocess.Popen[str] | None = None
    try:
        assert bench_proc.stdout is not None
        line_queue, reader_thread = _start_stdout_reader(bench_proc.stdout)

        ready_deadline = time.monotonic() + READY_TIMEOUT_S
        ready_line: str | None = None
        got_ready = False
        while True:
            if should_stop():
                _kill_and_reap(bench_proc)
                return None
            remaining = ready_deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                ready_line = line_queue.get(timeout=min(POLL_STEP_S, remaining))
                got_ready = True
                break
            except queue.Empty:
                continue

        if not got_ready:
            _kill_and_reap(bench_proc)
            return _error_row(
                ways=ways,
                cat_bytes=cat_bytes,
                size_bytes=size_bytes,
                passes=passes,
                message="timeout waiting for ready line",
            )

        if ready_line is None:
            _kill_and_reap(bench_proc)
            return _error_row(
                ways=ways,
                cat_bytes=cat_bytes,
                size_bytes=size_bytes,
                passes=passes,
                message=f"bench exited before ready (code {bench_proc.returncode})",
            )

        try:
            ready_obj = json.loads(ready_line)
            if not isinstance(ready_obj, dict) or "ready" not in ready_obj:
                raise ValueError("missing 'ready' key")
            bench_pid = int(ready_obj["ready"])
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            _kill_and_reap(bench_proc)
            return _error_row(
                ways=ways,
                cat_bytes=cat_bytes,
                size_bytes=size_bytes,
                passes=passes,
                message=f"invalid ready line: {exc}",
            )

        rc.assign(GROUP_NAME, bench_pid)
        mbm_before = rc.read_mbm_total(GROUP_NAME)

        if perf_path and perf_events:
            perf_cmd = [
                perf_path,
                "stat",
                "-x,",
                "-e",
                perf_events,
                "-p",
                str(bench_pid),
            ]
            perf_start_err: OSError | None = None
            try:
                perf_proc = subprocess.Popen(
                    perf_cmd,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                active_procs["perf"] = perf_proc
            except OSError as exc:
                perf_start_err = exc

            if perf_start_err is not None or perf_proc is None:
                _kill_and_reap(bench_proc)
                return _error_row(
                    ways=ways,
                    cat_bytes=cat_bytes,
                    size_bytes=size_bytes,
                    passes=passes,
                    message=f"failed to start perf: {perf_start_err}",
                )

            settle_deadline = time.monotonic() + 0.2
            while True:
                if should_stop():
                    _kill_and_reap(bench_proc)
                    _kill_and_reap(perf_proc)
                    return None
                remaining = settle_deadline - time.monotonic()
                if remaining <= 0:
                    break
                time.sleep(min(POLL_STEP_S, remaining))

            if perf_proc.poll() is not None:
                try:
                    _, early_stderr = perf_proc.communicate(timeout=5)
                except subprocess.TimeoutExpired:
                    _kill_and_reap(perf_proc)
                    early_stderr = ""
                active_procs["perf"] = None
                _kill_and_reap(bench_proc)
                tail = _stderr_tail(early_stderr)
                msg = f"perf exited early (code {perf_proc.returncode})"
                if tail:
                    msg = f"{msg}: {tail}"
                return _error_row(
                    ways=ways,
                    cat_bytes=cat_bytes,
                    size_bytes=size_bytes,
                    passes=passes,
                    message=msg,
                )

        try:
            os.kill(bench_pid, signal.SIGUSR1)
        except OSError as exc:
            _kill_and_reap(bench_proc)
            _kill_and_reap(perf_proc)
            return _error_row(
                ways=ways,
                cat_bytes=cat_bytes,
                size_bytes=size_bytes,
                passes=passes,
                message=f"failed to signal bench: {exc}",
            )

        run_deadline = time.monotonic() + RUN_TIMEOUT_S
        rc_code: int | None = None
        while True:
            if should_stop():
                _kill_and_reap(bench_proc)
                _kill_and_reap(perf_proc)
                return None
            rc_code = bench_proc.poll()
            if rc_code is not None:
                break
            remaining = run_deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                rc_code = bench_proc.wait(timeout=min(POLL_STEP_S, remaining))
                break
            except subprocess.TimeoutExpired:
                continue

        if rc_code is None:
            _kill_and_reap(bench_proc)
            _kill_and_reap(perf_proc)
            return _error_row(
                ways=ways,
                cat_bytes=cat_bytes,
                size_bytes=size_bytes,
                passes=passes,
                message=f"bench timed out after {RUN_TIMEOUT_S:.0f}s",
            )

        perf_counts: dict[str, int | float | None] = {}
        perf_error: str | None = None
        if perf_proc is not None:
            if perf_proc.poll() is None:
                try:
                    perf_proc.send_signal(signal.SIGINT)
                except OSError:
                    pass
            perf_deadline = time.monotonic() + 10.0
            perf_exited = False
            while True:
                if should_stop():
                    _kill_and_reap(bench_proc)
                    _kill_and_reap(perf_proc)
                    return None
                if perf_proc.poll() is not None:
                    perf_exited = True
                    break
                remaining = perf_deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    perf_proc.wait(timeout=min(POLL_STEP_S, remaining))
                    perf_exited = True
                    break
                except subprocess.TimeoutExpired:
                    continue
            if perf_exited:
                try:
                    _, perf_stderr = perf_proc.communicate(timeout=5)
                except subprocess.TimeoutExpired:
                    _kill_and_reap(perf_proc)
                    perf_stderr = ""
            else:
                _kill_and_reap(perf_proc)
                perf_stderr = ""
            perf_counts = _parse_perf_csv(perf_stderr)
            active_procs["perf"] = None

            rc_perf = perf_proc.returncode
            if rc_perf not in (0, -signal.SIGINT, 128 + signal.SIGINT):
                tail = _stderr_tail(perf_stderr)
                perf_error = f"perf exited with code {rc_perf}"
                if tail:
                    perf_error = f"{perf_error}: {tail}"
            elif not any(v is not None for v in perf_counts.values()):
                perf_error = "perf produced no parsed counts"

        mbm_after = rc.read_mbm_total(GROUP_NAME)
        occupancy_bytes = rc.read_occupancy(GROUP_NAME)

        reader_thread.join(timeout=5)
        stdout_lines: list[str] = []
        while True:
            try:
                item = line_queue.get_nowait()
            except queue.Empty:
                break
            if item is None:
                break
            stdout_lines.append(item)

        if rc_code != 0:
            return _error_row(
                ways=ways,
                cat_bytes=cat_bytes,
                size_bytes=size_bytes,
                passes=passes,
                message=f"bench exited with code {rc_code}",
            )

        median_ns: int | None = None
        for line in stdout_lines:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                obj = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict) and obj.get("summary") is True:
                if "median_ns" in obj:
                    median_ns = int(obj["median_ns"])

        if median_ns is None or median_ns <= 0:
            return _error_row(
                ways=ways,
                cat_bytes=cat_bytes,
                size_bytes=size_bytes,
                passes=passes,
                message="missing or invalid summary line from bench",
            )

        mbm_bytes_per_pass = (mbm_after - mbm_before) / passes
        mbm_ratio = mbm_bytes_per_pass / size_bytes
        gbps = size_bytes / median_ns

        return {
            "ways": ways,
            "cat_bytes": cat_bytes,
            "size_bytes": size_bytes,
            "passes": passes,
            "median_ns": median_ns,
            "gbps": gbps,
            "mbm_bytes_per_pass": mbm_bytes_per_pass,
            "mbm_ratio": mbm_ratio,
            "occupancy_bytes": occupancy_bytes,
            "perf": perf_counts,
            "error": perf_error,
        }
    finally:
        _kill_and_reap(bench_proc)
        _kill_and_reap(perf_proc)
        if bench_proc.stdout is not None:
            try:
                bench_proc.stdout.close()
            except OSError:
                pass
        if perf_proc is not None and perf_proc.stderr is not None:
            try:
                perf_proc.stderr.close()
            except OSError:
                pass
        active_procs["bench"] = None
        active_procs["perf"] = None


def _build_summary(
    rows: list[dict[str, Any]],
    ways_list: list[int],
    l3_bytes: int,
    w_total: int,
) -> dict[str, Any]:
    per_ways: dict[str, Any] = {}
    all_passed = True

    for w in ways_list:
        cat_bytes = w * l3_bytes / w_total
        w_rows = [r for r in rows if r["ways"] == w]
        above = [r for r in w_rows if r["size_bytes"] >= 2 * cat_bytes]
        below = [r for r in w_rows if r["size_bytes"] <= 0.5 * cat_bytes]

        above_ok = bool(above) and all(
            r["error"] is None
            and r["mbm_ratio"] is not None
            and 0.9 <= r["mbm_ratio"] <= 1.1
            for r in above
        )
        below_ok = bool(below) and all(
            r["error"] is None
            and r["mbm_ratio"] is not None
            and r["mbm_ratio"] <= 0.1
            for r in below
        )
        pass_w = above_ok and below_ok
        if not pass_w:
            all_passed = False

        per_ways[str(w)] = {
            "cat_bytes": cat_bytes,
            "total_count": len(w_rows),
            "above_count": len(above),
            "below_count": len(below),
            "pass_w": pass_w,
        }

    return {
        "passed": all_passed,
        "ways": per_ways,
    }


def _chown_for_sudo(out_dir: pathlib.Path) -> None:
    uid_str = os.environ.get("SUDO_UID")
    gid_str = os.environ.get("SUDO_GID")
    if uid_str is None or gid_str is None:
        return
    try:
        uid = int(uid_str)
        gid = int(gid_str)
    except ValueError:
        return
    os.chown(out_dir, uid, gid, follow_symlinks=False)
    for child in out_dir.iterdir():
        os.chown(child, uid, gid, follow_symlinks=False)


def _open_exclusive_write(path: pathlib.Path):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o644)
    return os.fdopen(fd, "w", encoding="utf-8")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="measure.a0_counter_validation",
        description="Experiment a0: resctrl CAT/MBM/occupancy counter validation.",
    )
    parser.add_argument("--out", required=True, help="Run output directory.")
    parser.add_argument(
        "--resctrl-root",
        default="/sys/fs/resctrl",
        help="Path to resctrl filesystem root.",
    )
    parser.add_argument(
        "--bench",
        default="src/measure/stream_bench",
        help="Path to stream_bench executable.",
    )
    parser.add_argument(
        "--sizes-mb",
        default=DEFAULT_SIZES_MB,
        help="Comma-separated buffer sizes in MiB.",
    )
    parser.add_argument(
        "--ways",
        default=DEFAULT_WAYS,
        help="Comma-separated L3 way counts.",
    )
    parser.add_argument(
        "--reserve-ways",
        type=int,
        default=2,
        help="Highest N ways reserved for the root group.",
    )
    parser.add_argument("--passes", type=int, default=5, help="Timed passes per run.")
    parser.add_argument("--cpu", type=int, default=2, help="CPU core to pin benchmark.")
    parser.add_argument(
        "--l3-bytes",
        type=int,
        default=None,
        help="Total L3 size in bytes (default: read from sysfs or 100663296).",
    )
    parser.add_argument("--perf", default=None, help="Path to perf binary.")
    parser.add_argument(
        "--perf-events",
        default=None,
        help="Comma-separated perf events for perf stat.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    try:
        sizes_mb = _parse_int_list(args.sizes_mb, "--sizes-mb", minimum=1)
        ways_list = _parse_int_list(args.ways, "--ways", minimum=1)
        if args.passes < 1:
            raise ValueError("--passes must be >= 1")
        if args.cpu < 0:
            raise ValueError("--cpu must be >= 0")
        if args.reserve_ways < 1:
            raise ValueError("--reserve-ways must be >= 1")
        l3_bytes = args.l3_bytes if args.l3_bytes is not None else _default_l3_bytes()
        if l3_bytes <= 0:
            raise ValueError("--l3-bytes must be > 0")
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    resctrl_root = pathlib.Path(args.resctrl_root)
    rc = Resctrl(resctrl_root)

    try:
        info = rc.info()
    except PermissionError:
        print("needs root: run with sudo", file=sys.stderr)
        return 1
    except (OSError, ValueError) as exc:
        print(f"error: cannot read resctrl info: {exc}", file=sys.stderr)
        return 2

    cbm_mask = int(info["cbm_mask"])
    min_cbm_bits = int(info["min_cbm_bits"])
    w_total = cbm_mask.bit_count()

    try:
        if args.reserve_ways < min_cbm_bits or args.reserve_ways >= w_total:
            raise ValueError(
                f"--reserve-ways ({args.reserve_ways}) must be in "
                f"[{min_cbm_bits}, {w_total - 1}]"
            )
        reserve_mask = rc.ways_mask(args.reserve_ways, w_total - args.reserve_ways)
        for w in ways_list:
            if w < 1 or w > w_total - args.reserve_ways:
                raise ValueError(
                    f"ways value {w} out of bounds [1, {w_total - args.reserve_ways}]"
                )
            rc.ways_mask(w, 0)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    group_dir = resctrl_root / GROUP_NAME
    if group_dir.exists():
        tasks_path = group_dir / "tasks"
        try:
            tasks_text = tasks_path.read_text(encoding="utf-8")
        except PermissionError:
            print("needs root: run with sudo", file=sys.stderr)
            return 1
        except FileNotFoundError:
            tasks_text = ""
        if tasks_text.strip():
            print(f"group {GROUP_NAME} in use", file=sys.stderr)
            return 1

    out_dir = pathlib.Path(args.out)
    try:
        os.lstat(out_dir)
    except FileNotFoundError:
        pass
    else:
        print(f"error: --out already exists: {out_dir}", file=sys.stderr)
        return 2

    saved_l3_line: str | None = None
    active_procs: dict[str, subprocess.Popen[str] | None] = {
        "bench": None,
        "perf": None,
    }
    rows: list[dict[str, Any]] = []
    exit_code = 0
    restore_failed = False
    out_created = False
    metrics_file: Any = None
    stop_signum: int | None = None
    root_modified = False

    def _signal_handler(signum: int, _frame: object) -> None:
        nonlocal stop_signum
        if stop_signum is None:
            stop_signum = signum

    def _should_stop() -> bool:
        return stop_signum is not None

    prev_sigint = signal.getsignal(signal.SIGINT)
    prev_sigterm = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    try:
        os.mkdir(out_dir, 0o755)
        out_created = True
        metrics_path = out_dir / "metrics.jsonl"
        metrics_file = _open_exclusive_write(metrics_path)

        saved_l3_line = _read_root_l3_line(resctrl_root)
        rc.set_l3_mask("", reserve_mask, cache_id=0)
        root_modified = True
        rc.create_group(GROUP_NAME)

        for w in ways_list:
            if _should_stop():
                break
            mask = rc.ways_mask(w, 0)
            rc.set_l3_mask(GROUP_NAME, mask, cache_id=0)
            cat_bytes = w * l3_bytes / w_total
            for s_mb in sizes_mb:
                if _should_stop():
                    break
                size_bytes = s_mb * 1024 * 1024
                row = _run_single(
                    rc,
                    bench=args.bench,
                    ways=w,
                    cat_bytes=cat_bytes,
                    size_bytes=size_bytes,
                    passes=args.passes,
                    cpu=args.cpu,
                    perf_path=args.perf,
                    perf_events=args.perf_events,
                    active_procs=active_procs,
                    should_stop=_should_stop,
                )
                if row is None or _should_stop():
                    break
                rows.append(row)
                metrics_file.write(json.dumps(row) + "\n")
                metrics_file.flush()

        if not _should_stop():
            metrics_file.close()

            summary = _build_summary(rows, ways_list, l3_bytes, w_total)
            summary_path = out_dir / "summary.json"
            with _open_exclusive_write(summary_path) as sf:
                sf.write(json.dumps(summary, indent=2) + "\n")

            extra: dict[str, object] = {
                "options": {
                    "out": str(out_dir),
                    "resctrl_root": str(resctrl_root),
                    "bench": str(args.bench),
                    "sizes_mb": sizes_mb,
                    "ways": ways_list,
                    "reserve_ways": args.reserve_ways,
                    "passes": args.passes,
                    "cpu": args.cpu,
                    "l3_bytes": l3_bytes,
                    "perf": args.perf,
                    "perf_events": args.perf_events,
                },
                "info": info,
                "W": w_total,
                "l3_bytes": l3_bytes,
            }
            write_manifest(
                out_dir,
                run_id=out_dir.name or "a0",
                extra=extra,
                files=(metrics_path, summary_path),
            )
    except PermissionError:
        print("needs root: run with sudo", file=sys.stderr)
        exit_code = 1
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        exit_code = 1
    finally:
        # Handlers stay installed (flag-only) until cleanup ends, so a signal here cannot skip the restore.
        if metrics_file is not None and not metrics_file.closed:
            try:
                metrics_file.close()
            except OSError:
                pass
        _kill_and_reap(active_procs.get("bench"))
        _kill_and_reap(active_procs.get("perf"))
        try:
            if root_modified:
                _restore_root_l3_line(resctrl_root, saved_l3_line)
        except OSError as exc:
            print(
                f"ERROR: failed to restore root L3 schemata to {saved_l3_line}: {exc}",
                file=sys.stderr,
            )
            restore_failed = True
        try:
            rc.remove_group(GROUP_NAME)
        except OSError:
            pass
        if out_created:
            try:
                _chown_for_sudo(out_dir)
            except OSError:
                pass
        signal.signal(signal.SIGINT, prev_sigint)
        signal.signal(signal.SIGTERM, prev_sigterm)

    if restore_failed:
        return 3
    if exit_code != 0:
        return exit_code
    if stop_signum is not None:
        return 128 + stop_signum
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
