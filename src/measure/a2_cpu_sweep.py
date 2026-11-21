"""Experiment a2: CPU latency and L3 cache sweep driver (Python stdlib only)."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import signal
import subprocess
import tempfile
import sys
import time
from typing import Any, Callable

from measure.manifest import write_manifest
from measure.resctrl import Resctrl
from measure.touched import config_for_touched_bytes, touched_bytes
from models.random_gguf import generate_random_gguf

DEFAULT_SIZES_MB = "16,32,48,64,96,128,256,512"
DEFAULT_QUANTS = "q8_0,q4_0"
DEFAULT_THREADS = "1,3,6"
DEFAULT_CAT_SIZES_MB = "48,96,192"
DEFAULT_CAT_WAYS = "1,2,4,8,14"
DEFAULT_PHASES = "grid,cat"
DEFAULT_L3_BYTES = 100663296
DEFAULT_L3_SIZE_PATH = pathlib.Path("/sys/devices/system/cpu/cpu0/cache/index3/size")
DEFAULT_LLAMA_BENCH = "third_party/llama.cpp/build/bin/llama-bench"
DEFAULT_MODELS_DIR = "/mnt/hdd1/cpu-draft-specdec/gguf/random"
GROUP_NAME = "a2bench"
RUN_TIMEOUT_S = 600.0
POLL_STEP_S = 0.1
_SRC_DIR = str(pathlib.Path(__file__).resolve().parent.parent)


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


def _parse_str_list(text: str, name: str, allowed: set[str]) -> list[str]:
    parts = [p.strip().lower() for p in text.split(",") if p.strip()]
    if not parts:
        raise ValueError(f"{name} must not be empty")
    for part in parts:
        if part not in allowed:
            raise ValueError(f"invalid value {part!r} in {name}; expected one of {sorted(allowed)}")
    return parts


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


def _read_counter(read: Callable[[], int], timeout_s: float = 3.0) -> int:
    deadline = time.monotonic() + timeout_s
    while True:
        try:
            return read()
        except ValueError:
            if time.monotonic() >= deadline:
                raise ValueError("resctrl counter stayed Unavailable") from None
            time.sleep(0.05)


def _warm_up_group(
    rc: Resctrl,
    should_stop: Callable[[], bool],
    timeout_s: float = 10.0,
) -> bool:
    """Run memory traffic in the group until its MBM counter becomes readable."""
    code = (
        "b = bytearray(16 * 1024 * 1024)\n"
        "while True:\n"
        "    for i in range(0, len(b), 64):\n"
        "        b[i] = 1\n"
    )
    proc = subprocess.Popen(
        [sys.executable, "-c", code],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        rc.assign(GROUP_NAME, proc.pid)
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline and not should_stop() and proc.poll() is None:
            try:
                rc.read_mbm_total(GROUP_NAME)
                return True
            except ValueError:
                time.sleep(0.1)
        return False
    finally:
        if proc.poll() is None:
            proc.kill()
        proc.wait()


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


def _subprocess_env() -> dict[str, str]:
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    if existing:
        parts = existing.split(os.pathsep)
        if _SRC_DIR not in parts:
            env["PYTHONPATH"] = f"{_SRC_DIR}{os.pathsep}{existing}"
    else:
        env["PYTHONPATH"] = _SRC_DIR
    return env


def _model_cache_path(
    models_dir: pathlib.Path, cfg: dict[str, object], quant: str, seed: int
) -> pathlib.Path:
    cfg_json = json.dumps(cfg, sort_keys=True)
    digest = hashlib.sha1(cfg_json.encode("utf-8")).hexdigest()[:8]
    arch = cfg["arch"]
    hidden = cfg["hidden"]
    layers = cfg["layers"]
    return models_dir / f"{arch}-{quant}-d{hidden}-L{layers}-s{seed}-{digest}.gguf"


def _ensure_model(
    models_dir: pathlib.Path,
    cfg: dict[str, object],
    quant: str,
    seed: int,
) -> pathlib.Path:
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = _model_cache_path(models_dir, cfg, quant, seed)
    if not model_path.exists():
        tmp_path = model_path.with_name(f".{model_path.name}.{os.getpid()}.tmp")
        try:
            generate_random_gguf(cfg, quant, seed, tmp_path)
            os.replace(tmp_path, model_path)
        except Exception:
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                pass
            raise
    return model_path


def _parse_llama_bench_jsonl(stdout_text: str) -> tuple[float, float]:
    for line in stdout_text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        try:
            obj = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        n_gen = obj.get("n_gen", 0)
        if isinstance(n_gen, (int, float)) and n_gen > 0:
            if "avg_ts" not in obj or "stddev_ts" not in obj:
                raise ValueError("llama-bench jsonl record missing avg_ts or stddev_ts")
            return float(obj["avg_ts"]), float(obj["stddev_ts"])
    raise ValueError("missing llama-bench jsonl record with n_gen > 0")


def _run_llama_bench(
    *,
    llama_bench: str,
    tasks_path: pathlib.Path,
    model_path: pathlib.Path,
    threads: int,
    n_gen: int,
    reps: int,
    env: dict[str, str],
    active_procs: dict[str, subprocess.Popen[str] | None],
    should_stop: Callable[[], bool],
) -> tuple[float, float] | None:
    cmd = [
        "taskset",
        "-c",
        f"0-{threads - 1}",
        sys.executable,
        "-m",
        "measure.exec_in_group",
        str(tasks_path),
        "--",
        llama_bench,
        "-m",
        str(model_path),
        "-ngl",
        "0",
        "-dev",
        "none",
        "-t",
        str(threads),
        "-p",
        "0",
        "-n",
        str(n_gen),
        "-r",
        str(reps),
        "-o",
        "jsonl",
    ]
    # Temporary files instead of pipes: llama.cpp logs enough on stderr to fill a pipe and block.
    out_f = tempfile.TemporaryFile(mode="w+", encoding="utf-8")
    err_f = tempfile.TemporaryFile(mode="w+", encoding="utf-8")
    proc = subprocess.Popen(
        cmd,
        stdout=out_f,
        stderr=err_f,
        text=True,
        env=env,
    )
    active_procs["bench"] = proc
    try:
        deadline = time.monotonic() + RUN_TIMEOUT_S
        while True:
            if should_stop():
                _kill_and_reap(proc)
                return None
            if proc.poll() is not None:
                break
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                _kill_and_reap(proc)
                raise RuntimeError(f"llama-bench timed out after {RUN_TIMEOUT_S:.0f}s")
            try:
                proc.wait(timeout=min(POLL_STEP_S, remaining))
                break
            except subprocess.TimeoutExpired:
                continue

        proc.wait(timeout=5)
        out_f.seek(0)
        err_f.seek(0)
        stdout_text, stderr_text = out_f.read(), err_f.read()
        if proc.returncode != 0:
            tail = _stderr_tail(stderr_text)
            msg = f"llama-bench exited with code {proc.returncode}"
            if tail:
                msg = f"{msg}: {tail}"
            raise RuntimeError(msg)
        return _parse_llama_bench_jsonl(stdout_text)
    finally:
        _kill_and_reap(proc)
        active_procs["bench"] = None
        out_f.close()
        err_f.close()


def _llama_bench_version(llama_bench: str) -> str | None:
    try:
        res = subprocess.run(
            [llama_bench, "--version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if res.returncode != 0:
        return None
    for line in res.stdout.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return None


def _error_row(
    *,
    phase: str,
    target_mb: int,
    quant: str,
    threads: int,
    ways: int,
    cat_bytes: float,
    model: str | None,
    config: dict[str, object] | None,
    stored_bytes: int | None,
    touched_val: int | float | None,
    message: str,
) -> dict[str, Any]:
    return {
        "phase": phase,
        "target_mb": target_mb,
        "quant": quant,
        "threads": threads,
        "ways": ways,
        "cat_bytes": cat_bytes,
        "model": model,
        "config": config,
        "stored_bytes": stored_bytes,
        "touched_bytes": touched_val,
        "tg_tps": None,
        "tg_tps_sd": None,
        "ms_per_token": None,
        "mbm_bytes_per_token": None,
        "mbm_ratio": None,
        "occupancy_bytes": None,
        "error": message,
    }


def _run_config(
    rc: Resctrl,
    *,
    phase: str,
    target_mb: int,
    quant: str,
    threads: int,
    ways: int,
    cat_bytes: float,
    llama_bench: str,
    models_dir: pathlib.Path,
    tasks_path: pathlib.Path,
    reps: int,
    n_timing: int,
    n_short: int,
    n_long: int,
    seed: int,
    env: dict[str, str],
    active_procs: dict[str, subprocess.Popen[str] | None],
    should_stop: Callable[[], bool],
) -> dict[str, Any] | None:
    cfg: dict[str, object] | None = None
    model_path: pathlib.Path | None = None
    stored_bytes: int | None = None
    touched_val: int | float | None = None

    try:
        cfg = config_for_touched_bytes(target_mb * 1024 * 1024, quant)
        model_path = _ensure_model(models_dir, cfg, quant, seed)
        tb = touched_bytes(model_path)
        stored_bytes = int(tb["stored_bytes"])
        touched_val = tb["touched_bytes_per_token"]

        timing_res = _run_llama_bench(
            llama_bench=llama_bench,
            tasks_path=tasks_path,
            model_path=model_path,
            threads=threads,
            n_gen=n_timing,
            reps=reps,
            env=env,
            active_procs=active_procs,
            should_stop=should_stop,
        )
        if timing_res is None:
            return None
        tg_tps, tg_tps_sd = timing_res
        if tg_tps <= 0:
            raise ValueError(f"invalid avg_ts from llama-bench: {tg_tps}")

        mbm_before_short = _read_counter(lambda: rc.read_mbm_total(GROUP_NAME))
        short_res = _run_llama_bench(
            llama_bench=llama_bench,
            tasks_path=tasks_path,
            model_path=model_path,
            threads=threads,
            n_gen=n_short,
            reps=1,
            env=env,
            active_procs=active_procs,
            should_stop=should_stop,
        )
        if short_res is None:
            return None
        mbm_after_short = _read_counter(lambda: rc.read_mbm_total(GROUP_NAME))
        delta_short = mbm_after_short - mbm_before_short

        mbm_before_long = _read_counter(lambda: rc.read_mbm_total(GROUP_NAME))
        long_res = _run_llama_bench(
            llama_bench=llama_bench,
            tasks_path=tasks_path,
            model_path=model_path,
            threads=threads,
            n_gen=n_long,
            reps=1,
            env=env,
            active_procs=active_procs,
            should_stop=should_stop,
        )
        if long_res is None:
            return None
        mbm_after_long = _read_counter(lambda: rc.read_mbm_total(GROUP_NAME))
        occupancy_bytes = _read_counter(lambda: rc.read_occupancy(GROUP_NAME))
        delta_long = mbm_after_long - mbm_before_long

        mbm_bytes_per_token = (delta_long - delta_short) / (n_long - n_short)
        mbm_ratio = mbm_bytes_per_token / touched_val
        ms_per_token = 1000.0 / tg_tps

        return {
            "phase": phase,
            "target_mb": target_mb,
            "quant": quant,
            "threads": threads,
            "ways": ways,
            "cat_bytes": cat_bytes,
            "model": str(model_path),
            "config": cfg,
            "stored_bytes": stored_bytes,
            "touched_bytes": touched_val,
            "tg_tps": tg_tps,
            "tg_tps_sd": tg_tps_sd,
            "ms_per_token": ms_per_token,
            "mbm_bytes_per_token": mbm_bytes_per_token,
            "mbm_ratio": mbm_ratio,
            "occupancy_bytes": occupancy_bytes,
            "error": None,
        }
    except Exception as exc:
        if should_stop():
            return None
        return _error_row(
            phase=phase,
            target_mb=target_mb,
            quant=quant,
            threads=threads,
            ways=ways,
            cat_bytes=cat_bytes,
            model=str(model_path) if model_path is not None else None,
            config=cfg,
            stored_bytes=stored_bytes,
            touched_val=touched_val,
            message=str(exc),
        )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="measure.a2_cpu_sweep",
        description="Experiment a2: CPU latency and L3 cache sweep.",
    )
    parser.add_argument("--out", required=True, help="Run output directory.")
    parser.add_argument(
        "--resctrl-root",
        default="/sys/fs/resctrl",
        help="Path to resctrl filesystem root.",
    )
    parser.add_argument(
        "--llama-bench",
        default=DEFAULT_LLAMA_BENCH,
        help="Path to llama-bench executable.",
    )
    parser.add_argument(
        "--models-dir",
        default=DEFAULT_MODELS_DIR,
        help="Directory for cached random GGUF models.",
    )
    parser.add_argument("--sizes-mb", default=DEFAULT_SIZES_MB)
    parser.add_argument("--quants", default=DEFAULT_QUANTS)
    parser.add_argument("--threads", default=DEFAULT_THREADS)
    parser.add_argument("--cat-sizes-mb", default=DEFAULT_CAT_SIZES_MB)
    parser.add_argument("--cat-ways", default=DEFAULT_CAT_WAYS)
    parser.add_argument("--cat-threads", type=int, default=6)
    parser.add_argument("--reserve-ways", type=int, default=2)
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--n-timing", type=int, default=128)
    parser.add_argument("--n-short", type=int, default=64)
    parser.add_argument("--n-long", type=int, default=320)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--l3-bytes", type=int, default=None)
    parser.add_argument("--phases", default=DEFAULT_PHASES)
    return parser.parse_args(argv)


def _pid_alive(token: str) -> bool:
    """True if `token` is the pid of a live process (stale entries only occur in fake trees)."""
    try:
        pid = int(token)
    except ValueError:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    try:
        sizes_mb = _parse_int_list(args.sizes_mb, "--sizes-mb", minimum=1)
        quants = _parse_str_list(args.quants, "--quants", {"q8_0", "q4_0"})
        threads_list = _parse_int_list(args.threads, "--threads", minimum=1)
        cat_sizes_mb = _parse_int_list(args.cat_sizes_mb, "--cat-sizes-mb", minimum=1)
        cat_ways = _parse_int_list(args.cat_ways, "--cat-ways", minimum=1)
        phases = _parse_str_list(args.phases, "--phases", {"grid", "cat"})
        if args.cat_threads < 1:
            raise ValueError("--cat-threads must be >= 1")
        if args.reserve_ways < 1:
            raise ValueError("--reserve-ways must be >= 1")
        if args.reps < 1:
            raise ValueError("--reps must be >= 1")
        if args.n_timing < 1:
            raise ValueError("--n-timing must be >= 1")
        if args.n_short < 1:
            raise ValueError("--n-short must be >= 1")
        if args.n_long <= args.n_short:
            raise ValueError("--n-long must be > --n-short")
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
    full_ways = w_total - args.reserve_ways

    try:
        if args.reserve_ways < min_cbm_bits or args.reserve_ways >= w_total:
            raise ValueError(
                f"--reserve-ways ({args.reserve_ways}) must be in "
                f"[{min_cbm_bits}, {w_total - 1}]"
            )
        reserve_mask = rc.ways_mask(args.reserve_ways, w_total - args.reserve_ways)
        if "grid" in phases:
            rc.ways_mask(full_ways, 0)
        if "cat" in phases:
            for w in cat_ways:
                if w < 1 or w > full_ways:
                    raise ValueError(f"--cat-ways value {w} out of bounds [1, {full_ways}]")
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
        if any(_pid_alive(tok) for tok in tasks_text.split()):
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
    active_procs: dict[str, subprocess.Popen[str] | None] = {"bench": None}
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
        if not _should_stop():
            init_ways = full_ways if "grid" in phases else cat_ways[0]
            rc.set_l3_mask(GROUP_NAME, rc.ways_mask(init_ways, 0), cache_id=0)
            if not _warm_up_group(rc, _should_stop):
                print(
                    f"warning: MBM counter for {GROUP_NAME} still Unavailable after warm-up",
                    file=sys.stderr,
                )

        env = _subprocess_env()
        models_dir = pathlib.Path(args.models_dir)
        group_tasks_path = resctrl_root / GROUP_NAME / "tasks"

        for phase in phases:
            if _should_stop():
                break
            if phase == "grid":
                rc.set_l3_mask(GROUP_NAME, rc.ways_mask(full_ways, 0), cache_id=0)
                cat_bytes = full_ways * l3_bytes / w_total
                for target_mb in sizes_mb:
                    if _should_stop():
                        break
                    for quant in quants:
                        if _should_stop():
                            break
                        for threads in threads_list:
                            if _should_stop():
                                break
                            row = _run_config(
                                rc,
                                phase="grid",
                                target_mb=target_mb,
                                quant=quant,
                                threads=threads,
                                ways=full_ways,
                                cat_bytes=cat_bytes,
                                llama_bench=args.llama_bench,
                                models_dir=models_dir,
                                tasks_path=group_tasks_path,
                                reps=args.reps,
                                n_timing=args.n_timing,
                                n_short=args.n_short,
                                n_long=args.n_long,
                                seed=args.seed,
                                env=env,
                                active_procs=active_procs,
                                should_stop=_should_stop,
                            )
                            if row is None or _should_stop():
                                break
                            metrics_file.write(json.dumps(row) + "\n")
                            metrics_file.flush()
            elif phase == "cat":
                for target_mb in cat_sizes_mb:
                    if _should_stop():
                        break
                    for quant in quants:
                        if _should_stop():
                            break
                        for ways in cat_ways:
                            if _should_stop():
                                break
                            rc.set_l3_mask(GROUP_NAME, rc.ways_mask(ways, 0), cache_id=0)
                            cat_bytes = ways * l3_bytes / w_total
                            row = _run_config(
                                rc,
                                phase="cat",
                                target_mb=target_mb,
                                quant=quant,
                                threads=args.cat_threads,
                                ways=ways,
                                cat_bytes=cat_bytes,
                                llama_bench=args.llama_bench,
                                models_dir=models_dir,
                                tasks_path=group_tasks_path,
                                reps=args.reps,
                                n_timing=args.n_timing,
                                n_short=args.n_short,
                                n_long=args.n_long,
                                seed=args.seed,
                                env=env,
                                active_procs=active_procs,
                                should_stop=_should_stop,
                            )
                            if row is None or _should_stop():
                                break
                            metrics_file.write(json.dumps(row) + "\n")
                            metrics_file.flush()

        if not _should_stop():
            metrics_file.close()
            extra: dict[str, object] = {
                "options": {
                    "out": str(out_dir),
                    "resctrl_root": str(resctrl_root),
                    "llama_bench": str(args.llama_bench),
                    "models_dir": str(models_dir),
                    "sizes_mb": sizes_mb,
                    "quants": quants,
                    "threads": threads_list,
                    "cat_sizes_mb": cat_sizes_mb,
                    "cat_ways": cat_ways,
                    "cat_threads": args.cat_threads,
                    "reserve_ways": args.reserve_ways,
                    "reps": args.reps,
                    "n_timing": args.n_timing,
                    "n_short": args.n_short,
                    "n_long": args.n_long,
                    "seed": args.seed,
                    "l3_bytes": l3_bytes,
                    "phases": phases,
                },
                "info": info,
                "W": w_total,
                "l3_bytes": l3_bytes,
                "llama_bench": str(args.llama_bench),
                "llama_bench_version": _llama_bench_version(args.llama_bench),
            }
            write_manifest(
                out_dir,
                run_id=out_dir.name or "a2",
                extra=extra,
                files=(metrics_path,),
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
