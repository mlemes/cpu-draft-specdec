"""Experiment a4: reference-draft acceptance driver wrapping llama-server."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import http.client
import json
import os
import pathlib
import signal
import subprocess
import sys
import time
from typing import Any

from measure.manifest import write_manifest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_LLAMA_SERVER = (
    REPO_ROOT / "third_party" / "llama.cpp" / "build" / "bin" / "llama-server"
)
DEFAULT_PLAN = "none:0:all,cpu:4:all,gpu:4:all,cpu:2:30,cpu:6:30,gpu:2:30,gpu:6:30"
DEFAULT_N_PREDICT = 64
DEFAULT_PORT = 18080
DEFAULT_DRAFT_THREADS = 6
DEFAULT_SEED = 1234

HEALTH_POLL_INTERVAL_S = 0.2
HEALTH_TIMEOUT_S = 180.0
REQUEST_TIMEOUT_S = 300.0
STOP_TIMEOUT_S = 10.0

VALID_DEVICES = ("none", "cpu", "gpu")
MODES: tuple[tuple[str, dict[str, Any]], ...] = (
    ("greedy", {"temperature": 0, "top_k": 0, "top_p": 1.0, "min_p": 0}),
    ("sampled", {"temperature": 0.7, "top_k": 20, "top_p": 0.8, "min_p": 0}),
)


@dataclass(frozen=True)
class PlanItem:
    device: str
    gamma: int
    subset: int | None
    label: str


@dataclass(frozen=True)
class PromptItem:
    prompt_id: str
    domain: str
    prompt: str
    index: int


def parse_plan(text: str) -> list[PlanItem]:
    raw_items = [part.strip() for part in text.split(",") if part.strip()]
    if not raw_items:
        raise ValueError("--plan must not be empty")
    items: list[PlanItem] = []
    for raw in raw_items:
        parts = raw.split(":")
        if len(parts) != 3:
            raise ValueError(f"invalid plan item {raw!r}: expected device:gamma:subset")
        device, gamma_str, subset_str = parts
        if device not in VALID_DEVICES:
            raise ValueError(
                f"invalid device {device!r} in plan item {raw!r}: expected one of {VALID_DEVICES}"
            )
        try:
            gamma = int(gamma_str)
        except ValueError as exc:
            raise ValueError(f"invalid gamma {gamma_str!r} in plan item {raw!r}") from exc
        if device == "none":
            if gamma < 0:
                raise ValueError(f"gamma must be >= 0 for device 'none', got {gamma}")
        elif gamma < 1:
            raise ValueError(f"gamma must be >= 1 for device {device!r}, got {gamma}")

        if subset_str == "all":
            subset: int | None = None
        else:
            try:
                subset = int(subset_str)
            except ValueError as exc:
                raise ValueError(
                    f"invalid subset {subset_str!r} in plan item {raw!r}: expected 'all' or positive int"
                ) from exc
            if subset < 1:
                raise ValueError(f"subset must be >= 1 in plan item {raw!r}, got {subset}")

        items.append(PlanItem(device=device, gamma=gamma, subset=subset, label=raw))
    return items


def load_prompt_files(paths: list[pathlib.Path]) -> list[list[PromptItem]]:
    all_files: list[list[PromptItem]] = []
    for path in paths:
        if not path.is_file():
            raise ValueError(f"prompt file does not exist or is not a file: {path}")
        file_prompts: list[PromptItem] = []
        with path.open("r", encoding="utf-8") as fh:
            for line_no, line in enumerate(fh, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    obj = json.loads(stripped)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
                if not isinstance(obj, dict):
                    raise ValueError(f"{path}:{line_no}: expected JSON object")
                for field in ("id", "domain", "prompt"):
                    if field not in obj:
                        raise ValueError(f"{path}:{line_no}: missing required field {field!r}")
                prompt_val = obj["prompt"]
                domain_val = obj["domain"]
                if not isinstance(prompt_val, str) or not isinstance(domain_val, str):
                    raise ValueError(f"{path}:{line_no}: 'domain' and 'prompt' must be strings")
                file_prompts.append(
                    PromptItem(
                        prompt_id=str(obj["id"]),
                        domain=domain_val,
                        prompt=prompt_val,
                        index=len(file_prompts),
                    )
                )
        all_files.append(file_prompts)
    return all_files


def build_server_cmd(
    *,
    llama_server: str,
    target: pathlib.Path,
    draft: pathlib.Path,
    port: int,
    draft_threads: int,
    item: PlanItem,
    draft_sampling: str | None = None,
) -> list[str]:
    cmd = [
        llama_server,
        "-m",
        str(target),
        "-ngl",
        "99",
        "-c",
        "2048",  # longest prompt is 573 tokens; 4096 left no VRAM for a GPU draft's KV cache
        "-np",
        "1",
        "--port",
        str(port),
        "--no-webui",
    ]
    if item.device != "none":
        cmd.extend([
            "-md",
            str(draft),
            "--spec-type",
            "draft-simple",  # default spec type is none: without this the draft loads but is never used
            "--spec-draft-n-max",
            str(item.gamma),
            "--spec-draft-n-min",
            str(item.gamma),
            "--spec-draft-p-min",
            "0",
        ])
        if draft_sampling:
            # probabilistic: the draft samples and the target verifies by rejection (min(1, p/q)) at temperature > 0
            cmd.extend(["--spec-draft-sampling", draft_sampling])
        if item.device == "cpu":
            # -tbd too: its default (all cores) oversubscribes the CPU during the draft's per-cycle batch (+1.6 ms/cycle measured)
            cmd.extend(["-ngld", "0", "-devd", "none", "-td", str(draft_threads), "-tbd", str(draft_threads)])
        else:
            cmd.extend(["-ngld", "99"])
    return cmd


def _port_free(port: int) -> bool:
    """True if nothing is listening on 127.0.0.1:port."""
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex(("127.0.0.1", port)) != 0


def _stop_server(proc: subprocess.Popen[bytes] | None) -> None:
    if proc is None:
        return
    # Signal the whole process group even if the leader already exited: children may survive it.
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    except OSError:
        try:
            proc.terminate()
        except OSError:
            pass
    try:
        proc.wait(timeout=STOP_TIMEOUT_S)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    except OSError:
        try:
            proc.kill()
        except OSError:
            pass
    try:
        proc.wait(timeout=5.0)
    except subprocess.TimeoutExpired:
        pass


def _wait_for_health(
    proc: subprocess.Popen[bytes],
    port: int,
    should_stop: Any,
) -> str | None:
    """Poll GET /health until HTTP 200. Returns None on success, or an error string."""
    deadline = time.monotonic() + HEALTH_TIMEOUT_S
    last_err: str | None = None
    while True:
        if should_stop():
            return "interrupted by signal"
        rc = proc.poll()
        if rc is not None:
            return f"llama-server exited prematurely with code {rc}"
        conn: http.client.HTTPConnection | None = None
        try:
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=2.0)
            conn.request("GET", "/health")
            resp = conn.getresponse()
            resp.read()
            if resp.status == 200:
                # Make sure the server we launched is the one answering (not another process on the port).
                if proc.poll() is not None:
                    return f"llama-server exited with code {proc.returncode} while another process answered on port {port}"
                return None
            last_err = f"health returned HTTP {resp.status}"
        except OSError as exc:
            last_err = str(exc)
        except http.client.HTTPException as exc:
            last_err = str(exc)
        finally:
            if conn is not None:
                conn.close()

        if time.monotonic() >= deadline:
            return f"health check timed out after {HEALTH_TIMEOUT_S:.0f}s ({last_err})"
        time.sleep(HEALTH_POLL_INTERVAL_S)


def _post_completion(port: int, payload: dict[str, Any]) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=REQUEST_TIMEOUT_S)
    try:
        conn.request(
            "POST",
            "/completion",
            body=body,
            headers={"Content-Type": "application/json"},
        )
        resp = conn.getresponse()
        raw = resp.read()
        if resp.status != 200:
            snippet = raw.decode("utf-8", errors="replace")[:200]
            raise RuntimeError(f"HTTP {resp.status}: {snippet}")
        parsed = json.loads(raw.decode("utf-8"))
        if not isinstance(parsed, dict):
            raise RuntimeError("completion response is not a JSON object")
        return parsed
    finally:
        conn.close()


def _num_or_none(val: Any) -> int | float | None:
    if isinstance(val, bool) or not isinstance(val, (int, float)):
        return None
    return val


def _make_row(
    *,
    item: PlanItem,
    mode: str | None,
    domain: str | None,
    prompt_id: str | None,
    timings: dict[str, Any] | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    predicted_n = _num_or_none(timings.get("predicted_n")) if timings else None
    predicted_ms = _num_or_none(timings.get("predicted_ms")) if timings else None
    prompt_n = _num_or_none(timings.get("prompt_n")) if timings else None
    prompt_ms = _num_or_none(timings.get("prompt_ms")) if timings else None
    draft_n = _num_or_none(timings.get("draft_n")) if timings else None
    draft_n_accepted = _num_or_none(timings.get("draft_n_accepted")) if timings else None
    if error is None and timings is not None and item.device != "none" and draft_n is None:
        # A server that rejected the draft (e.g. incompatible vocab) still answers, without drafting.
        error = "no draft statistics: speculative decoding inactive"

    tokens_per_s: float | None = None
    if predicted_n is not None and predicted_ms is not None and predicted_ms > 0:
        tokens_per_s = float(predicted_n) / float(predicted_ms) * 1000.0

    acceptance: float | None = None
    if draft_n is not None and draft_n_accepted is not None and draft_n > 0:
        acceptance = float(draft_n_accepted) / float(draft_n)

    return {
        "device": item.device,
        "gamma": item.gamma,
        "mode": mode,
        "domain": domain,
        "prompt_id": prompt_id,
        "predicted_n": predicted_n,
        "predicted_ms": predicted_ms,
        "tokens_per_s": tokens_per_s,
        "prompt_n": prompt_n,
        "prompt_ms": prompt_ms,
        "draft_n": draft_n,
        "draft_n_accepted": draft_n_accepted,
        "acceptance": acceptance,
        "error": error,
    }


def compute_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        mode = row.get("mode")
        domain = row.get("domain")
        if mode is None or domain is None:
            continue
        key = (str(row["device"]), int(row["gamma"]), str(mode), str(domain))
        groups.setdefault(key, []).append(row)

    none_tps: dict[tuple[str, str], float] = {}
    summary_rows: list[dict[str, Any]] = []
    for (device, gamma, mode, domain), group_rows in groups.items():
        tps_vals = [
            float(r["tokens_per_s"])
            for r in group_rows
            if r.get("error") is None and r.get("tokens_per_s") is not None
        ]
        acc_vals = [
            float(r["acceptance"])
            for r in group_rows
            if r.get("error") is None and r.get("acceptance") is not None
        ]
        mean_tps = (sum(tps_vals) / len(tps_vals)) if tps_vals else None
        mean_acc = (sum(acc_vals) / len(acc_vals)) if acc_vals else None
        if device == "none" and mean_tps is not None:
            none_tps[(mode, domain)] = mean_tps
        summary_rows.append(
            {
                "device": device,
                "gamma": gamma,
                "mode": mode,
                "domain": domain,
                "requests": len(group_rows),
                "mean_acceptance": mean_acc,
                "mean_tokens_per_s": mean_tps,
                "speedup": None,
            }
        )

    for entry in summary_rows:
        if entry["device"] == "none":
            mean_tps = entry["mean_tokens_per_s"]
            entry["speedup"] = 1.0 if mean_tps is not None and mean_tps > 0 else None
        else:
            base = none_tps.get((entry["mode"], entry["domain"]))
            mean_tps = entry["mean_tokens_per_s"]
            if base is not None and base > 0 and mean_tps is not None:
                entry["speedup"] = mean_tps / base
            else:
                entry["speedup"] = None
    return summary_rows


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="measure.a4_reference_draft",
        description="Experiment a4: reference-draft acceptance driver.",
    )
    parser.add_argument("--out", required=True, help="Output run directory (must not exist).")
    parser.add_argument("--target", required=True, help="Path to target GGUF model file.")
    parser.add_argument("--draft", required=True, help="Path to draft GGUF model file.")
    parser.add_argument("--prompts", required=True, help="Comma-separated JSONL prompt files.")
    parser.add_argument(
        "--llama-server",
        default=str(DEFAULT_LLAMA_SERVER),
        help=f"Path to llama-server binary (default: {DEFAULT_LLAMA_SERVER}).",
    )
    parser.add_argument(
        "--plan",
        default=DEFAULT_PLAN,
        help=f"Comma-separated device:gamma:subset items (default: {DEFAULT_PLAN}).",
    )
    parser.add_argument(
        "--n-predict",
        type=int,
        default=DEFAULT_N_PREDICT,
        help=f"Tokens to predict per request (default: {DEFAULT_N_PREDICT}).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help=f"TCP port for llama-server (default: {DEFAULT_PORT}).",
    )
    parser.add_argument(
        "--draft-threads",
        type=int,
        default=DEFAULT_DRAFT_THREADS,
        help=f"CPU threads for CPU draft (default: {DEFAULT_DRAFT_THREADS}).",
    )
    parser.add_argument(
        "--spec-draft-sampling",
        choices=("greedy", "probabilistic"),
        default=None,
        help="Pass --spec-draft-sampling to llama-server (default: not passed, server default greedy).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Base RNG seed (default: {DEFAULT_SEED}).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return exc.code if isinstance(exc.code, int) else 2

    out_dir = pathlib.Path(args.out)
    target_path = pathlib.Path(args.target)
    draft_path = pathlib.Path(args.draft)

    if out_dir.exists() or out_dir.is_symlink():
        print(f"error: --out already exists: {out_dir}", file=sys.stderr)
        return 2
    if not target_path.is_file():
        print(f"error: --target does not exist or is not a file: {target_path}", file=sys.stderr)
        return 2
    if not draft_path.is_file():
        print(f"error: --draft does not exist or is not a file: {draft_path}", file=sys.stderr)
        return 2
    if args.n_predict < 1:
        print(f"error: --n-predict must be >= 1, got {args.n_predict}", file=sys.stderr)
        return 2
    if not (1 <= args.port <= 65535):
        print(f"error: --port must be in 1..65535, got {args.port}", file=sys.stderr)
        return 2
    if args.draft_threads < 1:
        print(f"error: --draft-threads must be >= 1, got {args.draft_threads}", file=sys.stderr)
        return 2

    prompt_strs = [p.strip() for p in args.prompts.split(",") if p.strip()]
    if not prompt_strs:
        print("error: --prompts must not be empty", file=sys.stderr)
        return 2
    prompt_paths = [pathlib.Path(p) for p in prompt_strs]

    try:
        plan_items = parse_plan(args.plan)
        prompt_files = load_prompt_files(prompt_paths)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    out_dir.mkdir(parents=True, exist_ok=False)
    metrics_path = out_dir / "metrics.jsonl"
    summary_path = out_dir / "summary.json"

    stop_signum: int | None = None

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

    active_proc: subprocess.Popen[bytes] | None = None
    rows: list[dict[str, Any]] = []

    try:
        with metrics_path.open("w", encoding="utf-8") as mf:
            for item in plan_items:
                if _should_stop():
                    break
                cmd = build_server_cmd(
                    llama_server=args.llama_server,
                    target=target_path,
                    draft=draft_path,
                    port=args.port,
                    draft_threads=args.draft_threads,
                    item=item,
                    draft_sampling=args.spec_draft_sampling,
                )
                log_path = out_dir / f"server-{item.label}.log"
                if not _port_free(args.port):
                    # Another process holds the port: its answers would be measured instead of ours.
                    row = _make_row(item=item, mode=None, domain=None, prompt_id=None,
                                    error=f"port {args.port} already in use")
                    rows.append(row)
                    mf.write(json.dumps(row) + "\n")
                    mf.flush()
                    continue
                try:
                    with log_path.open("wb") as log_f:
                        active_proc = subprocess.Popen(
                            cmd,
                            stdout=log_f,
                            stderr=subprocess.STDOUT,
                            start_new_session=True,
                        )
                except OSError as exc:
                    row = _make_row(
                        item=item,
                        mode=None,
                        domain=None,
                        prompt_id=None,
                        error=f"failed to start llama-server: {exc}",
                    )
                    rows.append(row)
                    mf.write(json.dumps(row) + "\n")
                    mf.flush()
                    continue

                try:
                    health_err = _wait_for_health(active_proc, args.port, _should_stop)
                    if health_err is not None:
                        if not _should_stop():
                            row = _make_row(
                                item=item,
                                mode=None,
                                domain=None,
                                prompt_id=None,
                                error=health_err,
                            )
                            rows.append(row)
                            mf.write(json.dumps(row) + "\n")
                            mf.flush()
                        continue

                    for file_prompts in prompt_files:
                        selected = (
                            file_prompts if item.subset is None else file_prompts[: item.subset]
                        )
                        for p in selected:
                            if _should_stop():
                                break
                            for mode_name, mode_params in MODES:
                                if _should_stop():
                                    break
                                payload: dict[str, Any] = {
                                    "prompt": p.prompt,
                                    "n_predict": args.n_predict,
                                    **mode_params,
                                    "seed": args.seed + p.index,
                                    "cache_prompt": False,
                                }
                                try:
                                    resp = _post_completion(args.port, payload)
                                    timings = resp.get("timings")
                                    if isinstance(timings, dict):
                                        row = _make_row(
                                            item=item,
                                            mode=mode_name,
                                            domain=p.domain,
                                            prompt_id=p.prompt_id,
                                            timings=timings,
                                            error=None,
                                        )
                                    else:
                                        row = _make_row(
                                            item=item,
                                            mode=mode_name,
                                            domain=p.domain,
                                            prompt_id=p.prompt_id,
                                            timings=None,
                                            error="missing timings in completion response",
                                        )
                                except Exception as exc:
                                    row = _make_row(
                                        item=item,
                                        mode=mode_name,
                                        domain=p.domain,
                                        prompt_id=p.prompt_id,
                                        timings=None,
                                        error=str(exc),
                                    )
                                rows.append(row)
                                mf.write(json.dumps(row) + "\n")
                                mf.flush()
                        if _should_stop():
                            break
                finally:
                    _stop_server(active_proc)
                    active_proc = None

        summary = compute_summary(rows)
        summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

        write_manifest(
            out_dir,
            run_id=out_dir.name,
            extra={
                "options": {
                    "out": str(out_dir),
                    "target": str(target_path),
                    "draft": str(draft_path),
                    "prompts": [str(p) for p in prompt_paths],
                    "llama_server": str(args.llama_server),
                    "plan": args.plan,
                    "n_predict": args.n_predict,
                    "port": args.port,
                    "draft_threads": args.draft_threads,
                    "spec_draft_sampling": args.spec_draft_sampling,
                    "seed": args.seed,
                },
                "plan": [
                    {
                        "device": item.device,
                        "gamma": item.gamma,
                        "subset": item.subset if item.subset is not None else "all",
                    }
                    for item in plan_items
                ],
            },
            files=[target_path, draft_path],
        )
    finally:
        _stop_server(active_proc)
        signal.signal(signal.SIGINT, prev_sigint)
        signal.signal(signal.SIGTERM, prev_sigterm)

    if stop_signum is not None:
        return 128 + stop_signum
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
