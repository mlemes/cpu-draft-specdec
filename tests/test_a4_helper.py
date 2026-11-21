"""Unit tests for src/measure/a4_reference_draft.py."""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import signal
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

from measure import a4_reference_draft as a4


FAKE_LLAMA_SERVER = """#!/usr/bin/env python3
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
import signal
import sys
import time

parser = argparse.ArgumentParser()
parser.add_argument("-m", required=True)
parser.add_argument("-ngl", required=True)
parser.add_argument("-c", required=True)
parser.add_argument("-np", required=True)
parser.add_argument("--port", type=int, required=True)
parser.add_argument("--no-webui", action="store_true")
parser.add_argument("-md", default=None)
parser.add_argument("--spec-type", default=None)
parser.add_argument("--spec-draft-n-max", type=int, default=0)
parser.add_argument("--spec-draft-n-min", type=int, default=0)
parser.add_argument("--spec-draft-p-min", default=None)
parser.add_argument("-ngld", default=None)
parser.add_argument("-devd", default=None)
parser.add_argument("-td", default=None)
parser.add_argument("-tbd", default=None)
args = parser.parse_args()

log_file = os.environ.get("FAKE_SERVER_LOG")
if log_file:
    with open(log_file, "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"pid": os.getpid(), "argv": sys.argv[1:]}) + "\\n")

pid_file = os.environ.get("FAKE_SERVER_PID_FILE")
if pid_file:
    with open(pid_file, "a", encoding="utf-8") as fh:
        fh.write(f"{os.getpid()}\\n")

mode = os.environ.get("FAKE_SERVER_MODE", "ok")
if mode == "ignore_sigterm":
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
if mode == "exit_early":
    sys.stderr.write("fatal: simulated startup failure\\n")
    sys.exit(1)

req_log = os.environ.get("FAKE_SERVER_REQ_LOG")


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *a):
        sys.stderr.write((fmt % a) + "\\n")

    def do_GET(self):
        if self.path == "/health":
            if mode == "unhealthy":
                self.send_response(503)
                self.end_headers()
                self.wfile.write(b'{"status":"loading"}')
                return
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"status":"ok"}')
            return
        self.send_response(404)
        self.end_headers()

    def do_POST(self):
        if self.path != "/completion":
            self.send_response(404)
            self.end_headers()
            return
        length = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(length).decode("utf-8"))
        if req_log:
            with open(req_log, "a", encoding="utf-8") as fh:
                fh.write(json.dumps({
                    "md": args.md,
                    "ngld": args.ngld,
                    "gamma": args.spec_draft_n_max,
                    "body": body,
                }) + "\\n")

        if args.md is None:
            predicted_ms = 1000.0
            timings = {
                "predicted_n": 20,
                "predicted_ms": predicted_ms,
                "prompt_n": 10,
                "prompt_ms": 50.0,
            }
        elif args.ngld == "0":
            predicted_ms = 500.0
            timings = {
                "predicted_n": 20,
                "predicted_ms": predicted_ms,
                "prompt_n": 10,
                "prompt_ms": 50.0,
                "draft_n": 16,
                "draft_n_accepted": 12,
            }
        else:
            predicted_ms = 400.0
            timings = {
                "predicted_n": 20,
                "predicted_ms": predicted_ms,
                "prompt_n": 10,
                "prompt_ms": 50.0,
                "draft_n": 16,
                "draft_n_accepted": 8,
            }

        resp = json.dumps({"content": "ok", "timings": timings}).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(resp)))
        self.end_headers()
        self.wfile.write(resp)


ThreadingHTTPServer.allow_reuse_address = True
srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
srv.serve_forever()
"""


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


class TestA4ReferenceDraft(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self.tmp.name)

        self.server_bin = self.root / "llama-server"
        self.server_bin.write_text(FAKE_LLAMA_SERVER, encoding="utf-8")
        self.server_bin.chmod(0o755)

        self.target = self.root / "target.gguf"
        self.target.write_bytes(b"target-model-bytes")
        self.target_sha = hashlib.sha256(b"target-model-bytes").hexdigest()

        self.draft = self.root / "draft.gguf"
        self.draft.write_bytes(b"draft-model-bytes")
        self.draft_sha = hashlib.sha256(b"draft-model-bytes").hexdigest()

        self.prompts_code = self.root / "code.jsonl"
        self.prompts_code.write_text(
            "\n".join([
                json.dumps({"id": "c0", "domain": "code", "prompt": "def f():"}),
                json.dumps({"id": "c1", "domain": "code", "prompt": "def g():"}),
            ]) + "\n",
            encoding="utf-8",
        )
        self.prompts_chat = self.root / "chat.jsonl"
        self.prompts_chat.write_text(
            "\n".join([
                json.dumps({"id": "h0", "domain": "chat", "prompt": "Hello!"}),
                json.dumps({"id": "h1", "domain": "chat", "prompt": "How are you?"}),
            ]) + "\n",
            encoding="utf-8",
        )

        self.server_log = self.root / "server_calls.jsonl"
        self.req_log = self.root / "req_log.jsonl"
        self.pid_file = self.root / "server_pids.txt"
        self.out_dir = self.root / "run_a4"

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _env(self, extra: dict[str, str] | None = None) -> dict[str, str]:
        env = {
            "FAKE_SERVER_LOG": str(self.server_log),
            "FAKE_SERVER_REQ_LOG": str(self.req_log),
            "FAKE_SERVER_PID_FILE": str(self.pid_file),
        }
        if extra:
            env.update(extra)
        return env

    def test_plan_parsing(self) -> None:
        items = a4.parse_plan(a4.DEFAULT_PLAN)
        self.assertEqual(len(items), 7)
        self.assertEqual(items[0], a4.PlanItem("none", 0, None, "none:0:all"))
        self.assertEqual(items[1], a4.PlanItem("cpu", 4, None, "cpu:4:all"))
        self.assertEqual(items[3], a4.PlanItem("cpu", 2, 30, "cpu:2:30"))
        self.assertEqual(items[6], a4.PlanItem("gpu", 6, 30, "gpu:6:30"))

        for bad in ("", "none:0", "tpu:4:all", "cpu:0:all", "none:-1:all", "cpu:4:0", "cpu:4:foo"):
            with self.assertRaises(ValueError, msg=bad):
                a4.parse_plan(bad)

    def test_existing_out_returns_2(self) -> None:
        self.out_dir.mkdir()
        rc = a4.main([
            "--out", str(self.out_dir),
            "--target", str(self.target),
            "--draft", str(self.draft),
            "--prompts", str(self.prompts_code),
            "--llama-server", str(self.server_bin),
        ])
        self.assertEqual(rc, 2)

    def test_end_to_end_metrics_summary_manifest_and_shutdown(self) -> None:
        port = _free_port()
        with mock.patch.dict(os.environ, self._env()):
            rc = a4.main([
                "--out", str(self.out_dir),
                "--target", str(self.target),
                "--draft", str(self.draft),
                "--prompts", f"{self.prompts_code},{self.prompts_chat}",
                "--llama-server", str(self.server_bin),
                "--plan", "none:0:all,cpu:4:1,gpu:4:all",
                "--n-predict", "20",
                "--port", str(port),
                "--draft-threads", "6",
                "--seed", "100",
            ])
        self.assertEqual(rc, 0)

        # Verify server argv per plan item and that all server processes exited.
        server_calls = [
            json.loads(line)
            for line in self.server_log.read_text(encoding="utf-8").splitlines()
        ]
        self.assertEqual(len(server_calls), 3)
        none_argv, cpu_argv, gpu_argv = (c["argv"] for c in server_calls)
        self.assertNotIn("-md", none_argv)
        for argv in (cpu_argv, gpu_argv):
            self.assertEqual(argv[argv.index("--spec-type") + 1], "draft-simple")
        self.assertIn("-ngld", cpu_argv)
        self.assertEqual(cpu_argv[cpu_argv.index("-ngld") + 1], "0")
        self.assertEqual(cpu_argv[cpu_argv.index("-devd") + 1], "none")
        self.assertEqual(cpu_argv[cpu_argv.index("-tbd") + 1], cpu_argv[cpu_argv.index("-td") + 1])
        self.assertEqual(cpu_argv[cpu_argv.index("-td") + 1], "6")
        self.assertEqual(cpu_argv[cpu_argv.index("--spec-draft-n-max") + 1], "4")
        self.assertEqual(cpu_argv[cpu_argv.index("--spec-draft-n-min") + 1], "4")
        self.assertEqual(cpu_argv[cpu_argv.index("--spec-draft-p-min") + 1], "0")
        self.assertEqual(gpu_argv[gpu_argv.index("-ngld") + 1], "99")

        pids = [int(x) for x in self.pid_file.read_text(encoding="utf-8").splitlines() if x.strip()]
        self.assertEqual(len(pids), 3)
        for pid in pids:
            self.assertFalse(_pid_alive(pid), f"server pid {pid} still alive")

        # Per-item server logs exist.
        for label in ("none:0:all", "cpu:4:1", "gpu:4:all"):
            self.assertTrue((self.out_dir / f"server-{label}.log").is_file())

        # Verify seeds: SEED + index within each file.
        reqs = [
            json.loads(line)
            for line in self.req_log.read_text(encoding="utf-8").splitlines()
        ]
        # none:0:all -> 4 prompts * 2 modes = 8
        # cpu:4:1 -> 2 prompts (1 per file) * 2 modes = 4
        # gpu:4:all -> 4 prompts * 2 modes = 8
        self.assertEqual(len(reqs), 20)
        none_seeds = [r["body"]["seed"] for r in reqs[:8]]
        self.assertEqual(none_seeds, [100, 100, 101, 101, 100, 100, 101, 101])
        cpu_seeds = [r["body"]["seed"] for r in reqs[8:12]]
        self.assertEqual(cpu_seeds, [100, 100, 100, 100])

        # Verify metrics.jsonl fields.
        rows = [
            json.loads(line)
            for line in (self.out_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        self.assertEqual(len(rows), 20)
        expected_keys = {
            "device", "gamma", "mode", "domain", "prompt_id",
            "predicted_n", "predicted_ms", "tokens_per_s",
            "prompt_n", "prompt_ms", "draft_n", "draft_n_accepted",
            "acceptance", "error",
        }
        for row in rows:
            self.assertEqual(set(row.keys()), expected_keys)
            self.assertIsNone(row["error"])

        none_row = rows[0]
        self.assertEqual(none_row["device"], "none")
        self.assertAlmostEqual(none_row["tokens_per_s"], 20.0)
        self.assertIsNone(none_row["draft_n"])
        self.assertIsNone(none_row["draft_n_accepted"])
        self.assertIsNone(none_row["acceptance"])

        cpu_row = rows[8]
        self.assertEqual(cpu_row["device"], "cpu")
        self.assertAlmostEqual(cpu_row["tokens_per_s"], 40.0)
        self.assertAlmostEqual(cpu_row["acceptance"], 0.75)

        gpu_row = rows[12]
        self.assertEqual(gpu_row["device"], "gpu")
        self.assertAlmostEqual(gpu_row["tokens_per_s"], 50.0)
        self.assertAlmostEqual(gpu_row["acceptance"], 0.5)

        # Verify summary.json.
        summary = json.loads((self.out_dir / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(len(summary), 12)  # 3 plan items * 2 modes * 2 domains
        by_key = {(s["device"], s["gamma"], s["mode"], s["domain"]): s for s in summary}

        self.assertEqual(by_key[("none", 0, "greedy", "code")]["requests"], 2)
        self.assertIsNone(by_key[("none", 0, "greedy", "code")]["mean_acceptance"])
        self.assertAlmostEqual(by_key[("none", 0, "greedy", "code")]["mean_tokens_per_s"], 20.0)
        self.assertEqual(by_key[("none", 0, "greedy", "code")]["speedup"], 1.0)

        self.assertEqual(by_key[("cpu", 4, "sampled", "chat")]["requests"], 1)
        self.assertAlmostEqual(by_key[("cpu", 4, "sampled", "chat")]["mean_acceptance"], 0.75)
        self.assertAlmostEqual(by_key[("cpu", 4, "sampled", "chat")]["mean_tokens_per_s"], 40.0)
        self.assertAlmostEqual(by_key[("cpu", 4, "sampled", "chat")]["speedup"], 2.0)

        self.assertEqual(by_key[("gpu", 4, "greedy", "code")]["requests"], 2)
        self.assertAlmostEqual(by_key[("gpu", 4, "greedy", "code")]["mean_acceptance"], 0.5)
        self.assertAlmostEqual(by_key[("gpu", 4, "greedy", "code")]["speedup"], 2.5)

        # Verify manifest.json.
        manifest = json.loads((self.out_dir / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["run_id"], self.out_dir.name)
        file_shas = {pathlib.Path(f["path"]).name: f["sha256"] for f in manifest["files"]}
        self.assertEqual(file_shas, {"target.gguf": self.target_sha, "draft.gguf": self.draft_sha})
        self.assertIn("options", manifest["extra"])
        self.assertIn("plan", manifest["extra"])

    def test_health_timeout_records_error_and_stops_server(self) -> None:
        port = _free_port()
        with (
            mock.patch.dict(os.environ, self._env({"FAKE_SERVER_MODE": "unhealthy"})),
            mock.patch.object(a4, "HEALTH_TIMEOUT_S", 0.5),
            mock.patch.object(a4, "HEALTH_POLL_INTERVAL_S", 0.05),
        ):
            rc = a4.main([
                "--out", str(self.out_dir),
                "--target", str(self.target),
                "--draft", str(self.draft),
                "--prompts", str(self.prompts_code),
                "--llama-server", str(self.server_bin),
                "--plan", "none:0:all",
                "--port", str(port),
            ])
        self.assertEqual(rc, 0)
        rows = [
            json.loads(line)
            for line in (self.out_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        self.assertEqual(len(rows), 1)
        self.assertIsNotNone(rows[0]["error"])
        self.assertIn("timed out", rows[0]["error"])
        pids = [int(x) for x in self.pid_file.read_text(encoding="utf-8").splitlines() if x.strip()]
        for pid in pids:
            self.assertFalse(_pid_alive(pid))

    def test_sigterm_stops_server_and_exits_143(self) -> None:
        port = _free_port()
        env = os.environ.copy()
        env.update(self._env({"FAKE_SERVER_MODE": "unhealthy"}))
        env["PYTHONPATH"] = str(pathlib.Path(__file__).resolve().parents[1] / "src")
        proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "measure.a4_reference_draft",
                "--out", str(self.out_dir),
                "--target", str(self.target),
                "--draft", str(self.draft),
                "--prompts", str(self.prompts_code),
                "--llama-server", str(self.server_bin),
                "--plan", "none:0:all",
                "--port", str(port),
            ],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        deadline = time.monotonic() + 5.0
        while not self.pid_file.exists() or not self.pid_file.read_text(encoding="utf-8").strip():
            if time.monotonic() > deadline:
                proc.kill()
                self.fail("fake server never started")
            time.sleep(0.05)

        proc.send_signal(signal.SIGTERM)
        rc = proc.wait(timeout=10.0)
        self.assertEqual(rc, 128 + signal.SIGTERM)

        pids = [int(x) for x in self.pid_file.read_text(encoding="utf-8").splitlines() if x.strip()]
        for pid in pids:
            self.assertFalse(_pid_alive(pid), f"server pid {pid} survived SIGTERM cleanup")


if __name__ == "__main__":
    unittest.main()
