"""Generate the Stage B offline-acceptance evaluation set (docs/specs/stage-b-tooling.md section 3.7).

For each a4 prompt, the deployed target (llama-server, greedy) generates N tokens; for every generated position the
server returns the target's top-32 log-probabilities from a plain softmax of the logits (temperature < 0 mode).
Output: one JSONL file per domain, each line
    {"id", "domain", "prompt_tokens": [...], "gen_tokens": [...], "top_ids": [[...] * N], "top_logp": [[...] * N]}

    python3 src/data/gen_eval_set.py --target GGUF --prompts DIR --out DIR [--n-predict 128]
"""
from __future__ import annotations

import argparse
import http.client
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SERVER = ROOT / "third_party/llama.cpp/build/bin/llama-server"


def call(port: int, path: str, body: dict) -> dict:
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=300)
    try:
        conn.request("POST", path, json.dumps(body), {"Content-Type": "application/json"})
        resp = conn.getresponse()
        data = resp.read()
        if resp.status != 200:
            raise RuntimeError(f"{path} HTTP {resp.status}: {data[:200]!r}")
        return json.loads(data)
    finally:
        conn.close()


def wait_health(port: int, proc: subprocess.Popen, timeout: float = 300) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"llama-server exited with {proc.returncode}")
        try:
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=2)
            conn.request("GET", "/health")
            if conn.getresponse().status == 200:
                return
        except OSError:
            pass
        time.sleep(0.5)
    raise RuntimeError("health timeout")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True)
    ap.add_argument("--prompts", type=Path, required=True, help="directory with prompts_<domain>.jsonl")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--domains", default="general,pt_fin,json")
    ap.add_argument("--n-predict", type=int, default=128)
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--port", type=int, default=8093)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    cmd = [str(SERVER), "-m", args.target, "-ngl", "99", "-c", "2048", "-np", "1", "--port", str(args.port), "--no-webui"]
    log = (args.out / "server.log").open("wb")
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    try:
        wait_health(args.port, proc)
        for domain in args.domains.split(","):
            rows = [json.loads(l) for l in (args.prompts / f"prompts_{domain}.jsonl").open(encoding="utf-8")]
            with (args.out / f"eval_{domain}.jsonl").open("w", encoding="utf-8") as f:
                for r in rows:
                    # json prompts already contain the chat template: parse special tokens; plain text does not.
                    special = domain == "json"
                    ptoks = call(args.port, "/tokenize", {"content": r["prompt"], "add_special": False,
                                                          "parse_special": special})["tokens"]
                    res = call(args.port, "/completion", {"prompt": ptoks, "n_predict": args.n_predict,
                                                          "temperature": -1, "n_probs": args.k,
                                                          "return_tokens": True, "cache_prompt": False,
                                                          "ignore_eos": False})
                    probs = res["completion_probabilities"]
                    gen = res["tokens"]
                    if len(probs) != len(gen):
                        raise RuntimeError(f"{r['id']}: {len(probs)} prob rows for {len(gen)} tokens")
                    top_ids = [[t["id"] for t in p["top_logprobs"]] for p in probs]
                    top_logp = [[t["logprob"] for t in p["top_logprobs"]] for p in probs]
                    for g, ids in zip(gen, top_ids):
                        if ids[0] != g:  # greedy: generated token must be the top-1
                            print(f"warning: {r['id']} generated {g} but top-1 {ids[0]}", file=sys.stderr)
                    f.write(json.dumps({"id": r["id"], "domain": domain, "prompt_tokens": ptoks, "gen_tokens": gen,
                                        "top_ids": top_ids, "top_logp": top_logp}) + "\n")
            print(f"{domain}: {len(rows)} rows", file=sys.stderr)
    finally:
        os.killpg(proc.pid, signal.SIGTERM)
        proc.wait()
        log.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
