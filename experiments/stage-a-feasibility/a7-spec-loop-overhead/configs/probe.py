"""Probe per-cycle time of llama-server speculative decoding under flag variants."""
import json, os, signal, subprocess, sys, time, urllib.request
R = "/home/mlemes/Projects/research/cpu-draft-specdec"
G = "/mnt/hdd1/cpu-draft-specdec/gguf"
BIN = f"{R}/third_party/llama.cpp/build/bin/llama-server"
PROMPTS = [json.loads(l)["prompt"] for l in open(f"{R}/experiments/stage-a-feasibility/a4-reference-draft/configs/prompts_general.jsonl")][:int(os.environ.get("NP", "10"))]
GAMMA = 4
BASE = [BIN, "-m", f"{G}/llama-2-7b.Q4_K_M.gguf", "-ngl", "99", "-c", "2048", "-np", "1", "--port", "8091", "--no-webui",
        "-md", f"{G}/llama-68m-Q8_0.gguf", "--spec-type", "draft-simple", "--spec-draft-n-max", str(GAMMA),
        "--spec-draft-n-min", str(GAMMA), "--spec-draft-p-min", "0"]
def post(body):
    req = urllib.request.Request("http://127.0.0.1:8091/completion", json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=120))
def run(label, extra):
    p = subprocess.Popen(BASE + extra, stdout=subprocess.DEVNULL, stderr=open(f"/tmp/claude-1000/-home-mlemes-Projects/edd7c62e-b8b3-4c20-ac16-11c45d2d4a88/scratchpad/probe/{label}.log", "wb"), start_new_session=True)
    try:
        for _ in range(240):
            try:
                if urllib.request.urlopen("http://127.0.0.1:8091/health", timeout=1).status == 200: break
            except Exception: time.sleep(0.5)
        post({"prompt": "Hello", "n_predict": 8, "temperature": 0})
        pm = pn = dn = da = 0
        for pr in PROMPTS:
            t = post({"prompt": pr, "n_predict": 64, "temperature": 0, "cache_prompt": False, "seed": 0})["timings"]
            pm += t["predicted_ms"]; pn += t["predicted_n"]; dn += t.get("draft_n", 0) or 0; da += t.get("draft_n_accepted", 0) or 0
        cycles = dn / GAMMA if dn else pn
        print(f"{label:28} tps {1000*pn/pm:6.1f} acc {da/max(dn,1):.3f} cycle_ms {pm/cycles:6.2f}", flush=True)
    finally:
        os.killpg(p.pid, signal.SIGTERM); p.wait()
VARIANTS = {
    "gpu": ["-ngld", "99"],
    "cpu_td4": ["-ngld", "0", "-devd", "none", "-td", "4"],
    "cpu_td4_tbd4": ["-ngld", "0", "-devd", "none", "-td", "4", "-tbd", "4"],
    "cpu_td1_tbd1": ["-ngld", "0", "-devd", "none", "-td", "1", "-tbd", "1"],
    "cpu_td2_tbd2_t1": ["-ngld", "0", "-devd", "none", "-td", "2", "-tbd", "2", "-t", "1", "-tb", "1"],
    "cpu_td4_poll0": ["-ngld", "0", "-devd", "none", "-td", "4", "-tbd", "4", "--poll-draft", "0", "--poll-batch-draft", "0"],
    "cpu_td3_t1_C": ["-ngld", "0", "-devd", "none", "-td", "3", "-tbd", "3", "-t", "1", "-tb", "1", "-Crd", "1-3", "-Cbd", "e"],
}
for name in (sys.argv[1:] or VARIANTS):
    run(name, VARIANTS[name])
