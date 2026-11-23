"""c1 test 2: greedy outputs with speculation (CPU draft, GPU draft, n-gram) vs target-only greedy."""
import json, subprocess, sys, time, urllib.request, pathlib
sys.path.insert(0, "src")
from measure.a4_reference_draft import build_server_cmd, parse_plan

ROOT = pathlib.Path(".")
TARGET = pathlib.Path("/mnt/hdd1/cpu-draft-specdec/gguf/Qwen3-8B-Q4_K_M.gguf")
DRAFT = pathlib.Path("/mnt/hdd1/cpu-draft-specdec/gguf/students/ds-kd-pruned-json23-96M-Q8_0.gguf")
SRV = str(ROOT / "third_party/llama.cpp/build/bin/llama-server")
P = ROOT / "experiments/stage-a-feasibility/a4-reference-draft/configs"
N_PER = 20
prompts = []
for f in ("general", "pt_fin", "json"):
    rows = [json.loads(l) for l in open(P / f"prompts_{f}.jsonl")][:N_PER]
    prompts += rows
PORT = 18090

def run(plan):
    item = parse_plan(plan)[0]
    cmd = build_server_cmd(llama_server=SRV, target=TARGET, draft=DRAFT, port=PORT, draft_threads=4, item=item)
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(300):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2); break
            except Exception:
                time.sleep(2)
        outs = []
        for r in prompts:
            body = json.dumps({"prompt": r["prompt"], "n_predict": 128, "temperature": 0, "seed": 1, "cache_prompt": False}).encode()
            req = urllib.request.Request(f"http://127.0.0.1:{PORT}/completion", body, {"Content-Type": "application/json"})
            outs.append(json.load(urllib.request.urlopen(req, timeout=300))["content"])
        return outs
    finally:
        proc.terminate(); proc.wait()

base = run("none:0:all")
res = {}
for plan in ("cpu:4:all", "gpu:4:all", "ngram:4:all"):
    o = run(plan)
    same = [a == b for a, b in zip(base, o)]
    # first divergence position in characters, for the unequal ones
    div = []
    for a, b, s in zip(base, o, same):
        if not s:
            k = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
            div.append({"chars": k, "len": len(a)})
    res[plan] = {"n": len(same), "identical": sum(same), "diverging": div}
    print(plan, sum(same), "/", len(same), flush=True)
out = sys.argv[1]
json.dump(res, open(out, "w"), indent=1)
