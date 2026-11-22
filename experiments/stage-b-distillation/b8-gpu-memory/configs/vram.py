import json, os, pathlib, signal, subprocess, sys, time, urllib.request
sys.path.insert(0, "src")
from measure.a4_reference_draft import PlanItem, build_server_cmd
H = pathlib.Path("/mnt/hdd1/cpu-draft-specdec")
out = []
for ctx in (None, 8192, 12288):
    for dev in ("none", "gpu", "cpu"):
        cmd = build_server_cmd(llama_server="third_party/llama.cpp/build/bin/llama-server",
                               target=H / "gguf/Qwen3-8B-Q4_K_M.gguf",
                               draft=H / "gguf/students/ds-kd-pruned-json23-96M-Q8_0.gguf", port=8099,
                               draft_threads=4, item=PlanItem(device=dev, gamma=4, subset=None, label="x"))
        if ctx:
            i = cmd.index("-c") if "-c" in cmd else None
            if i is not None: cmd[i + 1] = str(ctx)
            else: cmd += ["-c", str(ctx)]
        log = open(f'/tmp/claude-1000/-home-mlemes-Projects/edd7c62e-b8b3-4c20-ac16-11c45d2d4a88/scratchpad/vram-{ctx}-{dev}.log', 'w')
        p = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            for _ in range(240):
                if p.poll() is not None: break
                try:
                    if urllib.request.urlopen("http://127.0.0.1:8099/health", timeout=1).status == 200: break
                except Exception: time.sleep(0.5)
            time.sleep(2)
            q = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                               capture_output=True, text=True).stdout
            mib = sum(int(l.split(",")[1]) for l in q.strip().splitlines() if int(l.split(",")[0]) == p.pid)
            out.append({"ctx": ctx or "default", "draft": dev, "server_vram_mib": mib, "alive": p.poll() is None})
            print(out[-1], flush=True)
        finally:
            
            try: os.killpg(p.pid, signal.SIGKILL)
            except ProcessLookupError: pass
            p.wait(timeout=30)
print(json.dumps(out))
