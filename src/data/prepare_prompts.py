"""Build the Stage A prompt sets (a4) and held-out/frequency text (a5) from public corpora.

Runs locally, not in the delegation sandbox: reading parquet needs pyarrow.
    uv run --with pyarrow python -I src/data/prepare_prompts.py --corpora /mnt/hdd1/cpu-draft-specdec/corpora --out experiments/stage-a-feasibility/a4-reference-draft/configs

Domains and splits (deterministic, seed 0):
- general: WikiText-103 validation (frequency) and test (prompts, held-out).
- pt_fin: BCB Copom minutes 232-261 (frequency) and 262-282 (prompts, held-out).
- json: Hermes json-mode-singleturn rows; first 100 after a seeded shuffle are prompts, the rest frequency.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import random
import re
import sys
from pathlib import Path

PROMPTS_PER_DOMAIN = 100
WORDS_MIN, WORDS_MAX = 120, 260
HELDOUT_WORDS = 400  # continuation kept after each prompt for coverage scoring (a5)


def wikitext_articles(parquet: Path) -> list[str]:
    import pyarrow.parquet as pq

    lines = pq.read_table(parquet).column("text").to_pylist()
    articles, cur = [], []
    for line in lines:
        if re.match(r"^ = [^=].* = \n$", line):
            if cur:
                articles.append(" ".join(cur))
            cur = []
        elif line.strip() and not line.startswith(" = "):
            cur.append(line.strip())
    if cur:
        articles.append(" ".join(cur))
    def detok(a: str) -> str:
        a = a.replace(" @-@ ", "-").replace(" @,@ ", ",").replace(" @.@ ", ".")
        a = re.sub(r"\s+([,.;:!?%)\]}'])", r"\1", a)      # no space before closing punctuation
        a = re.sub(r"([(\[{])\s+", r"\1", a)              # no space after opening brackets
        a = re.sub(r"\s+'s\b", "'s", a)
        a = re.sub(r'" (.*?) "', r'"\1"', a)
        return re.sub(r"\s+", " ", a).strip()

    return [detok(a) for a in articles]


def copom_texts(atas_dir: Path, lo: int, hi: int) -> list[str]:
    out = []
    for n in range(lo, hi + 1):
        f = atas_dir / f"{n}.json"
        if not f.exists():
            continue
        d = json.loads(f.read_text(encoding="utf-8"))
        c = d.get("conteudo", d)
        c = c[0] if isinstance(c, list) and c else c
        t = c.get("textoAta") if isinstance(c, dict) else None
        if not t:
            continue
        t = re.sub(r"<(br|/p|/li|/h\d)\s*/?>", "\n", t, flags=re.I)
        t = html.unescape(re.sub(r"<[^>]+>", " ", t))
        out.append(re.sub(r"[ \t ]+", " ", t).strip())
    return out


def windows(texts: list[str], rng: random.Random, n: int) -> list[tuple[str, str]]:
    """Pick n (prompt, continuation) word windows from distinct random positions."""
    cands = []
    for ti, t in enumerate(texts):
        words = t.split()
        need = WORDS_MAX + HELDOUT_WORDS
        for start in range(0, max(0, len(words) - need), 150):
            cands.append((ti, start))
    rng.shuffle(cands)
    out = []
    for ti, start in cands[:n]:
        words = texts[ti].split()
        k = rng.randint(WORDS_MIN, WORDS_MAX)
        out.append((" ".join(words[start:start + k]), " ".join(words[start + k:start + k + HELDOUT_WORDS])))
    if len(out) < n:
        raise SystemExit(f"only {len(out)} windows available, need {n}")
    return out


def qwen_chat(system: str, user: str) -> str:
    # Qwen3 chat template with thinking disabled (empty think block), as used by llama.cpp for Qwen3.
    return (f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n"
            "<|im_start|>assistant\n<think>\n\n</think>\n\n")


def hermes_rows(path: Path) -> list[dict]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    out = []
    for r in rows:
        conv = r.get("conversations") or []
        sys_msg = next((m["value"] for m in conv if m.get("from") == "system"), None)
        user = next((m["value"] for m in conv if m.get("from") == "human"), None)
        answer = next((m["value"] for m in conv if m.get("from") == "gpt"), None)
        if sys_msg and user and answer:
            out.append({"system": sys_msg, "user": user, "answer": answer})
    return out


def write_jsonl(path: Path, rows: list[dict]) -> str:
    data = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    path.write_text(data, encoding="utf-8")
    return hashlib.sha256(data.encode()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpora", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True, help="prompt-set directory (committed)")
    ap.add_argument("--freq-out", type=Path, default=None, help="frequency text directory (default: <corpora>/derived)")
    args = ap.parse_args()
    rng = random.Random(0)
    freq_dir = args.freq_out or args.corpora / "derived"
    args.out.mkdir(parents=True, exist_ok=True)
    freq_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"seed": 0, "prompts_per_domain": PROMPTS_PER_DOMAIN, "sets": {}}

    test = wikitext_articles(args.corpora / "wikitext103/test.parquet")
    val = wikitext_articles(args.corpora / "wikitext103/validation.parquet")
    gen = [{"id": f"general-{i:03d}", "domain": "general", "prompt": p, "heldout": h,
            "source": "Salesforce/wikitext wikitext-103-raw-v1 test (CC-BY-SA-3.0)"}
           for i, (p, h) in enumerate(windows(test, rng, PROMPTS_PER_DOMAIN))]
    (freq_dir / "general.txt").write_text("\n\n".join(val), encoding="utf-8")

    held = copom_texts(args.corpora / "copom/atas", 262, 282)
    freq = copom_texts(args.corpora / "copom/atas", 232, 261)
    pt = [{"id": f"pt_fin-{i:03d}", "domain": "pt_fin", "prompt": p, "heldout": h,
           "source": "Banco Central do Brasil, atas do Copom 262-282 (dados abertos)"}
          for i, (p, h) in enumerate(windows(held, rng, PROMPTS_PER_DOMAIN))]
    (freq_dir / "pt_fin.txt").write_text("\n\n".join(freq), encoding="utf-8")

    hrows = hermes_rows(args.corpora / "hermes_json/json-mode-singleturn.json")
    rng.shuffle(hrows)
    js = [{"id": f"json-{i:03d}", "domain": "json", "prompt": qwen_chat(r["system"], r["user"]),
           "heldout": r["answer"], "source": "NousResearch/hermes-function-calling-v1 json-mode-singleturn (Apache-2.0)"}
          for i, r in enumerate(hrows[:PROMPTS_PER_DOMAIN])]
    (freq_dir / "json.txt").write_text(
        "\n\n".join(qwen_chat(r["system"], r["user"]) + r["answer"] for r in hrows[PROMPTS_PER_DOMAIN:]),
        encoding="utf-8")

    for name, rows in (("general", gen), ("pt_fin", pt), ("json", js)):
        manifest["sets"][name] = {"file": f"prompts_{name}.jsonl", "rows": len(rows),
                                  "sha256": write_jsonl(args.out / f"prompts_{name}.jsonl", rows)}
    for name in ("general", "pt_fin", "json"):
        p = freq_dir / f"{name}.txt"
        manifest["sets"][name]["frequency_text"] = {"path": str(p), "bytes": p.stat().st_size,
                                                    "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
    (args.out / "prompts_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: {"rows": v["rows"], "freq_bytes": v["frequency_text"]["bytes"]} for k, v in manifest["sets"].items()}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
