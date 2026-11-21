"""Build the Stage B training corpus (docs/specs/stage-b-tooling.md section 1).

Runs locally (pyarrow):
    uv run -q --no-project --with pyarrow python -I src/data/build_train_corpus.py \
        --corpora /mnt/hdd1/cpu-draft-specdec/corpora --out /mnt/hdd1/cpu-draft-specdec/corpora/train

Evaluation material is excluded: WikiText test split, Copom minutes 262-282, and the first 100 Hermes rows
after the seeded shuffle used by prepare_prompts.py.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_prompts import PROMPTS_PER_DOMAIN, copom_texts, hermes_rows, qwen_chat, wikitext_articles  # noqa: E402


def take(docs: list[str], budget_chars: int, rng: random.Random) -> list[str]:
    """Whole documents in seeded random order until the character budget is reached."""
    order = list(range(len(docs)))
    rng.shuffle(order)
    out, total = [], 0
    for i in order:
        if total >= budget_chars:
            break
        if docs[i].strip():
            out.append(docs[i])
            total += len(docs[i])
    return out


def ptwiki_texts(paths: list[Path]) -> list[str]:
    import pyarrow.parquet as pq

    texts: list[str] = []
    for p in paths:
        texts.extend(pq.read_table(p, columns=["text"]).column("text").to_pylist())
    return texts


def write_domain(path: Path, domain: str, docs: list[str]) -> dict:
    h = hashlib.sha256()
    with path.open("w", encoding="utf-8") as f:
        for i, text in enumerate(docs):
            line = json.dumps({"id": f"{domain}-{i:07d}", "domain": domain, "text": text}, ensure_ascii=False) + "\n"
            f.write(line)
            h.update(line.encode("utf-8"))
    return {"file": path.name, "documents": len(docs), "chars": sum(map(len, docs)), "sha256": h.hexdigest()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpora", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--general-chars", type=int, default=60_000_000)
    ap.add_argument("--pt-general-chars", type=int, default=60_000_000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    c = args.corpora
    sets = {
        "general": take(wikitext_articles(c / "wikitext103/train/train-00000.parquet")
                        + wikitext_articles(c / "wikitext103/train/train-00001.parquet"), args.general_chars, rng),
        "pt_general": take(ptwiki_texts(sorted((c / "ptwiki").glob("train-*.parquet"))), args.pt_general_chars, rng),
        "pt_fin": copom_texts(c / "copom/atas", 232, 261),
    }
    hrows = hermes_rows(c / "hermes_json/json-mode-singleturn.json")
    random.Random(0).shuffle(hrows)  # same order as prepare_prompts.py; first 100 are evaluation prompts
    sets["json"] = [qwen_chat(r["system"], r["user"]) + r["answer"] + "<|im_end|>" for r in hrows[PROMPTS_PER_DOMAIN:]]

    manifest = {"seed": args.seed, "domains": {}, "sources": {
        "general": "Salesforce/wikitext wikitext-103-raw-v1 train (CC-BY-SA-3.0)",
        "pt_general": "wikimedia/wikipedia 20231101.pt shards 0-1 (CC-BY-SA-4.0)",
        "pt_fin": "Banco Central do Brasil, atas do Copom 232-261 (dados abertos); 262-282 held out",
        "json": "NousResearch/hermes-function-calling-v1 json-mode-singleturn (Apache-2.0), rows after the 100 evaluation prompts",
    }, "parse_special": {"json": True}}
    for name, docs in sets.items():
        manifest["domains"][name] = write_domain(args.out / f"{name}.jsonl", name, docs)
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: (v["documents"], v["chars"]) for k, v in manifest["domains"].items()}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
