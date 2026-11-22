"""Build the extra structured-output training domain (json_extra) from the other Hermes function-calling files.

Sources (NousResearch/hermes-function-calling-v1, Apache-2.0): json-mode-agentic, func-calling-singleturn,
func-calling, glaive-function-calling-5k. json-mode-singleturn, whose first 100 shuffled rows are the evaluation
prompts, is not used here (its remaining rows are already in the json domain).

Each conversation is rendered with the Qwen3 chat template: assistant turns get the empty think block that
llama.cpp inserts when thinking is disabled, and tool results become user turns wrapped in <tool_response>.

    python3 -I src/data/build_json_extra.py --src /mnt/hdd1/cpu-draft-specdec/corpora/hermes_json \
        --out /mnt/hdd1/cpu-draft-specdec/corpora/train/json_extra.jsonl
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

FILES = ["json-mode-agentic.json", "func-calling-singleturn.json", "func-calling.json",
         "glaive-function-calling-5k.json"]
ROLE = {"system": "system", "human": "user", "gpt": "assistant", "tool": "user"}


def render(conv: list[dict]) -> str | None:
    parts = []
    for m in conv:
        role = ROLE.get(m.get("from", ""))
        text = m.get("value")
        if role is None or not isinstance(text, str):
            return None
        if m["from"] == "tool":
            text = f"<tool_response>\n{text}\n</tool_response>"
        if role == "assistant":
            text = "<think>\n\n</think>\n\n" + text
        parts.append(f"<|im_start|>{role}\n{text}<|im_end|>\n")
    if not any(m.get("from") == "gpt" for m in conv):
        return None
    return "".join(parts)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    h = hashlib.sha256()
    n = chars = skipped = 0
    seen: set[str] = set()
    with args.out.open("w", encoding="utf-8") as f:
        for name in FILES:
            for r in json.loads((args.src / name).read_text(encoding="utf-8")):
                text = render(r.get("conversations") or [])
                if text is None or text in seen:
                    skipped += 1
                    continue
                seen.add(text)
                line = json.dumps({"id": f"json_extra-{n:07d}", "domain": "json_extra", "text": text},
                                  ensure_ascii=False) + "\n"
                f.write(line)
                h.update(line.encode("utf-8"))
                n += 1
                chars += len(text)
    print(json.dumps({"documents": n, "chars": chars, "skipped": skipped, "sha256": h.hexdigest()}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
