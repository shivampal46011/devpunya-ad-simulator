"""Append an ads_get_creatives result (inline JSON string or file path) to
data/creatives.jsonl, deduped by creative id.

Usage: python3 save_creatives_chunk.py <result_file>
The result file may be either the raw tool JSON {"ad_creatives": [...]} or the
persisted-output wrapper {"ad_creatives": "<json string>"}.
"""
import json
import sys
from pathlib import Path

DATA = Path(__file__).parent / "data"
OUT = DATA / "creatives.jsonl"
KEEP = ["id", "name", "object_type", "body", "title", "call_to_action_type",
        "video_id", "image_url", "link_url", "status"]


def main():
    d = json.load(open(sys.argv[1]))
    creatives = d.get("ad_creatives", d)
    if isinstance(creatives, str):
        creatives = json.loads(creatives)

    seen = set()
    if OUT.exists():
        with open(OUT) as f:
            for line in f:
                seen.add(json.loads(line)["id"])

    added = 0
    with open(OUT, "a") as f:
        for c in creatives:
            if c["id"] in seen:
                continue
            f.write(json.dumps({k: c.get(k) for k in KEEP}) + "\n")
            seen.add(c["id"])
            added += 1
    print(f"chunk: {len(creatives)} | new: {added} | total creatives: {len(seen)}")


if __name__ == "__main__":
    main()
