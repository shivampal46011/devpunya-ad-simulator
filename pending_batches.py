"""Print the next batch of creative_ids still missing from data/creatives.jsonl.

Usage: python3 pending_batches.py <account_id> [batch_size]
Prints a JSON array of up to batch_size creative ids (default 100) belonging to
that account, or "DONE <n_total>" when every mapped creative has been fetched.
"""
import json
import sys
from pathlib import Path

DATA = Path(__file__).parent / "data"


def main():
    account_id = sys.argv[1]
    size = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    mapped = []
    seen = set()
    with open(DATA / "ad_creative_map.jsonl") as f:
        for line in f:
            row = json.loads(line)
            cid = row.get("creative_id")
            if cid and cid not in seen and row.get("account_id") == account_id:
                seen.add(cid)
                mapped.append(cid)
    have = set()
    cpath = DATA / "creatives.jsonl"
    if cpath.exists():
        with open(cpath) as f:
            for line in f:
                have.add(json.loads(line)["id"])
    todo = [c for c in mapped if c not in have]
    if not todo:
        print(f"DONE {len(mapped)}")
        return
    print(f"remaining {len(todo)} of {len(mapped)}", file=sys.stderr)
    print(json.dumps(todo[:size]))


if __name__ == "__main__":
    main()
