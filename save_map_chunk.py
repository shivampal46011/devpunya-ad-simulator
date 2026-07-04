"""Append an ads_get_ad_entities result file to data/ad_creative_map.jsonl.

Usage: python3 save_map_chunk.py <result_file> <account_id>
Prints rows added, dedup total, and min/max created_time for paging.
"""
import json
import sys
from pathlib import Path

from dateutil import parser as dtp

DATA = Path(__file__).parent / "data"
OUT = DATA / "ad_creative_map.jsonl"


def main():
    src, account_id = sys.argv[1], sys.argv[2]
    d = json.load(open(src))
    ents = d["ad_entities"]
    if isinstance(ents, str):
        ents = json.loads(ents)

    seen = set()
    if OUT.exists():
        with open(OUT) as f:
            for line in f:
                seen.add(json.loads(line)["ad_id"])

    added = 0
    dates = []
    with open(OUT, "a") as f:
        for e in ents:
            ct = e.get("created_time")
            if ct:
                try:
                    dates.append(dtp.parse(ct).date().isoformat())
                except (ValueError, OverflowError):
                    pass
            if e["id"] in seen:
                continue
            f.write(json.dumps({
                "ad_id": e["id"], "ad_name": e.get("name"),
                "creative_id": e.get("creative_id"),
                "created_time": dates[-1] if ct and dates else None,
                "account_id": account_id,
            }) + "\n")
            seen.add(e["id"])
            added += 1

    print(f"chunk rows: {len(ents)} | new: {added} | total mapped: {len(seen)}")
    if dates:
        print(f"created_time range in chunk: {min(dates)} -> {max(dates)}")


if __name__ == "__main__":
    main()
