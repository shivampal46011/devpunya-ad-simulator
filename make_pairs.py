"""Build blind validation pairs from the backtest's held-out test copies.

Pairs are same-offering-type, relative CTR difference >= 20%. Outputs:
  data/validation_pairs.json  - what the persona panel sees (no outcomes)
  data/validation_truth.json  - actual winners, for scoring only
"""
import itertools
import json
from pathlib import Path

import numpy as np

from backtest import build_copy_table

DATA = Path(__file__).parent / "data"
N_PAIRS = 50


def main():
    t = build_copy_table()
    cutoff = t.first_seen.quantile(0.6).normalize()
    test = t[t.first_seen >= cutoff].reset_index(drop=True)
    print(f"test copies: {len(test)} (first_seen >= {cutoff.date()})")

    cands = []
    for i, j in itertools.combinations(range(len(test)), 2):
        a, b = test.iloc[i], test.iloc[j]
        if a.offering_type != b.offering_type:
            continue
        if abs(a.ctr - b.ctr) < 0.2 * max(a.ctr, b.ctr):
            continue
        cands.append((i, j))
    print(f"candidate pairs: {len(cands)}")

    rng = np.random.default_rng(42)
    picks = rng.choice(len(cands), size=min(N_PAIRS, len(cands)), replace=False)

    pairs, truth = [], {}
    for n, k in enumerate(sorted(picks)):
        i, j = cands[k]
        a, b = test.iloc[i], test.iloc[j]
        if rng.random() < 0.5:            # randomize A/B position
            a, b = b, a
        pid = f"pair_{n:02d}"
        pairs.append({
            "pair_id": pid,
            "offering_type": a.offering_type,
            "A": {"copy": a.copy_key, "title": a.title or "",
                  "language": a.language, "format": a.fmt},
            "B": {"copy": b.copy_key, "title": b.title or "",
                  "language": b.language, "format": b.fmt},
        })
        truth[pid] = {"winner": "A" if a.ctr > b.ctr else "B",
                      "ctr_A": round(float(a.ctr), 5), "ctr_B": round(float(b.ctr), 5),
                      "impr_A": int(a.impressions), "impr_B": int(b.impressions)}

    json.dump(pairs, open(DATA / "validation_pairs.json", "w"),
              ensure_ascii=False, indent=1)
    json.dump(truth, open(DATA / "validation_truth.json", "w"), indent=1)
    print(f"wrote {len(pairs)} pairs")


if __name__ == "__main__":
    main()
