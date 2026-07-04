"""Score the persona panel against actual CTR outcomes, head-to-head with the
Phase 1 ML model, plus an ensemble of both.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from backtest import build_copy_table, featurize

DATA = Path(__file__).parent / "data"

USER_W = {"P1": .13, "P2": .13, "P3": .04, "P4": .13, "P5": .13, "P6": .19,
          "P7": .18, "P8": .015, "P9": .07, "P10": .07, "P11": .03,
          "P12": .03, "P13": .04}
REV_W = {"P1": .25, "P2": .26, "P3": .15, "P4": .20, "P5": .20, "P6": .06,
         "P7": .05, "P8": .02, "P9": .04, "P10": .04, "P11": .05,
         "P12": .03, "P13": .02}


def load_votes():
    votes = []
    for f in sorted(DATA.glob("panel_votes_*.json")):
        votes.extend(json.load(open(f)))
    return {v["pair_id"]: v for v in votes}


def ml_scores():
    t = build_copy_table()
    cutoff = t.first_seen.quantile(0.6).normalize()
    train, test = t[t.first_seen < cutoff], t[t.first_seen >= cutoff]
    ytr = train["ctr"].rank(pct=True).values
    _, (Ftr, Fte) = featurize(train, test)
    model = HistGradientBoostingRegressor(max_depth=3, max_iter=300,
                                          learning_rate=0.05,
                                          l2_regularization=1.0, random_state=0)
    model.fit(Ftr, ytr)
    return dict(zip(test.copy_key, model.predict(Fte)))


def main():
    votes = load_votes()
    truth = json.load(open(DATA / "validation_truth.json"))
    pairs = json.load(open(DATA / "validation_pairs.json"))
    ml = ml_scores()

    rows = []
    for p in pairs:
        pid = p["pair_id"]
        if pid not in votes:
            continue
        v = votes[pid]["votes"]
        share_users = sum(USER_W[k] * (1 if v[k]["choice"] == "A" else 0)
                          for k in v) / sum(USER_W[k] for k in v)
        share_rev = sum(REV_W[k] * (1 if v[k]["choice"] == "A" else 0)
                        for k in v) / sum(REV_W[k] for k in v)
        share_conf = sum(v[k]["conf"] * (1 if v[k]["choice"] == "A" else -1)
                         for k in v)
        maj = sum(1 if v[k]["choice"] == "A" else -1 for k in v)
        ml_a, ml_b = ml.get(p["A"]["copy"]), ml.get(p["B"]["copy"])
        ml_diff = (ml_a - ml_b) if ml_a is not None and ml_b is not None else None
        rows.append({
            "pair_id": pid, "actual": truth[pid]["winner"],
            "panel_majority": "A" if maj > 0 else "B",
            "panel_users": "A" if share_users > .5 else "B",
            "panel_rev": "A" if share_rev > .5 else "B",
            "panel_conf": "A" if share_conf > 0 else "B",
            "ml": None if ml_diff is None else ("A" if ml_diff > 0 else "B"),
            "share_users": share_users, "ml_diff": ml_diff,
        })
    df = pd.DataFrame(rows)
    n = len(df)
    print(f"pairs scored: {n}")
    for col in ["panel_majority", "panel_users", "panel_rev", "panel_conf", "ml"]:
        d = df[df[col].notna()]
        print(f"  {col:15s}: {(d[col] == d.actual).mean():.1%} ({len(d)} pairs)")

    both = df[df.ml.notna()].copy()
    ens = (both.share_users - 0.5) * 2 + np.clip(both.ml_diff * 4, -1, 1)
    both["ensemble"] = np.where(ens > 0, "A", "B")
    print(f"  {'ensemble':15s}: {(both.ensemble == both.actual).mean():.1%} ({len(both)} pairs)")

    per = []
    for k in USER_W:
        ok = tot = 0
        for p in pairs:
            pid = p["pair_id"]
            if pid in votes and k in votes[pid]["votes"]:
                tot += 1
                ok += votes[pid]["votes"][k]["choice"] == truth[pid]["winner"]
        per.append((k, ok / tot if tot else np.nan, tot))
    print("\nper-persona accuracy:")
    for k, acc, tot in sorted(per, key=lambda x: -x[1]):
        print(f"  {k:4s} {acc:.0%}  ({tot} pairs)")

    df.to_csv(DATA / "panel_scores.csv", index=False)


if __name__ == "__main__":
    main()
