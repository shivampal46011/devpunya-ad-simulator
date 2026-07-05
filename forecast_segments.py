"""Forecast new / emerging audience segments.

1. Drift & misfit detection: are recent users increasingly poorly explained
   by the 7 known clusters? Cluster the recent misfits -> candidate new segments.
2. Segment growth trajectories by quarter.
3. Whitespace: problem themes with proven conversion efficiency but tiny
   spend share -> segments you could grow deliberately.
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from cluster_segments import build_user_matrix

DATA = Path(__file__).parent / "data"

THEMES = {
    "enemies_court": r"baglamukhi|shatru|enem|court|victory",
    "wealth_property": r"varahi|matrika|laxmi|lakshmi|wealth|property|land",
    "protection_blackmagic": r"bhairav|kaal|black ?magic|kala ?jadu|protection",
    "children": r"santan|putra|kameshti",
    "career_luck": r"rahu|shani|navgrah|mahadasha|career",
    "health": r"mrityunjaya|health|rog|arogya|dhanvantari",
    "marriage": r"vivah|marriage|mangal ?dosh|katyayani|kundli ?match",
    "ancestors_peace": r"pitru|shradh|narayan ?bali|tarpan",
    "generic_devotion": r"rudrabhishek|jaap|abhishek|shaktipeeth|chadhawa|chadawa",
}


def theme_of(name):
    s = (name or "").lower()
    for t, pat in THEMES.items():
        if re.search(pat, s):
            return t
    return "other"


def main():
    u, X, tcols = build_user_matrix()
    seg = pd.read_parquet(DATA / "user_segments.parquet").set_index("user_id")
    u = u.join(seg[["segment"]])

    o = pd.read_parquet(DATA / "orders_attributed.parquet")
    first = o.groupby("user_id").created_date.min()
    u["first_q"] = pd.PeriodIndex(u.index.map(first), freq="Q").astype(str)

    km = KMeans(n_clusters=7, n_init=10, random_state=0).fit(X)
    d = np.linalg.norm(X - km.cluster_centers_[km.labels_], axis=1)
    u["dist"] = d
    thr = np.quantile(d, 0.90)
    u["misfit"] = d > thr

    print("=" * 25, "1. DRIFT / MISFIT BY QUARTER", "=" * 25)
    q = u.groupby("first_q").agg(users=("dist", "size"),
                                 mean_dist=("dist", "mean"),
                                 misfit_share=("misfit", "mean")).round(3)
    print(q.to_string())

    recent = u[(u.first_q >= "2026Q1") & u.misfit]
    print(f"\nrecent misfits (2026Q1+, poorly explained): {len(recent):,}")
    if len(recent) >= 200:
        Xr = X[u.reset_index().index[(u.first_q >= "2026Q1").values
                                     & u.misfit.values]]
        km2 = KMeans(n_clusters=3, n_init=10, random_state=0).fit(Xr)
        recent = recent.copy()
        recent["cand"] = km2.labels_
        print("\ncandidate emerging segments (clusters of recent misfits):")
        for c, dd in recent.groupby("cand"):
            tm = dd[tcols].fillna(0).mean()
            print(f"  cand {c}: {len(dd):,} users | med order Rs.{dd.med_amount.median():.0f}"
                  f" | orders/user {dd.n_orders.mean():.1f} | puja {dd.puja_share.mean():.0%}"
                  f" | IG {dd.ig_share.mean():.0%} | reels {dd.reels_share.mean():.0%}"
                  f" | dominant {tm.idxmax() if tm.max() > 0 else 'n/a'}")

    print("\n" + "=" * 25, "2. SEGMENT GROWTH TRAJECTORY", "=" * 25)
    tr = pd.crosstab(u.first_q, u.segment, normalize="index") * 100
    print(tr.round(0).astype(int).to_string())
    last3 = tr.iloc[-3:]
    slope = (last3.iloc[-1] - last3.iloc[0]) / 2
    print("\ntrend (pp/quarter over last 3 quarters):",
          {int(k): round(v, 1) for k, v in slope.items()})

    print("\n" + "=" * 25, "3. WHITESPACE THEMES", "=" * 25)
    ads = pd.read_parquet(DATA / "ads_with_copy.parquet")
    ads["theme"] = [theme_of(f"{n} {c}") for n, c in
                    zip(ads.ad_name.fillna(""), ads.campaign.fillna(""))]
    w = ads.groupby("theme").agg(ads=("ad_id", "count"), spend=("spend", "sum"),
                                 orders=("orders", "sum"),
                                 revenue=("revenue", "sum"))
    w["spend_share"] = w.spend / w.spend.sum()
    w["cpa"] = w.spend / w.orders.replace(0, np.nan)
    w["roas"] = w.revenue / w.spend
    w = w.sort_values("spend", ascending=False)
    print(w.round(2).to_string())


if __name__ == "__main__":
    main()
