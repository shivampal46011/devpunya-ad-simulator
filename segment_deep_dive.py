"""Deep-dive on discovered segments:
1. Value & loyalty: revenue share, LTV, repeat depth per segment.
2. Sub-clustering inside the two mega-segments.
3. Creative response profile per segment: which copy attributes convert it.
4. Acquisition trend by month.
Outputs printed + data/segment_creative_profile.csv
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

DATA = Path(__file__).parent / "data"

URGENCY = re.compile(r"(limited|last chance|don.t miss|today|tonight|hurry|only \d|rare|अंतिम|आज|सिर्फ)", re.I)
SOCIAL = re.compile(r"(devotees|families|celebrit|vip|builders|हज़ार|लाख|\d{2,}[,\d]*\s*(log|people))", re.I)
PRICE = re.compile(r"(₹|rs\.?\s?\d|/-)", re.I)
FEAR = re.compile(r"(black magic|negative energ|shatru|enem|dosh|kaal|संकट|शत्रु|काला जादू|भय|dushman)", re.I)
BENEFIT = re.compile(r"(blessing|prosperit|wealth|success|growth|समृद्धि|सफलता|धन|विजय|victory)", re.I)


def copy_flags(body):
    if not isinstance(body, str) or not body.strip():
        return None
    dev = len(re.findall(r"[ऀ-ॿ]", body))
    return {
        "hindi_script": dev / max(len(body), 1) > 0.3,
        "urgency": bool(URGENCY.search(body)),
        "social_proof": bool(SOCIAL.search(body)),
        "price_mention": bool(PRICE.search(body)),
        "fear_frame": bool(FEAR.search(body)),
        "benefit_frame": bool(BENEFIT.search(body)),
        "long_copy": len(body) > 700,
    }


def main():
    u = pd.read_parquet(DATA / "user_segments.parquet")
    o = pd.read_parquet(DATA / "orders_attributed.parquet")
    ads = pd.read_parquet(DATA / "ads_with_copy.parquet")

    plat = o.ad_platform.str.lower().replace({"fb": "facebook", "ig": "instagram"})
    o = o[plat.isin(["facebook", "instagram"])].copy()
    o = o.merge(u[["user_id", "segment"]], on="user_id", how="inner")
    o = o.merge(ads.drop_duplicates("ad_name")[["ad_name", "copy_body", "format"]],
                left_on="adname", right_on="ad_name", how="left")

    print("=" * 30, "1. VALUE & LOYALTY", "=" * 30)
    seg_rev = o.groupby("segment").total_amount.sum()
    v = u.groupby("segment").agg(users=("user_id", "count"),
                                 ltv=("total_amount", "mean"),
                                 p90_ltv=("total_amount", lambda s: s.quantile(.9)),
                                 repeat_2plus=("n_orders", lambda s: (s >= 2).mean()),
                                 heavy_5plus=("n_orders", lambda s: (s >= 5).mean()))
    v["rev_share"] = (seg_rev / seg_rev.sum()).round(3)
    v["ltv"] = v.ltv.round(0)
    v["p90_ltv"] = v.p90_ltv.round(0)
    v[["repeat_2plus", "heavy_5plus"]] = v[["repeat_2plus", "heavy_5plus"]].round(3)
    print(v.sort_values("rev_share", ascending=False).to_string())

    print("\n" + "=" * 30, "2. SUB-CLUSTERS IN MEGA-SEGMENTS", "=" * 30)
    tcols = [c for c in u.columns if c.startswith("topic_")]
    for seg in [3, 6]:
        d = u[u.segment == seg].copy()
        num = d[["n_orders", "total_amount", "med_amount"]].apply(np.log1p)
        beh = d[["puja_share", "ig_share", "reels_share"]].fillna(0)
        top = d[tcols].fillna(0)
        X = np.hstack([StandardScaler().fit_transform(num),
                       StandardScaler().fit_transform(beh),
                       StandardScaler().fit_transform(top) * 1.5])
        best = max(range(2, 5), key=lambda k: silhouette_score(
            X, KMeans(k, n_init=4, random_state=0).fit_predict(X),
            sample_size=5000, random_state=0))
        lab = KMeans(best, n_init=8, random_state=0).fit_predict(X)
        d["sub"] = lab
        print(f"\nsegment {seg} -> {best} sub-clusters:")
        for s, dd in d.groupby("sub"):
            tm = dd[tcols].fillna(0).mean()
            dom = tm.idxmax() if tm.max() > 0 else "no-copy-data"
            print(f"  {seg}.{s}: {len(dd):,} users | ltv Rs.{dd.total_amount.mean():.0f} "
                  f"| orders {dd.n_orders.mean():.1f} | puja {dd.puja_share.mean():.0%} "
                  f"| IG {dd.ig_share.mean():.0%} | reels {dd.reels_share.mean():.0%} | {dom}")

    print("\n" + "=" * 30, "3. CREATIVE RESPONSE PROFILE", "=" * 30)
    flags = o.copy_body.map(copy_flags)
    fl = pd.DataFrame([f if f else {} for f in flags], index=o.index)
    o2 = pd.concat([o[["segment", "format"]], fl], axis=1).dropna(subset=["urgency"])
    prof = o2.groupby("segment")[["hindi_script", "urgency", "social_proof",
                                  "price_mention", "fear_frame", "benefit_frame",
                                  "long_copy"]].mean()
    base = o2[["hindi_script", "urgency", "social_proof", "price_mention",
               "fear_frame", "benefit_frame", "long_copy"]].mean()
    lift = (prof / base - 1)
    print("\nshare of converting ads with attribute (overall baseline in header):")
    hdr = " | ".join(f"{c} {base[c]:.0%}" for c in base.index)
    print("baseline:", hdr)
    print((prof * 100).round(0).astype(int).to_string())
    print("\nlift vs baseline (%, + = segment over-responds to attribute):")
    print((lift * 100).round(0).astype(int).to_string())
    prof.to_csv(DATA / "segment_creative_profile.csv")

    print("\n" + "=" * 30, "4. ACQUISITION TREND", "=" * 30)
    first = o.groupby("user_id").agg(seg=("segment", "first"),
                                     first_order=("created_date", "min"))
    first["q"] = pd.PeriodIndex(first.first_order, freq="Q").astype(str)
    tr = pd.crosstab(first.q, first.seg, normalize="index")
    print((tr * 100).round(0).astype(int).to_string())


if __name__ == "__main__":
    main()
