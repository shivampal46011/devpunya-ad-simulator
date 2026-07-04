"""Unsupervised segmentation: discover audience segments without hand labels.

1. NMF topic model over distinct copy bodies -> latent creative themes.
2. User-level feature matrix: behavior + platform/placement/state one-hots +
   topic affinities of the ads each user converted on.
3. KMeans, k chosen by silhouette; clusters profiled afterwards.

Outputs: data/copy_topics.csv, data/user_segments.parquet, printed profiles.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

DATA = Path(__file__).parent / "data"
N_TOPICS = 8


def copy_topics(ads):
    bodies = ads[ads.has_copy].drop_duplicates("copy_body")[["copy_body"]].copy()
    tfidf = TfidfVectorizer(max_features=4000, min_df=2, sublinear_tf=True,
                            token_pattern=r"[\wऀ-ॿ]{3,}")
    X = tfidf.fit_transform(bodies.copy_body)
    nmf = NMF(n_components=N_TOPICS, random_state=0, max_iter=500)
    W = nmf.fit_transform(X)
    vocab = np.array(tfidf.get_feature_names_out())
    tops = {}
    for t in range(N_TOPICS):
        tops[t] = ", ".join(vocab[np.argsort(nmf.components_[t])[::-1][:8]])
        print(f"topic {t}: {tops[t]}")
    bodies[[f"topic_{t}" for t in range(N_TOPICS)]] = W / (W.sum(1, keepdims=True) + 1e-9)
    pd.Series(tops, name="top_terms").to_csv(DATA / "copy_topics.csv")
    return bodies


def build_user_matrix():
    ads = pd.read_parquet(DATA / "ads_with_copy.parquet")
    topics = copy_topics(ads)
    ad2copy = ads.drop_duplicates("ad_name")[["ad_name", "copy_body"]]

    o = pd.read_parquet(DATA / "orders_attributed.parquet")
    plat = o.ad_platform.str.lower().replace({"fb": "facebook", "ig": "instagram"})
    o = o[plat.isin(["facebook", "instagram"])].copy()
    o["plat"] = plat
    o["otype"] = o.order_type.replace({"chadawa": "chadhawa"})
    o["place"] = (o.placement.str.lower().str.replace("mobile_", "", regex=False)
                  .where(lambda s: s.isin(["reels", "feed", "stories"]), "unknown"))
    o = o.merge(ad2copy, left_on="adname", right_on="ad_name", how="left")
    o = o.merge(topics, on="copy_body", how="left")

    tcols = [f"topic_{t}" for t in range(N_TOPICS)]
    g = o.groupby("user_id")
    u = g.agg(
        n_orders=("id", "count"),
        total_amount=("total_amount", "sum"),
        med_amount=("total_amount", "median"),
        puja_share=("otype", lambda s: (s == "puja").mean()),
        ig_share=("plat", lambda s: (s == "instagram").mean()),
        reels_share=("place", lambda s: (s == "reels").mean()),
        state=("stateinfo", lambda s: s.mode().iat[0]
               if len(s.dropna().mode()) else None),
    )
    u[tcols] = g[tcols].mean()

    top_states = u.state.value_counts().head(12).index
    st = pd.get_dummies(u.state.where(u.state.isin(top_states), "other"),
                        prefix="st", dtype=float)
    num = u[["n_orders", "total_amount", "med_amount"]].apply(np.log1p)
    beh = u[["puja_share", "ig_share", "reels_share"]].fillna(0)
    top = u[tcols].fillna(u[tcols].mean())

    X = np.hstack([
        StandardScaler().fit_transform(num),
        StandardScaler().fit_transform(beh) * 1.2,
        StandardScaler().fit_transform(top) * 1.5,
        st.values * 0.6,
    ])
    return u, X, tcols


def main():
    u, X, tcols = build_user_matrix()
    print(f"\nusers: {len(u):,} | features: {X.shape[1]}")

    sample = np.random.default_rng(0).choice(len(u), min(15000, len(u)), replace=False)
    best_k, best_s = None, -1
    for k in range(4, 13):
        km = KMeans(n_clusters=k, n_init=4, random_state=0).fit(X[sample])
        s = silhouette_score(X[sample], km.labels_, sample_size=6000, random_state=0)
        print(f"k={k:2d} silhouette={s:.3f}")
        if s > best_s:
            best_k, best_s = k, s

    km = KMeans(n_clusters=best_k, n_init=10, random_state=0).fit(X)
    u["segment"] = km.labels_
    u.reset_index().to_parquet(DATA / "user_segments.parquet")
    print(f"\nchosen k={best_k} (silhouette {best_s:.3f})\n")

    for seg, d in u.groupby("segment"):
        dom_topic = d[tcols].mean().idxmax()
        print(f"-- segment {seg}: {len(d):,} users ({len(d)/len(u):.0%}) --")
        print(f"   orders/user {d.n_orders.mean():.1f} | median order Rs.{d.med_amount.median():.0f}"
              f" | puja share {d.puja_share.mean():.0%} | IG share {d.ig_share.mean():.0%}"
              f" | reels {d.reels_share.mean():.0%}")
        print(f"   top states: {d.state.value_counts().head(3).index.tolist()}"
              f" | dominant {dom_topic}")


if __name__ == "__main__":
    main()
