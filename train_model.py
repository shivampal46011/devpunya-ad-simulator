"""Train the production CTR-percentile model on ALL history and save
data/model.joblib (backtest.py remains the honest evaluation; this just
fits the same architecture on the full data for the app to use).
"""
from pathlib import Path

import joblib
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.feature_extraction.text import TfidfVectorizer

from backtest import build_copy_table, copy_features

DATA = Path(__file__).parent / "data"


def main():
    t = build_copy_table()
    y = t["ctr"].rank(pct=True).values

    tfidf = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                            min_df=3, max_features=20000, sublinear_tf=True)
    Xt = tfidf.fit_transform(t.copy_key)
    svd = TruncatedSVD(n_components=min(60, Xt.shape[1] - 1), random_state=0)
    T = svd.fit_transform(Xt)

    cats = {c: t[c].dropna().unique().tolist()[:8]
            for c in ["offering_type", "language", "fmt", "cta"]}
    m = copy_features(t.copy_key)
    m["log_n_ads"] = np.log1p(t.n_ads.values)
    for col, levels in cats.items():
        for lv in levels:
            m[f"{col}={lv}"] = (t[col] == lv).astype(int).values

    model = HistGradientBoostingRegressor(max_depth=3, max_iter=300,
                                          learning_rate=0.05,
                                          l2_regularization=1.0, random_state=0)
    model.fit(np.hstack([m.values, T]), y)
    joblib.dump({"tfidf": tfidf, "svd": svd, "model": model, "cats": cats,
                 "n_copies": len(t)}, DATA / "model.joblib")
    print(f"trained on {len(t)} copies -> data/model.joblib")


if __name__ == "__main__":
    main()
