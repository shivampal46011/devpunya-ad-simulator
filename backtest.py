"""Phase 1 backtest: how well can ad copy + metadata predict performance?

Unit of modeling = distinct copy body (performance pooled across all ads
sharing it). Temporal split by the copy's first appearance, so no copy text
leaks between train and test. Reports Spearman rank correlation, pairwise
win-rate, and top-quartile precision for CTR and ROAS, for a metadata-only
model vs metadata+text.
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.feature_extraction.text import TfidfVectorizer

DATA = Path(__file__).parent / "data"
MIN_IMPR, MIN_SPEND = 3000, 1000
EMOJI_RE = re.compile("[\U0001F000-\U0001FAFF☀-➿]")
BULLET_RE = re.compile("[✅☑✔•\U0001F4FF\U0001F549]")

URGENCY = r"(limited|last chance|don.t miss|today|tonight|hurry|only \d|rare|अंतिम|आज|सिर्फ)"
SOCIAL = r"(devotees|families|celebrit|vip|builders|\d{2,}[,\d]* (log|people)|हज़ार|लाख)"


def copy_features(s):
    body = s.fillna("")
    dev = body.str.count(r"[ऀ-ॿ]")
    total = body.str.len().replace(0, 1)
    return pd.DataFrame({
        "len_chars": body.str.len(),
        "n_lines": body.str.count("\n") + 1,
        "n_emoji": body.map(lambda x: len(EMOJI_RE.findall(x))),
        "n_bullets": body.map(lambda x: len(BULLET_RE.findall(x))),
        "n_questions": body.str.count(r"\?"),
        "n_exclaim": body.str.count(r"!"),
        "pct_devanagari": dev / total,
        "has_urgency": body.str.contains(URGENCY, case=False, regex=True).astype(int),
        "has_social_proof": body.str.contains(SOCIAL, case=False, regex=True).astype(int),
        "has_price": body.str.contains(r"(₹|rs\.?\s?\d|/-)", case=False).astype(int),
        "has_book_now": body.str.contains(r"book now", case=False).astype(int),
        "mentions_name_gotra": body.str.contains(r"(name|gotra|नाम|गोत्र)", case=False).astype(int),
    })


def build_copy_table():
    ads = pd.read_parquet(DATA / "ads_with_copy.parquet")
    ads = ads[ads["has_copy"]].copy()
    ads["copy_key"] = ads["copy_body"].str.strip()

    g = ads.groupby("copy_key")
    t = g.agg(
        first_seen=("first_date", "min"),
        n_ads=("ad_id", "count"),
        spend=("spend", "sum"),
        impressions=("impressions", "sum"),
        clicks=("link_clicks", "sum"),
        orders=("orders", "sum"),
        revenue=("revenue", "sum"),
        offering_type=("offering_type", lambda s: s.mode().iat[0] if len(s.mode()) else None),
        language=("language", lambda s: s.mode().iat[0] if len(s.mode()) else None),
        fmt=("format", lambda s: s.mode().iat[0] if len(s.mode()) else None),
        cta=("call_to_action_type", lambda s: s.mode().iat[0] if len(s.mode()) else None),
        title=("copy_title", "first"),
    ).reset_index()
    t = t[(t.impressions >= MIN_IMPR) & (t.spend >= MIN_SPEND)].copy()
    t["ctr"] = t.clicks / t.impressions
    t["roas"] = t.revenue / t.spend
    return t


def featurize(train, test):
    tfidf = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                            min_df=3, max_features=20000, sublinear_tf=True)
    Xtr_t = tfidf.fit_transform(train.copy_key)
    Xte_t = tfidf.transform(test.copy_key)
    svd = TruncatedSVD(n_components=min(60, Xtr_t.shape[1] - 1, len(train) - 1),
                       random_state=0)
    Ttr, Tte = svd.fit_transform(Xtr_t), svd.transform(Xte_t)

    def meta(df, cats):
        m = copy_features(df.copy_key)
        m["log_n_ads"] = np.log1p(df.n_ads.values)
        for col, levels in cats.items():
            for lv in levels:
                m[f"{col}={lv}"] = (df[col] == lv).astype(int).values
        return m.values

    cats = {c: train[c].dropna().unique().tolist()[:8]
            for c in ["offering_type", "language", "fmt", "cta"]}
    Mtr, Mte = meta(train, cats), meta(test, cats)
    return (Mtr, Mte), (np.hstack([Mtr, Ttr]), np.hstack([Mte, Tte]))


def evaluate(y_true, y_pred, w):
    rho = spearmanr(y_true, y_pred).statistic
    idx = np.arange(len(y_true))
    wins = tot = 0
    rng = np.random.default_rng(0)
    for _ in range(4000):
        i, j = rng.choice(idx, 2, replace=False)
        if abs(y_true[i] - y_true[j]) < 0.2 * max(abs(y_true[i]), abs(y_true[j]), 1e-9):
            continue
        tot += 1
        if (y_pred[i] > y_pred[j]) == (y_true[i] > y_true[j]):
            wins += 1
    q75_true = np.quantile(y_true, 0.75)
    top_pred = y_pred >= np.quantile(y_pred, 0.75)
    prec = (y_true[top_pred] >= q75_true).mean() if top_pred.sum() else np.nan
    return rho, wins / tot if tot else np.nan, prec


def run(t, target, cutoff):
    train, test = t[t.first_seen < cutoff], t[t.first_seen >= cutoff]
    ytr = train[target].rank(pct=True).values
    yte = test[target].values
    (Mtr, Mte), (Ftr, Fte) = featurize(train, test)
    print(f"\n=== target: {target} | train {len(train)} copies (<{cutoff.date()}), "
          f"test {len(test)} copies ===")
    for label, Xtr, Xte in [("metadata only", Mtr, Mte), ("metadata + text", Ftr, Fte)]:
        model = HistGradientBoostingRegressor(
            max_depth=3, max_iter=300, learning_rate=0.05,
            l2_regularization=1.0, random_state=0)
        model.fit(Xtr, ytr)
        pred = model.predict(Xte)
        rho, win, prec = evaluate(yte, pred, test.impressions.values)
        print(f"  {label:16s}: spearman {rho:+.3f} | pairwise win-rate {win:.1%} "
              f"| top-quartile precision {prec:.1%}")
    base = np.repeat(train[target].median(), len(test))
    rng = np.random.default_rng(1)
    rho, win, prec = evaluate(yte, rng.random(len(test)), None)
    print(f"  {'random baseline':16s}: spearman {rho:+.3f} | pairwise win-rate {win:.1%} "
          f"| top-quartile precision {prec:.1%}")


def main():
    t = build_copy_table()
    print(f"copy-level table: {len(t)} distinct copies with >= {MIN_IMPR} impr "
          f"and >= Rs.{MIN_SPEND} spend")
    print(f"  first_seen range {t.first_seen.min().date()} -> {t.first_seen.max().date()}")
    cutoff = t.first_seen.quantile(0.6).normalize()
    tr, te = (t.first_seen < cutoff).sum(), (t.first_seen >= cutoff).sum()
    print(f"  temporal split at {cutoff.date()} (60th pct): {tr} train / {te} test")
    for target in ["ctr", "roas"]:
        run(t, target, cutoff)


if __name__ == "__main__":
    main()
