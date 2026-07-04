"""Build the modeling dataset: one row per ad with lifetime KPIs and
attributes parsed from ad naming conventions.

Input : data/fb_ad_daily.parquet
Output: data/ads_master.parquet + a printed data-quality report
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).parent / "data"

LANGUAGES = ["english", "hindi", "tamil", "telugu", "kannada", "marathi",
             "bengali", "gujarati", "punjabi", "malayalam", "hinglish"]
FORMATS = ["video", "image", "static", "carousel", "reel", "ugc"]


def parse_ad_name(name):
    """Best-effort parse of names like
    'Victoryincourtcases-Varahi_Ashaadnavratri puja-video-Vivek-English-v2'
    'Kaalejadusebachav| AshtBhairav | Kalashtami | Video | Hindi| V1 - Copy 2'
    """
    out = {"hook_theme": None, "language": None, "format": None,
           "version": None, "copy_variant": None, "creator": None}
    if not isinstance(name, str) or not name.strip():
        return out
    lower = name.lower()

    m = re.search(r"copy\s*(\d+)", lower)
    if m:
        out["copy_variant"] = int(m.group(1))
    m = re.search(r"\bv(?:ersion)?[\s\-_]?(\d+)\b", lower)
    if m:
        out["version"] = int(m.group(1))
    for lang in LANGUAGES:
        if re.search(rf"\b{lang}\b", lower):
            out["language"] = lang
            break
    for fmt in FORMATS:
        if re.search(rf"\b{fmt}\b", lower):
            out["format"] = fmt
            break

    tokens = [t.strip() for t in re.split(r"[|\-_]", name) if t.strip()]
    if tokens:
        out["hook_theme"] = tokens[0][:80]
    # creator = a token that is a single capitalized word and none of the known fields
    known = set(LANGUAGES) | set(FORMATS)
    for t in tokens[1:]:
        tl = t.lower()
        if (re.fullmatch(r"[A-Z][a-z]{2,15}", t) and tl not in known
                and not re.search(r"\d", t)):
            out["creator"] = t
    return out


def main():
    df = pd.read_parquet(DATA / "fb_ad_daily.parquet")
    df = df[df["ad_id"].notna() & (df["ad_id"] != "")]

    num = ["spend", "impressions", "reach", "link_click", "unique_link_click",
           "landing_page_view", "add_to_cart", "orders", "revenue",
           "video_plays", "video_30s", "video_p25", "video_p50", "video_p75",
           "video_p95", "video_p100"]
    for c in num:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    g = df.groupby("ad_id")
    ads = g.agg(
        ad_name=("ad_name", "last"),
        campaign=("campaign", "last"),
        ad_set=("ad_set", "last"),
        offering_type=("type", "last"),
        objective=("objective", "last"),
        account_id=("account_id", "last"),
        first_date=("date", "min"),
        last_date=("date", "max"),
        active_days=("date", "nunique"),
        spend=("spend", "sum"),
        impressions=("impressions", "sum"),
        reach=("reach", "max"),
        link_clicks=("link_click", "sum"),
        unique_link_clicks=("unique_link_click", "sum"),
        landing_page_views=("landing_page_view", "sum"),
        add_to_cart=("add_to_cart", "sum"),
        orders=("orders", "sum"),
        revenue=("revenue", "sum"),
        video_plays=("video_plays", "sum"),
        video_30s=("video_30s", "sum"),
        video_p25=("video_p25", "sum"),
        video_p50=("video_p50", "sum"),
        video_p75=("video_p75", "sum"),
        video_p100=("video_p100", "sum"),
        frequency_mean=("frequency", "mean"),
        quality_ranking=("quality_ranking", "last"),
        engagement_rate_ranking=("engagement_rate_ranking", "last"),
        conversion_rate_ranking=("conversion_rate_ranking", "last"),
        creative_link=("ad_copy_link", "last"),
    ).reset_index()

    imp = ads["impressions"].replace(0, np.nan)
    clicks = ads["link_clicks"].replace(0, np.nan)
    spend = ads["spend"].replace(0, np.nan)
    plays = ads["video_plays"].replace(0, np.nan)
    ads["ctr"] = ads["link_clicks"] / imp
    ads["cpm"] = ads["spend"] / imp * 1000
    ads["cpc"] = ads["spend"] / clicks
    ads["cvr_click_to_order"] = ads["orders"] / clicks
    ads["cpa"] = ads["spend"] / ads["orders"].replace(0, np.nan)
    ads["roas"] = ads["revenue"] / spend
    ads["hook_rate"] = ads["video_p25"] / plays
    ads["hold_rate"] = ads["video_p75"] / plays
    ads["completion_rate"] = ads["video_p100"] / plays

    parsed = pd.DataFrame([parse_ad_name(n) for n in ads["ad_name"]],
                          index=ads.index)
    ads = pd.concat([ads, parsed], axis=1)

    ads["spend_tier"] = pd.cut(
        ads["spend"], bins=[-1, 500, 2000, 10000, 50000, np.inf],
        labels=["<500", "500-2k", "2k-10k", "10k-50k", "50k+"])
    # ads with too little exposure to judge — keep but flag
    ads["reliable"] = (ads["impressions"] >= 3000) & (ads["spend"] >= 1000)

    ads.to_parquet(DATA / "ads_master.parquet", index=False)

    rel = ads[ads["reliable"]]
    print(f"ads_master: {len(ads):,} ads | reliable (>=3k impr & >=Rs.1k spend): {len(rel):,}")
    print(f"date span : {ads.first_date.min().date()} -> {ads.last_date.max().date()}")
    print(f"total     : spend Rs.{ads.spend.sum():,.0f} | orders {ads.orders.sum():,.0f} "
          f"| revenue Rs.{ads.revenue.sum():,.0f}")
    print("\n-- parse coverage (share of ads with field extracted) --")
    for c in ["hook_theme", "language", "format", "version", "copy_variant", "creator"]:
        print(f"  {c:13s}: {ads[c].notna().mean():.0%}")
    print("\n-- reliable ads: KPI distribution --")
    print(rel[["ctr", "cpm", "cvr_click_to_order", "roas", "hook_rate", "hold_rate"]]
          .describe(percentiles=[.25, .5, .75, .9]).round(4).to_string())
    print("\n-- offering type breakdown --")
    print(ads.groupby("offering_type", dropna=False)
          .agg(ads=("ad_id", "count"), spend=("spend", "sum"), orders=("orders", "sum"))
          .sort_values("spend", ascending=False).head(10).to_string())
    print("\n-- language breakdown (reliable ads) --")
    print(rel.groupby("language", dropna=False)
          .agg(ads=("ad_id", "count"), median_roas=("roas", "median"))
          .sort_values("ads", ascending=False).to_string())


if __name__ == "__main__":
    main()
