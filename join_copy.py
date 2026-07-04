"""Join harvested creative text onto ads_master -> data/ads_with_copy.parquet."""
import json
from pathlib import Path

import pandas as pd

DATA = Path(__file__).parent / "data"


def read_jsonl(path):
    with open(path) as f:
        return pd.DataFrame(json.loads(l) for l in f)


def main():
    ads = pd.read_parquet(DATA / "ads_master.parquet")
    amap = read_jsonl(DATA / "ad_creative_map.jsonl")
    cre = read_jsonl(DATA / "creatives.jsonl").rename(columns={
        "id": "creative_id", "name": "creative_name", "body": "copy_body",
        "title": "copy_title", "status": "creative_status"})

    amap = amap.drop_duplicates("ad_id")[["ad_id", "creative_id", "created_time"]]
    ads = ads.merge(amap, on="ad_id", how="left")
    ads = ads.merge(
        cre[["creative_id", "copy_body", "copy_title", "call_to_action_type",
             "object_type", "creative_name", "creative_status"]],
        on="creative_id", how="left")

    ads["has_copy"] = ads["copy_body"].notna() & (ads["copy_body"].str.strip() != "")
    ads["copy_len"] = ads["copy_body"].str.len()
    ads.to_parquet(DATA / "ads_with_copy.parquet", index=False)

    rel = ads[ads["reliable"]]
    print(f"ads total {len(ads):,} | mapped to creative: {ads.creative_id.notna().mean():.0%} "
          f"| with copy text: {ads.has_copy.mean():.0%}")
    print(f"reliable ads {len(rel):,} | mapped: {rel.creative_id.notna().mean():.0%} "
          f"| with copy: {rel.has_copy.mean():.0%}")
    wc = ads[ads.has_copy]
    print(f"distinct copy bodies: {wc.copy_body.nunique():,} across {len(wc):,} ads")
    print(f"copy length: median {wc.copy_len.median():.0f} chars, "
          f"p10 {wc.copy_len.quantile(.1):.0f}, p90 {wc.copy_len.quantile(.9):.0f}")
    cov = (ads.assign(year_month=ads.first_date.dt.to_period('M').astype(str))
           .groupby('year_month').agg(ads=('ad_id', 'count'), with_copy=('has_copy', 'sum')))
    cov['pct'] = (cov.with_copy / cov.ads * 100).round(0).astype(int)
    print("\ncopy coverage by ad first-seen month:")
    print(cov.to_string())


if __name__ == "__main__":
    main()
