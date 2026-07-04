"""Extract ad performance history from the devp-read replica into local parquet.

Outputs (in ./data):
  fb_ad_daily.parquet   - one row per ad x day, Meta insights flattened
  orders_attributed.parquet - one row per order with ad attribution + user segment fields
"""
import json
from pathlib import Path

import pandas as pd

import db

DATA = Path(__file__).parent / "data"
DATA.mkdir(exist_ok=True)


def first_action(payload, key, action_type=None):
    items = payload.get(key) or []
    for it in items:
        if action_type is None or it.get("action_type") == action_type:
            try:
                return float(it.get("value"))
            except (TypeError, ValueError):
                return None
    return None


def extract_fb_daily(con):
    rows = con.run("""
        SELECT id, ad_id, ad_name, campaign, ad_set, date, type, offering_id,
               account_id, spend, impressions, reach, link_click, unique_link_click,
               landing_page_view, add_to_cart, orders, revenue, ad_copy_link, json
        FROM fb_analytics
        WHERE "deletedAt" IS NULL
    """)
    cols = [c["name"] for c in con.columns]
    df = pd.DataFrame(rows, columns=cols)

    flat = []
    for payload in df["json"]:
        if isinstance(payload, str):
            try:
                payload = json.loads(payload)
            except (TypeError, ValueError):
                payload = None
        if not isinstance(payload, dict):
            flat.append({})
            continue
        flat.append({
            "objective": payload.get("objective"),
            "ctr_meta": payload.get("ctr"),
            "cpm": payload.get("cpm"),
            "cpc": payload.get("cpc"),
            "frequency": payload.get("frequency"),
            "clicks_all": payload.get("clicks"),
            "quality_ranking": payload.get("quality_ranking"),
            "engagement_rate_ranking": payload.get("engagement_rate_ranking"),
            "conversion_rate_ranking": payload.get("conversion_rate_ranking"),
            "video_plays": first_action(payload, "video_play_actions"),
            "video_30s": first_action(payload, "video_30_sec_watched_actions"),
            "video_p25": first_action(payload, "video_p25_watched_actions"),
            "video_p50": first_action(payload, "video_p50_watched_actions"),
            "video_p75": first_action(payload, "video_p75_watched_actions"),
            "video_p95": first_action(payload, "video_p95_watched_actions"),
            "video_p100": first_action(payload, "video_p100_watched_actions"),
            "video_avg_watch_s": first_action(payload, "video_avg_time_watched_actions"),
            "purchases_meta": first_action(payload, "actions", "purchase")
                              or first_action(payload, "actions", "omni_purchase"),
        })
    meta = pd.DataFrame(flat, index=df.index)
    df = pd.concat([df.drop(columns=["json"]), meta], axis=1)

    for c in ["spend", "revenue", "ctr_meta", "cpm", "cpc", "frequency", "clicks_all"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["date"] = pd.to_datetime(df["date"])
    df.to_parquet(DATA / "fb_ad_daily.parquet", index=False)
    return df


def extract_orders(con):
    rows = con.run("""
        SELECT id, order_id, user_id, created_date, order_type, total_amount,
               user_order_count, is_repeat_order, campaign, adname, ad_platform,
               placement, cityinfo, stateinfo, device, platform, signup_date,
               days_after_booking, wish, wish_category::text AS wish_category
        FROM analytics
        WHERE "deletedAt" IS NULL
    """)
    cols = [c["name"] for c in con.columns]
    df = pd.DataFrame(rows, columns=cols)
    df["created_date"] = pd.to_datetime(df["created_date"])
    df.to_parquet(DATA / "orders_attributed.parquet", index=False)
    return df


def main():
    con = db.connect("devp")
    try:
        fb = extract_fb_daily(con)
        print(f"fb_ad_daily: {len(fb):,} rows, {fb.ad_id.nunique():,} ads, "
              f"{fb.date.min().date()} -> {fb.date.max().date()}")
        orders = extract_orders(con)
        print(f"orders_attributed: {len(orders):,} rows, "
              f"{orders.created_date.min().date()} -> {orders.created_date.max().date()}")
    finally:
        con.close()


if __name__ == "__main__":
    main()
