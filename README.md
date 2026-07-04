# Ad-copy audience simulator — data foundation

Goal: a calibrated simulator of the DevPunya Facebook/Instagram audience that
ranks ad copies, validated by backtesting against real ad history.

## Data sources
- Postgres read replica `devp-read.czyaggyi6bp9.ap-south-1.rds.amazonaws.com`
  (database `devp`, credentials from Secrets Manager `rdsDevProd`; the helper
  forces read-only). Never point scripts at the primary `devp` host.
- Meta Ads MCP connector (session-level) for creative text harvest, accounts:
  - 1076498786942935 — Devpunya Puja | India
  - 176372785467093 — Devpunya Chadhawa | India
  - 1171723244253332 — Devpunya Puja | USA

## Pipeline
1. `extract.py` — pulls `fb_analytics` (ad x day insights incl. video funnel,
   quality rankings) and `analytics` (order-level attribution + geo/platform)
   into `data/fb_ad_daily.parquet` and `data/orders_attributed.parquet`.
2. `build_dataset.py` — rolls up to one row per ad (`data/ads_master.parquet`)
   with lifetime KPIs (CTR, CPM, CVR, ROAS, hook/hold/completion rates) and
   attributes parsed from ad names (theme, language, format, version, creator).
   Flags `reliable` = >=3k impressions and >=Rs.1k spend.
3. Copy harvest (Meta MCP): `save_map_chunk.py` accumulates
   `data/ad_creative_map.jsonl` (ad_id -> creative_id);
   `pending_batches.py` + `save_creatives_chunk.py` accumulate
   `data/creatives.jsonl` (body/primary text, title/headline, CTA, video id).

## Dataset facts (as of 2026-07-04)
- fb_analytics: 2024-12-22 -> 2026-07-03, 26,991 ad-days, 3,603 ads,
  Rs.36.4M spend, 28,281 orders, Rs.22.9M tracked revenue.
- 1,223 ads pass the reliability bar -> modeling base.
- orders_attributed: 75,084 ad-attributed orders back to 2024-04-28 with
  state, placement (Reels dominant), platform, repeat flag, AOV
  (puja median Rs.991 vs chadhawa Rs.151).
- Copy harvest (2026-07-04): 100% of the 3,603 ads mapped to a creative;
  84% have copy text (95% of reliable ads). 579 distinct copy bodies across
  3,024 ads — heavy reuse, so each copy has performance from multiple ads.
  `data/ads_with_copy.parquet` is the joined modeling table (via join_copy.py).
- Known gaps: `account_id` null in fb_analytics before Dec 2025; ~16% of ads
  (mostly Aug-Oct 2025 vintage) have no recoverable body text — catalog/
  dynamic creatives and old boosted posts where Meta returns no body;
  offering types `puja`/`chadhawa` rows carry 0 orders (order tracking lived
  on the duplicate `pooja`/`chadhaava` rows).

## Phase 1 backtest result (backtest.py, 2026-07-04)
Unit = distinct copy body (230 pass exposure bar), temporal split 2026-03-25
(137 train / 93 test), HistGradientBoosting on TF-IDF SVD + structural +
metadata features.
- CTR: spearman +0.35, pairwise win-rate 67%, top-quartile precision 54%
  (random: 37% win-rate, 21% precision). Real, significant signal (p<0.001).
  Metadata-only ties metadata+text — category/format/language carry most of it.
- ROAS: spearman +0.18, win-rate 58% — weak from copy alone; conversion is
  dominated by offer, price point, and festival timing.
Conclusion: simulator should rank on CTR/hook appeal from copy, and layer
offering/seasonality priors for conversion. Sets the bar Phase 2 must beat.

## Phase 4 persona panel validation (2026-07-04)
13 personas (personas.json) built from measured segment stats; 5 LLM judge
agents voted on 50 blind same-offering-type test pairs (make_pairs.py,
score_panel.py, data/panel_votes_*.json).
- Headline: panel 56%, ML 42% — same-category pairs are much harder than the
  Phase 1 mixed-category test (there the ML's 67% was largely category signal).
- Diagnosis: same copy with different videos varies +/-26% in CTR (CV 0.26),
  so pairs with <2x CTR gap are video/auction noise — both methods score
  chance there (panel 50%, ML 30%, n=30).
- On pairs with a REAL gap (>=2x, n=20): panel 65%, ML 60%. The simulator
  can call meaningful differences; it cannot call coin-flips, by design.
- Best personas: P3 whale 68%, P6 chadhawa 60%; worst: P11 wealth/property
  34% (newest segment, thinnest history — needs recalibration).
- Product implication: output three-way verdicts (A / B / too-close-to-call)
  instead of forcing a winner on small predicted gaps.

## Next phases
- Scale validation to 200+ pairs for tighter confidence intervals; proper
  train/val split before reweighting or pruning personas.
- Fix language feature (parse actual script of copy, not ad-name token).
- Ensemble + tool (paste copies -> ranked verdicts + persona reactions).
- Later: vision layer for video creative (the dominant unexplained factor).
