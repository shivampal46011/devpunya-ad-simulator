"""DevPunya ad-copy simulator — Streamlit app.

Run: streamlit run app.py
"""
import json
from pathlib import Path

import pandas as pd
import streamlit as st

import simulator

ROOT = Path(__file__).parent
st.set_page_config(page_title="DevPunya ad simulator", page_icon="🪔",
                   layout="wide")

SEGMENT_SUMMARY = [
    ("Enemy & obstacle puja buyers", "26% of users · 51% of revenue", "fear-framing +77%, long copy +36%, urgency +30%; median ₹1,020"),
    ("English feed puja buyers", "13% of users · 20% of revenue", "urgency +56%; rejects Hindi script, social proof, prices; feed-only"),
    ("Chadhawa offer responders", "37% of users · 11% of revenue", "price mention +168%, social proof +74%; ceiling ~₹500"),
    ("Hindi reels devotees", "14% of users · 8% of revenue", "Hindi script +391%; repelled by every persuasion tactic"),
    ("Wealth & property seekers", "3% of users · 5% of revenue", "fear +46%, urgency +46%; short copy, video-led; surging in 2026"),
    ("Shani remedial seekers", "3% of users · 3% of revenue", "social proof +59%; wants exact remedy named; never mention price"),
    ("Temple loyalists", "4% of users · 2% of revenue", "plain short copy naming their temple; immune to tactics"),
]


@st.cache_resource
def get_model():
    return simulator.load_model()


@st.cache_resource
def get_personas():
    return simulator.load_personas()


st.title("DevPunya ad-copy simulator")
st.caption("Simulated Facebook/Instagram audience calibrated on your ad history "
           "(Dec 2024 – Jul 2026, 3,603 ads, 43k buyers). Validated: 65% "
           "pairwise accuracy on held-out ads with a real (≥2×) CTR gap.")

tab_judge, tab_audience, tab_method = st.tabs(
    ["Judge my copies", "Audience segments", "Methodology & honesty"])

with tab_judge:
    col_cfg, _ = st.columns([1, 3])
    with col_cfg:
        n = st.selectbox("How many copies to compare?", [1, 2, 3, 4, 5], index=1)
        offering = st.selectbox("Offering type", ["pooja", "chadhaava"],
                                format_func=lambda x: {"pooja": "Puja",
                                                       "chadhaava": "Chadhawa"}[x])
    cols = st.columns(min(n, 3))
    copies = []
    for i in range(n):
        with cols[i % len(cols)]:
            copies.append(st.text_area(
                f"Copy {chr(65 + i)}", height=220, key=f"copy_{i}",
                placeholder="Paste primary text here…"))

    if st.button("Run simulation", type="primary"):
        filled = [(f"Copy {chr(65 + i)}", c) for i, c in enumerate(copies)
                  if c and c.strip()]
        if not filled:
            st.warning("Paste at least one copy.")
        else:
            labels, texts = zip(*filled)
            results, verdicts = simulator.judge(list(texts), offering,
                                                list(labels))
            st.subheader("Ranking")
            rank_df = pd.DataFrame([{
                "rank": i + 1, "copy": r["label"],
                "combined score": r["combined"],
                "ML percentile (vs history)": r["ml_percentile"],
                "panel appeal (user-weighted)": round(r["panel_users"], 1),
                "panel appeal (revenue-weighted)": round(r["panel_revenue"], 1),
            } for i, r in enumerate(results)])
            st.dataframe(rank_df, hide_index=True, width="stretch")

            if len(results) > 1:
                st.subheader("Verdicts")
                for v in verdicts:
                    msg = f"**{v['a']} vs {v['b']}** — gap {v['gap']} pts: "
                    if v["call"] == "clear":
                        st.success(msg + f"{v['a']} is the clear pick.")
                    elif v["call"] == "lean":
                        st.info(msg + f"lean {v['a']}, but the video creative "
                                      "could flip it.")
                    else:
                        st.warning(msg + "too close to call from copy alone — "
                                         "let the video decide or A/B test both.")

            st.subheader("Per-persona reactions")
            for r in results:
                with st.expander(f"{r['label']} — combined {r['combined']}"):
                    a = r["attrs"]
                    fired = [k.replace("_", " ") for k, v in a.items() if v]
                    st.write("Attributes detected: " +
                             (", ".join(fired) if fired else "none"))
                    pdf = r["personas"][["name", "segment", "appeal"]] \
                        .sort_values("appeal", ascending=False)
                    st.bar_chart(pdf.set_index("name")["appeal"],
                                 horizontal=True, height=360)

with tab_audience:
    st.subheader("The 7 discovered segments (unsupervised, 43,160 buyers)")
    for name, size, traits in SEGMENT_SUMMARY:
        c1, c2, c3 = st.columns([2, 2, 4])
        c1.markdown(f"**{name}**")
        c2.caption(size)
        c3.caption(traits)
    st.divider()
    st.subheader("The 13 personas")
    for p in get_personas():
        with st.expander(f"{p['name']} — {p['segment']}"):
            st.write(p["profile"])
            st.caption(f"Price band: {p['price_band']}")
            st.caption(f"Behavior: {p['behavior']}")
            st.json(p["creative_lifts"])

with tab_method:
    st.markdown("""
### How the score works
- **ML percentile** — gradient-boosted model over character n-grams +
  structural copy features + offering metadata, trained on every historical
  copy's CTR rank. Backtested honestly with a temporal split.
- **Panel appeal** — 13 personas whose response profiles come from measured
  segment behavior (creative-attribute lifts of 43k real buyers). Each persona
  scores the copy; votes are weighted by user share or revenue share.
- **Combined** — 50/50 blend, ranked. Gaps under 3 points are declared
  *too close to call* on purpose.

### What validation showed (be honest with yourself)
- On held-out ads with a **real CTR gap (≥2×)**: panel 65%, ML 60% accuracy.
- On small gaps: chance. The same copy with different videos varies ±26% in
  CTR — **the video, not the text, decides close races.**
- ROAS is only weakly predictable from copy (win-rate 58%); conversion depends
  on offer, price and festival timing.

### Rules of thumb the data supports
- Fear-framing + long copy sells pujas to the revenue core; never discounts.
- Explicit low price + crowd numbers sells chadhawa; keep it in the first line.
- Devanagari devotional copy owns the reels devotees; drop all sales tactics there.
- English buyers need urgency + legitimacy (temple, priests, video proof).
""")
