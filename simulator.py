"""Core scoring engine: attribute extraction, deterministic persona panel,
and the trained ML model. Used by the Streamlit app and CLI.
"""
import json
import math
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "data" / "model.joblib"

PATTERNS = {
    "urgency": re.compile(r"(limited|last chance|don.t miss|today|tonight|hurry|only \d|rare|अंतिम|आज|सिर्फ|jaldi)", re.I),
    "social_proof": re.compile(r"(devotees|families|celebrit|vip|builders|हज़ार|लाख|\d{2,}[,\d]*\s*(log|people|devotee))", re.I),
    "price_mention": re.compile(r"(₹|rs\.?\s?\d|/-)", re.I),
    "fear_frame": re.compile(r"(black magic|negative energ|shatru|enem|dosh|kaal|संकट|शत्रु|काला जादू|भय|dushman|obstacle|blockage)", re.I),
    "benefit_frame": re.compile(r"(blessing|prosperit|wealth|success|growth|समृद्धि|सफलता|धन|विजय|victory|peace|shanti)", re.I),
}


def extract_attributes(text):
    """Copy attributes. hindi_script uses the ACTUAL script of the text
    (fix from panel validation: ad-name language labels were unreliable)."""
    text = text or ""
    dev_share = len(re.findall(r"[ऀ-ॿ]", text)) / max(len(text), 1)
    attrs = {k: bool(p.search(text)) for k, p in PATTERNS.items()}
    attrs["hindi_script"] = dev_share > 0.3
    attrs["long_copy"] = len(text) > 700
    return attrs


def load_personas():
    return json.load(open(ROOT / "personas.json"))["personas"]


def persona_appeal(persona, attrs):
    """0-100 appeal: logistic over the sum of measured lifts whose
    attributes are present in the copy."""
    s = sum(lift / 100 for a, lift in persona.get("creative_lifts", {}).items()
            if attrs.get(a))
    return round(100 / (1 + math.exp(-1.2 * s)), 1)


def panel_scores(text, personas=None):
    personas = personas or load_personas()
    attrs = extract_attributes(text)
    rows = []
    for p in personas:
        rows.append({"id": p["id"], "name": p["name"], "segment": p["segment"],
                     "weight_users": p["weight_users"],
                     "weight_revenue": p["weight_revenue"],
                     "appeal": persona_appeal(p, attrs)})
    df = pd.DataFrame(rows)
    return {
        "attrs": attrs,
        "personas": df,
        "panel_users": float(np.average(df.appeal, weights=df.weight_users)),
        "panel_revenue": float(np.average(df.appeal, weights=df.weight_revenue)),
    }


def load_model():
    return joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None


def ml_percentile(bundle, text, offering_type="pooja"):
    """Predicted CTR percentile (0-100) vs historical copies."""
    from backtest import copy_features
    X_t = bundle["svd"].transform(bundle["tfidf"].transform([text]))
    m = copy_features(pd.Series([text]))
    m["log_n_ads"] = 0.0
    for col, levels in bundle["cats"].items():
        for lv in levels:
            m[f"{col}={lv}"] = 1.0 if (col == "offering_type" and lv == offering_type) else 0.0
    pred = bundle["model"].predict(np.hstack([m.values, X_t]))[0]
    return round(float(np.clip(pred, 0, 1)) * 100, 1)


def judge(copies, offering_type="pooja", labels=None):
    """Score a list of copies; returns per-copy scores + pairwise verdicts."""
    bundle = load_model()
    personas = load_personas()
    labels = labels or [f"Copy {chr(65 + i)}" for i in range(len(copies))]
    out = []
    for lab, c in zip(labels, copies):
        ps = panel_scores(c, personas)
        ml = ml_percentile(bundle, c, offering_type) if bundle else None
        combined = (0.5 * ps["panel_users"] + 0.5 * ml) if ml is not None \
            else ps["panel_users"]
        out.append({"label": lab, "copy": c, **ps, "ml_percentile": ml,
                    "combined": round(combined, 1)})
    out.sort(key=lambda r: -r["combined"])
    verdicts = []
    for i in range(len(out) - 1):
        a, b = out[i], out[i + 1]
        gap = a["combined"] - b["combined"]
        verdicts.append({
            "a": a["label"], "b": b["label"], "gap": round(gap, 1),
            "call": "clear" if gap >= 8 else
                    ("lean" if gap >= 3 else "too close to call"),
        })
    return out, verdicts
