"""Prediction + recommendation logic (used by the Streamlit app)."""
import json
import joblib
import numpy as np
import pandas as pd
import config as C
from features import add_features

_waste_model = None
_demand_models = None


def load():
    global _waste_model, _demand_models
    if _waste_model is None:
        _waste_model = joblib.load(C.WASTE_MODEL_PATH)
        _demand_models = joblib.load(C.DEMAND_Q_PATH)
    return _waste_model, _demand_models


def metrics():
    return json.load(open(C.METRICS_PATH))


def predict_waste(row: dict) -> float:
    wm, _ = load()
    return float(max(0.0, wm.predict(add_features(pd.DataFrame([row]))[C.WASTE_FEATURES])[0]))


def predict_waste_df(df: pd.DataFrame) -> np.ndarray:
    wm, _ = load()
    return np.clip(wm.predict(add_features(df)[C.WASTE_FEATURES]), 0, None)


def predict_demand(row: dict) -> dict:
    """Return demand quantiles in kg, e.g. {0.1: 80.2, 0.5: 95.0, ...}."""
    _, dm = load()
    X = pd.DataFrame([row])[C.DEMAND_FEATURES]
    q = {k: float(m.predict(X)[0]) for k, m in dm.items()}
    vals = np.maximum.accumulate(np.array([q[k] for k in sorted(q)]))  # no crossing quantiles
    return dict(zip(sorted(q), vals.tolist()))


def recommend(row: dict, cost_per_kg: float, shortage_cost_per_kg: float) -> dict:
    """Newsvendor rule: cook the demand quantile = Cs / (Cs + Cw)."""
    q = predict_demand(row)
    critical = shortage_cost_per_kg / (shortage_cost_per_kg + cost_per_kg)
    keys = sorted(q)
    recommended = float(np.interp(critical, keys, [q[k] for k in keys]))
    current = row["food_prepared_kg"]
    waste_now = predict_waste(row)
    waste_new = predict_waste({**row, "food_prepared_kg": recommended})
    saved = max(0.0, waste_now - waste_new)
    risk_now = float(np.interp(current, [q[k] for k in keys], keys, left=keys[0] * 0.5, right=0.99))
    return dict(demand_quantiles=q, critical_ratio=critical, recommended_kg=recommended,
                waste_current=waste_now, waste_recommended=waste_new, waste_saved_kg=saved,
                money_saved=saved * cost_per_kg, co2_saved_kg=saved * C.CO2E_PER_KG_WASTE,
                service_level_current=min(risk_now, 0.99))


def risk_level(waste_kg: float, prepared_kg: float) -> str:
    share = waste_kg / max(prepared_kg, 1e-6)
    return "LOW" if share < 0.10 else "MEDIUM" if share < 0.20 else "HIGH"