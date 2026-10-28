"""Backtest: what if kitchens had cooked the AI-recommended quantity on unseen future data?"""
import numpy as np
import pandas as pd
import config as C
from features import add_features
from engine import load


def per_row(cost=C.DEFAULT_COST_PER_KG, shortage=C.DEFAULT_SHORTAGE_COST_PER_KG) -> pd.DataFrame:
    """Held-out (last 20%) rows with simulated outcomes for current practice vs the AI plan."""
    df = add_features(pd.read_csv(C.DATA_PATH, parse_dates=["date"]).sort_values("date").reset_index(drop=True))
    test = df.iloc[int(len(df) * 0.8):].copy()
    _, dm = load()
    crit = shortage / (shortage + cost)
    keys = sorted(dm)
    preds = np.maximum.accumulate(
        np.column_stack([dm[k].predict(test[C.DEMAND_FEATURES]) for k in keys]), axis=1)
    rec = np.array([np.interp(crit, keys, row) for row in preds])
    demand = test["food_demand_kg"].values

    def outcome(prepared):
        consumed = np.minimum(prepared, demand)
        waste = prepared - consumed + consumed * 0.045
        short = np.maximum(demand - prepared, 0)
        return waste, short

    w0, s0 = outcome(test["food_prepared_kg"].values)
    w1, s1 = outcome(rec)
    test["recommended_kg"] = rec
    test["waste_current"], test["waste_ai"] = w0, w1
    test["short_current"], test["short_ai"] = s0, s1
    return test


def run(cost=C.DEFAULT_COST_PER_KG, shortage=C.DEFAULT_SHORTAGE_COST_PER_KG):
    t = per_row(cost, shortage)
    w0, s0, w1, s1 = t["waste_current"].values, t["short_current"].values, t["waste_ai"].values, t["short_ai"].values
    out = dict(
        waste_current_kg=w0.sum(), waste_ai_kg=w1.sum(),
        waste_reduction_pct=100 * (1 - w1.sum() / w0.sum()),
        shortage_days_current_pct=100 * (s0 > 0.5).mean(), shortage_days_ai_pct=100 * (s1 > 0.5).mean(),
        total_cost_current=w0.sum() * cost + s0.sum() * shortage,
        total_cost_ai=w1.sum() * cost + s1.sum() * shortage,
    )
    out["cost_saving_pct"] = 100 * (1 - out["total_cost_ai"] / out["total_cost_current"])
    return out


if __name__ == "__main__":
    for k, v in run().items():
        print(f"{k:28s} {v:12.2f}")