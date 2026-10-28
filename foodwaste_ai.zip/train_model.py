"""Train, compare and save the models."""
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import config as C
from features import add_features

QUANTILES = [0.10, 0.50, 0.80, 0.90, 0.95]


def preprocessor(features):
    cat = [c for c in features if c in C.CATEGORICAL]
    num = [c for c in features if c not in C.CATEGORICAL]
    return ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
        ("num", StandardScaler(), num),
    ])


def candidate_models():
    return {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),
        "Random Forest": RandomForestRegressor(n_estimators=300, min_samples_leaf=3, n_jobs=-1, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=3, random_state=42),
        "Hist Gradient Boosting": HistGradientBoostingRegressor(max_iter=300, learning_rate=0.06, random_state=42),
    }


def main():
    C.MODEL_DIR.mkdir(exist_ok=True)
    if not C.DATA_PATH.exists():
        from generate_data import generate
        C.DATA_PATH.parent.mkdir(exist_ok=True)
        generate().to_csv(C.DATA_PATH, index=False)

    df = add_features(pd.read_csv(C.DATA_PATH, parse_dates=["date"]).sort_values("date").reset_index(drop=True))
    cut = int(len(df) * 0.8)                      # train on the PAST, test on the FUTURE
    train, test = df.iloc[:cut], df.iloc[cut:]
    X_tr, y_tr = train[C.WASTE_FEATURES], train["food_waste_kg"]
    X_te, y_te = test[C.WASTE_FEATURES], test["food_waste_kg"]
    tscv = TimeSeriesSplit(n_splits=5)

    results, fitted = [], {}
    for name, est in candidate_models().items():
        pipe = Pipeline([("prep", preprocessor(C.WASTE_FEATURES)), ("model", est)])
        cv_mae = -cross_val_score(pipe, X_tr, y_tr, cv=tscv, scoring="neg_mean_absolute_error").mean()
        pipe.fit(X_tr, y_tr)
        pred = np.clip(pipe.predict(X_te), 0, None)
        results.append(dict(
            model=name, cv_mae=round(cv_mae, 3),
            test_mae=round(mean_absolute_error(y_te, pred), 3),
            test_rmse=round(float(np.sqrt(mean_squared_error(y_te, pred))), 3),
            test_r2=round(r2_score(y_te, pred), 4),
        ))
        fitted[name] = (pipe, pred)
        print(f"{name:24s} CV-MAE={cv_mae:6.3f}  Test-MAE={results[-1]['test_mae']:6.3f}  R2={results[-1]['test_r2']:.3f}")

    best_name = min(results, key=lambda r: r["cv_mae"])["model"]
    best, best_pred = fitted[best_name]
    print(f"\nBest model (lowest CV MAE): {best_name}")

    # Refit the winner on ALL data for deployment
    deploy = Pipeline([("prep", preprocessor(C.WASTE_FEATURES)), ("model", candidate_models()[best_name])])
    deploy.fit(df[C.WASTE_FEATURES], df["food_waste_kg"])

    baseline_mae = mean_absolute_error(y_te, np.full(len(y_te), y_tr.mean()))

    # Permutation importance (on unseen future data)
    imp = permutation_importance(best, X_te, y_te, n_repeats=8, random_state=42,
                                 scoring="neg_mean_absolute_error", n_jobs=-1)
    importance = pd.DataFrame({"feature": C.WASTE_FEATURES, "importance": imp.importances_mean,
                               "std": imp.importances_std}).sort_values("importance", ascending=False)
    importance.to_csv(C.IMPORTANCE_PATH, index=False)

    # Demand quantile models (for the recommended quantity)
    dem_models = {}
    for q in QUANTILES:
        m = Pipeline([("prep", preprocessor(C.DEMAND_FEATURES)),
                      ("model", GradientBoostingRegressor(loss="quantile", alpha=q, n_estimators=250,
                                                          learning_rate=0.05, max_depth=3, random_state=42))])
        m.fit(train[C.DEMAND_FEATURES], train["food_demand_kg"])
        dem_models[q] = m
    lo, hi = dem_models[0.10].predict(test[C.DEMAND_FEATURES]), dem_models[0.90].predict(test[C.DEMAND_FEATURES])
    coverage = float(np.mean((test["food_demand_kg"] >= lo) & (test["food_demand_kg"] <= hi)))
    med_mae = mean_absolute_error(test["food_demand_kg"], dem_models[0.50].predict(test[C.DEMAND_FEATURES]))
    for q in QUANTILES:  # refit on all data for deployment
        dem_models[q].fit(df[C.DEMAND_FEATURES], df["food_demand_kg"])

    # Waste prediction interval from held-out residuals
    resid = (y_te.values - best_pred)
    interval = [float(np.quantile(resid, 0.10)), float(np.quantile(resid, 0.90))]

    joblib.dump(deploy, C.WASTE_MODEL_PATH)
    joblib.dump(dem_models, C.DEMAND_Q_PATH)
    pd.DataFrame({"date": test["date"].values, "actual": y_te.values, "predicted": best_pred}).to_csv(C.PREDICTIONS_PATH, index=False)

    json.dump(dict(best_model=best_name, comparison=results, baseline_mae_mean_predictor=round(baseline_mae, 3),
                   residual_interval_80=interval, demand_p10_p90_coverage=round(coverage, 3),
                   demand_median_mae=round(med_mae, 3), n_train=len(train), n_test=len(test),
                   avg_waste_kg=round(float(df["food_waste_kg"].mean()), 2)),
              open(C.METRICS_PATH, "w"), indent=2)
    print(f"Demand model: P10-P90 coverage on future data = {coverage:.1%} (target 80%)")
    print("Saved models to", C.MODEL_DIR)


if __name__ == "__main__":
    main()