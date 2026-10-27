"""
DEPLOYMENT STEP 1 — TRAIN FINAL MODELS (Random Forest + XGBoost)
----------------------------------------------------------------------
Run once before launching the dashboard: pip install xgboost shap
(pickle is a Python built-in, no separate install needed)

Trains the two point-prediction models that made the final cut, and
saves everything app.py needs to load instantly (no retraining at
dashboard launch time):
  - random_forest.pkl, xgboost.pkl, feature_cols.pkl   (trained models)
  - test_predictions.csv                                (actual vs. both models' predictions, per district-year)
  - shap_global_importance.csv                          (for the dashboard's feature importance chart)
  - shap_values_test.pkl                                (per-row SHAP values, for per-district explanations)

All paths are anchored to THIS script's own folder (BASE_DIR below), so
it works the same whether run locally, in Colab, or as part of a
Streamlit Cloud deployment — no dependency on the current working
directory. Reads training data from ./data/ (a self-contained copy
inside this deployment folder — see README.md).
"""

import pandas as pd
import numpy as np
import pickle
import joblib  # bundled with scikit-learn, always available — used only
               # for its compression support when saving the Random Forest
               # (tree ensembles compress very well; this cuts file size
               # substantially so it fits GitHub's 25MB browser-upload limit)
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
import shap

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

def _find(names):
    """First existing file among `names`, looking next to this script and in ./data."""
    for folder in (BASE_DIR, DATA_DIR):
        for n in names:
            p = os.path.join(folder, n)
            if os.path.isfile(p):
                return p
    raise FileNotFoundError("Could not find any of: " + ", ".join(names))


# rice_dataset_final.csv is the same table as rice_model_ready_baseline.csv (checked row by row)
df = pd.read_csv(_find(["rice_model_ready_baseline.csv", "rice_dataset_final.csv"]))
TARGET = "RICE YIELD (Kg per ha)"

# Same collinearity drops used throughout this project (see 05_eda_plots_baseline.py)
COLLINEAR_DROPS = [
    "RAIN_JUN_SEP_mm", "ETo_Kharif_seasonal", "GDD_Kharif",
    "TEMP_JUN_SEP_C", "TEMP_OPTIMALITY", "N_TEMP_INTERACTION",
]
id_cols = ["Dist Code", "Dist Name", "State Name", "Year"]
drop_cols = id_cols + [TARGET, "RICE AREA (1000 ha)"] + COLLINEAR_DROPS
feature_cols = [c for c in df.columns if c not in drop_cols]

train_mask = df["Year"] <= 2005
test_mask = df["Year"] >= 2006
X_train, X_test = df.loc[train_mask, feature_cols], df.loc[test_mask, feature_cols]
y_train, y_test = df.loc[train_mask, TARGET], df.loc[test_mask, TARGET]

print(f"Train: {X_train.shape[0]} rows | Test: {X_test.shape[0]} rows | Features: {len(feature_cols)}")

# ---- Naive persistence, for the dashboard's comparison panel ----
naive_pred = df.loc[test_mask, "YIELD_LAG1"].values
naive_r2 = r2_score(y_test, naive_pred)

# ---- Random Forest (deployed baseline) ----
# Original hyperparameters, unchanged — joblib compression alone (below)
# takes this from 35MB to ~12MB with zero change to the model's actual
# predictions, so there's no need to shrink the model itself.
print("\nTraining Random Forest...")
rf = RandomForestRegressor(n_estimators=300, max_depth=12, min_samples_leaf=3,
                            random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
rf_r2 = r2_score(y_test, rf_pred)
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
print(f"Random Forest Test R²: {rf_r2:.4f} | RMSE: {rf_rmse:.2f}")

# ---- XGBoost (for SHAP explainability) ----
print("\nTraining XGBoost (with hyperparameter search)...")
param_grid = {"n_estimators": [200, 400], "max_depth": [3, 5, 7], "learning_rate": [0.01, 0.05, 0.1]}
tscv = TimeSeriesSplit(n_splits=5)
grid = GridSearchCV(XGBRegressor(random_state=42, n_jobs=-1, objective="reg:squarederror"),
                     param_grid, cv=tscv, scoring="r2", n_jobs=-1)
grid.fit(X_train, y_train)
xgb_model = grid.best_estimator_
xgb_pred = xgb_model.predict(X_test)
xgb_r2 = r2_score(y_test, xgb_pred)
xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_pred))
print(f"XGBoost best params: {grid.best_params_}")
print(f"XGBoost Test R²: {xgb_r2:.4f} | RMSE: {xgb_rmse:.2f}")

# ---- Save models ----
# Random Forest uses joblib with compression (compress=3 — good balance of
# size reduction vs. save/load speed; 9 is max compression but noticeably
# slower for little extra benefit on this model size). XGBoost and the
# feature list stay on plain pickle since they're already small and don't
# benefit much from compression.
rf_path = os.path.join(BASE_DIR, "random_forest.pkl")
joblib.dump(rf, rf_path, compress=3)
rf_size_mb = os.path.getsize(rf_path) / (1024 * 1024)
print(f"\nRandom Forest saved: {rf_size_mb:.1f} MB "
      f"({'under 25MB, should upload fine on GitHub' if rf_size_mb < 25 else 'STILL OVER 25MB — see note below'})")
with open(os.path.join(BASE_DIR, "xgboost.pkl"), "wb") as f:
    pickle.dump(xgb_model, f)
with open(os.path.join(BASE_DIR, "feature_cols.pkl"), "wb") as f:
    pickle.dump(feature_cols, f)

# ---- Save test predictions with district/year identifiers, for the dashboard's history chart ----
test_predictions = df.loc[test_mask, id_cols + [TARGET]].copy()
test_predictions["naive_pred"] = naive_pred
test_predictions["rf_pred"] = rf_pred
test_predictions["xgb_pred"] = xgb_pred
test_predictions.to_csv(os.path.join(BASE_DIR, "test_predictions.csv"), index=False)

# ---- SHAP (global + per-row, for the dashboard) ----
print("\nComputing SHAP values...")

# Defensive fix for a known shap/xgboost version-compatibility bug: newer
# XGBoost (2.x) stores base_score internally as a bracketed string like
# '[1.4633058E3]', which older shap versions fail to parse with
# "ValueError: could not convert string to float". This rewrites it to a
# plain float before handing the model to shap, so the script works
# regardless of exactly which shap/xgboost versions are installed.
# (The real fix is `pip install --upgrade shap` — this is just a fallback
# so the script doesn't hard-fail if that isn't done.)
import json
try:
    booster = xgb_model.get_booster()
    config = json.loads(booster.save_config())
    raw_base_score = config["learner"]["learner_model_param"]["base_score"]
    if isinstance(raw_base_score, str) and raw_base_score.strip().startswith("["):
        clean_value = raw_base_score.strip("[]")
        config["learner"]["learner_model_param"]["base_score"] = clean_value
        booster.load_config(json.dumps(config))
        print(f"  (patched base_score format: {raw_base_score} -> {clean_value})")
except Exception as e:
    print(f"  (base_score patch skipped: {e} — if TreeExplainer fails next, run "
          f"'pip install --upgrade shap')")

explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer(X_test)

global_importance = pd.Series(
    np.abs(shap_values.values).mean(axis=0), index=feature_cols
).sort_values(ascending=False)
global_importance.to_csv(os.path.join(BASE_DIR, "shap_global_importance.csv"), header=["mean_abs_shap"])

with open(os.path.join(BASE_DIR, "shap_values_test.pkl"), "wb") as f:
    pickle.dump({"shap_values": shap_values, "X_test": X_test.reset_index(drop=True),
                 "ids": df.loc[test_mask, id_cols].reset_index(drop=True)}, f)

# ---- Summary table for the dashboard / chatbot ----
# Scores are reported for the whole test window AND split in two. Analysis of the dataset showed that
# 2016-2025 is unusually smooth (yield wiggle around a straight line drops ~4x, climate columns look
# extrapolated), which makes "same as last year" almost perfect there and hides whether the models add
# value. 2006-2015 is the fairer comparison.
_years = df.loc[test_mask, "Year"].values


def _r2_between(lo, hi, pred):
    m = (_years >= lo) & (_years <= hi)
    return r2_score(y_test.values[m], np.asarray(pred)[m])


summary = pd.DataFrame([
    {"model": "Naive persistence", "test_r2": naive_r2, "test_rmse": np.sqrt(mean_squared_error(y_test, naive_pred)),
     "r2_2006_2015": _r2_between(2006, 2015, naive_pred), "r2_2016_2025": _r2_between(2016, 2025, naive_pred)},
    {"model": "Random Forest", "test_r2": rf_r2, "test_rmse": rf_rmse,
     "r2_2006_2015": _r2_between(2006, 2015, rf_pred), "r2_2016_2025": _r2_between(2016, 2025, rf_pred)},
    {"model": "XGBoost", "test_r2": xgb_r2, "test_rmse": xgb_rmse,
     "r2_2006_2015": _r2_between(2006, 2015, xgb_pred), "r2_2016_2025": _r2_between(2016, 2025, xgb_pred)},
])
summary.to_csv(os.path.join(BASE_DIR, "model_summary.csv"), index=False)
print("\n" + summary.round(3).to_string(index=False))

# ---- Production models: refit on ALL years, used for next-season forecasts ----
# The models above are trained on 1966-2005 so they can be evaluated honestly on 2006-2025. For forecasting
# 2026 that is the wrong choice: a tree model can never predict above the highest yields it saw in training,
# so a 1966-2005 model under-predicts today's much higher yields (in testing: Patna 4,163 vs 4,915 kg/ha).
# Standard practice: evaluate on a holdout, then refit on everything for production. These are used by the
# chatbot and the what-if panel. They cannot be scored (there is no holdout left) — that is expected.
print("\nRefitting production models on all years (for next-season forecasts)...")
X_all, y_all = df[feature_cols], df[TARGET]
rf_prod = RandomForestRegressor(n_estimators=300, max_depth=12, min_samples_leaf=3,
                                random_state=42, n_jobs=-1).fit(X_all, y_all)
xgb_prod = XGBRegressor(**grid.best_params_, random_state=42, n_jobs=-1,
                        objective="reg:squarederror").fit(X_all, y_all)
rf_prod_path = os.path.join(BASE_DIR, "random_forest_full.pkl")
joblib.dump(rf_prod, rf_prod_path, compress=3)
with open(os.path.join(BASE_DIR, "xgboost_full.pkl"), "wb") as f:
    pickle.dump(xgb_prod, f)
for _f in ("random_forest.pkl", "random_forest_full.pkl", "xgboost.pkl", "xgboost_full.pkl"):
    _mb = os.path.getsize(os.path.join(BASE_DIR, _f)) / (1024 * 1024)
    print(f"  {_f:24s} {_mb:6.1f} MB  {'OK for GitHub browser upload' if _mb < 25 else 'OVER 25MB - use git push or Git LFS'}")

print(f"""
Saved to {BASE_DIR}:
  random_forest.pkl, xgboost.pkl            (evaluation models: trained 1966-2005)
  random_forest_full.pkl, xgboost_full.pkl  (production models: refit on all years - used for forecasts)
  feature_cols.pkl, test_predictions.csv, model_summary.csv
  shap_global_importance.csv, shap_values_test.pkl

Next: run 02_train_state_space_model.py (mandatory), then launch app.py
""")
