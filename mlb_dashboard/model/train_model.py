"""
model/train_model.py
====================
Train the XGBoost game-prediction model and save it to model/model.pkl.

Improvements over baseline:
  - 27 features (Elo ratings, park factors, Pythagorean win%, 5-game window, differential features)
  - Early stopping to auto-select n_estimators
  - Isotonic calibration for better win-probability estimates
  - AUC-ROC as primary CV metric
  - TimeSeriesSplit CV so future games never train on past predictions
  - Temporal train/test split (first 80% / last 20% chronologically)
  - Separate xgb_for_shap model saved alongside calibrated model

Data priority:
  1. MLB Stats API  — official, free, never rate-limited (2019–2024)
  2. pybaseball     — Baseball Reference scraping (may be blocked)
  3. Synthetic data — last resort so the model always builds

Usage:
    python model/train_model.py
"""

import os
import pickle
import sys
import time

# Force UTF-8 output so Unicode symbols don't crash on Windows cp1252
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBClassifier

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data.fetch_data import fetch_mlb_games, fetch_pitching_stats, fetch_schedule, TEAM_ID_MAP
from data.feature_eng import (
    FEATURE_COLS, TEAM_CODES,
    build_sp_lookup, build_training_dataset, build_training_dataset_mlb,
)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")

# Shared hyperparameters — tuned for regularisation on ~10k game datasets
PARAMS = dict(
    max_depth=4,
    learning_rate=0.02,
    subsample=0.8,
    colsample_bytree=0.7,
    min_child_weight=8,
    gamma=0.15,
    reg_lambda=1.5,
    reg_alpha=0.1,
    random_state=42,
    eval_metric="logloss",
    verbosity=0,
)

try:
    from pybaseball import cache
    cache.enable()
    print("✓  pybaseball cache enabled")
except Exception:
    pass

# ── 1. Try MLB Stats API for 2019–2024 (most reliable) ───────────────────────

YEARS = list(range(2019, 2027))
print(f"\n── Phase 1: MLB Stats API ({YEARS[0]}–{YEARS[-1]}) ──────────────────────")

mlb_frames = []
for year in YEARS:
    try:
        df = fetch_mlb_games(year)
        if not df.empty:
            df["year"] = year
            mlb_frames.append(df)
            print(f"  ✓  {year}: {len(df):,} games")
        else:
            print(f"  ✗  {year}: no data returned")
    except Exception as e:
        print(f"  ✗  {year}: {e}")

if mlb_frames:
    all_games = pd.concat(mlb_frames, ignore_index=True)

    print(f"\nFetching SP stats for ERA/WHIP lookup ({YEARS[0]}–{YEARS[-1]})...")
    pitching_by_year: dict[int, dict] = {}
    for year in YEARS:
        try:
            pdf = fetch_pitching_stats(year)
            pitching_by_year[year] = build_sp_lookup(pdf)
            print(f"  ✓  {year}: {len(pitching_by_year[year])} pitchers indexed")
        except Exception as e:
            pitching_by_year[year] = {}
            print(f"  ✗  {year}: {e}")

    print(f"\nBuilding features from {len(all_games):,} games...")
    try:
        train_df = build_training_dataset_mlb(all_games, pitching_by_year=pitching_by_year)
        print(f"  ✓  Feature rows: {len(train_df):,}")
    except Exception as e:
        print(f"  ✗  Feature engineering failed: {e}")
        train_df = pd.DataFrame()
else:
    train_df = pd.DataFrame()

# ── 2. Fallback: pybaseball / Baseball Reference ─────────────────────────────

if train_df.empty or len(train_df) < 200:
    print("\n── Phase 2: pybaseball fallback (2024 only) ────────────────────────")
    print("(3 s pause between requests to avoid Baseball Reference blocks)\n")

    schedules = {}
    for i, code in enumerate(TEAM_CODES):
        try:
            schedules[code] = fetch_schedule(2024, code)
            print(f"  ✓  {code}  ({len(schedules[code])} games)")
        except Exception:
            print(f"  ✗  {code}: skipped")
        if i < len(TEAM_CODES) - 1:
            time.sleep(3)

    if schedules:
        try:
            pitching_df = fetch_pitching_stats(2024)
        except Exception:
            pitching_df = pd.DataFrame()
        train_df = build_training_dataset(schedules, pitching_df)

# ── 3. Last resort: synthetic data ───────────────────────────────────────────

if train_df.empty or len(train_df) < 200:
    print("\n⚠️  No real data available. Generating synthetic training set.")
    print("    Re-run after a few minutes or check your internet connection.\n")

    np.random.seed(42)
    n = 5000

    home_win_pct_5  = np.random.beta(5, 5, n)
    home_win_pct_10 = np.random.beta(5, 5, n)
    home_win_pct_20 = np.random.beta(5, 5, n)
    away_win_pct_5  = np.random.beta(5, 5, n)
    away_win_pct_10 = np.random.beta(5, 5, n)
    away_win_pct_20 = np.random.beta(5, 5, n)
    home_run_diff   = np.random.normal(0, 1.5, n)
    away_run_diff   = np.random.normal(0, 1.5, n)
    # Pythagorean win% — correlated with actual win% but noisier (run-differential based)
    home_pyth_wp    = (home_win_pct_10 * 0.9 + np.random.normal(0, 0.05, n)).clip(0.1, 0.9)
    away_pyth_wp    = (away_win_pct_10 * 0.9 + np.random.normal(0, 0.05, n)).clip(0.1, 0.9)
    home_streak     = np.random.randint(-7, 8, n).astype(float)
    away_streak     = np.random.randint(-7, 8, n).astype(float)
    home_rest       = np.random.choice([1, 1, 1, 2, 3, 4], n).astype(float)
    away_rest       = np.random.choice([1, 1, 1, 2, 3, 4], n).astype(float)
    h2h_win_pct     = np.random.beta(3, 3, n)
    home_sp_era     = np.random.normal(4.0, 0.8, n).clip(2.0, 7.0)
    away_sp_era     = np.random.normal(4.0, 0.8, n).clip(2.0, 7.0)
    home_sp_whip    = np.random.normal(1.25, 0.2, n).clip(0.8, 2.0)
    away_sp_whip    = np.random.normal(1.25, 0.2, n).clip(0.8, 2.0)
    # Elo — team quality tracker; correlated with win% but with independent noise
    home_elo        = 1500 + 200 * (home_win_pct_10 - 0.5) + np.random.normal(0, 30, n)
    away_elo        = 1500 + 200 * (away_win_pct_10 - 0.5) + np.random.normal(0, 30, n)
    elo_diff        = home_elo - away_elo
    # Park factors — sampled from the actual MLB distribution
    park_factor     = np.random.choice(
        [0.92, 0.93, 0.94, 0.95, 0.97, 0.98, 0.99, 1.00, 1.01, 1.02, 1.04, 1.13], n
    )

    logit = (
        0.15
        + 1.5  * (home_win_pct_10 - away_win_pct_10)
        + 0.8  * (home_win_pct_5  - away_win_pct_5)
        + 0.4  * (home_win_pct_20 - away_win_pct_20)
        + 0.3  * (home_run_diff - away_run_diff)
        + 0.5  * (home_pyth_wp - away_pyth_wp)
        + 0.05 * (home_streak - away_streak)
        - 0.02 * (home_rest - away_rest)
        - 0.2  * (home_sp_era - away_sp_era)
        - 0.15 * (home_sp_whip - away_sp_whip)
        + 0.4  * (h2h_win_pct - 0.5)
        + 0.008 * elo_diff
        + 0.3  * (park_factor - 1.0)
        + np.random.normal(0, 0.5, n)
    )
    prob = 1 / (1 + np.exp(-logit))
    labels = (np.random.uniform(size=n) < prob).astype(int)

    train_df = pd.DataFrame({
        "home_win_pct_5":   home_win_pct_5,
        "home_win_pct_10":  home_win_pct_10,
        "home_win_pct_20":  home_win_pct_20,
        "away_win_pct_5":   away_win_pct_5,
        "away_win_pct_10":  away_win_pct_10,
        "away_win_pct_20":  away_win_pct_20,
        "home_run_diff_15": home_run_diff,
        "away_run_diff_15": away_run_diff,
        "home_pyth_wp":     home_pyth_wp,
        "away_pyth_wp":     away_pyth_wp,
        "home_streak":      home_streak,
        "away_streak":      away_streak,
        "home_rest_days":   home_rest,
        "away_rest_days":   away_rest,
        "h2h_win_pct":      h2h_win_pct,
        "home_sp_era":      home_sp_era,
        "home_sp_whip":     home_sp_whip,
        "away_sp_era":      away_sp_era,
        "away_sp_whip":     away_sp_whip,
        "home_elo":         home_elo,
        "away_elo":         away_elo,
        "elo_diff":         elo_diff,
        "park_factor":      park_factor,
        "win_pct_diff_10":  home_win_pct_10 - away_win_pct_10,
        "run_diff_diff":    home_run_diff - away_run_diff,
        "streak_diff":      home_streak - away_streak,
        "era_diff":         away_sp_era - home_sp_era,
        "label":            labels,
    })

# ── 4. Validate dataset ───────────────────────────────────────────────────────

print(f"\n  Dataset shape  : {train_df.shape}")

if train_df.empty or len(train_df) < 50:
    print("\n❌  Not enough data to train. Try again in a few minutes.")
    sys.exit(1)

print(f"  Class balance  : {train_df['label'].value_counts().to_dict()}")

X = train_df[FEATURE_COLS]
y = train_df["label"]

# Temporal split: train on earliest 80%, evaluate on most recent 20%.
# Games in train_df are in chronological order from build_training_dataset_mlb.
split_idx = int(len(X) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
print(f"  Train: {len(X_train):,} games  |  Test (held-out): {len(X_test):,} games")

# ── 5. Cross-validation (diagnostic — AUC-ROC) ────────────────────────────────

print("\nRunning 5-fold time-series cross-validation (AUC-ROC)...")
cv = TimeSeriesSplit(n_splits=5)
cv_scores = []

for fold, (tr_idx, val_idx) in enumerate(cv.split(X_train), 1):
    Xtr, Xval = X_train.iloc[tr_idx], X_train.iloc[val_idx]
    ytr, yval = y_train.iloc[tr_idx], y_train.iloc[val_idx]

    m = XGBClassifier(n_estimators=500, **PARAMS)
    m.fit(Xtr, ytr, eval_set=[(Xval, yval)], verbose=False)
    score = roc_auc_score(yval, m.predict_proba(Xval)[:, 1])
    cv_scores.append(score)
    print(f"  Fold {fold}: AUC = {score:.4f}")

print(f"  CV mean AUC: {np.mean(cv_scores):.4f}  ±  {np.std(cv_scores):.4f}")

# ── 6. Find optimal n_estimators via early stopping ──────────────────────────

print("\nFinding optimal n_estimators via early stopping (temporal)...")
es_split = int(len(X_train) * 0.85)
X_tr, X_es = X_train.iloc[:es_split], X_train.iloc[es_split:]
y_tr, y_es = y_train.iloc[:es_split], y_train.iloc[es_split:]
pilot = XGBClassifier(n_estimators=2000, early_stopping_rounds=50, **PARAMS)
pilot.fit(X_tr, y_tr, eval_set=[(X_es, y_es)], verbose=False)
best_n = pilot.best_iteration + 1
print(f"  Best n_estimators: {best_n}")

# ── 7. Train calibrated model on full training split ─────────────────────────

print(f"\nTraining calibrated model (n_estimators={best_n}, isotonic cv=5)...")
base = XGBClassifier(n_estimators=best_n, **PARAMS)
model = CalibratedClassifierCV(base, method="isotonic", cv=5)
model.fit(X_train, y_train)

# ── 8. Train pure XGBoost for SHAP (CalibratedClassifierCV is not tree-native) ─

print("Training XGBoost for SHAP feature attribution...")
xgb_for_shap = XGBClassifier(n_estimators=best_n, **PARAMS)
xgb_for_shap.fit(X_train, y_train)

# ── 9. Evaluate on held-out test set ─────────────────────────────────────────

y_pred  = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
auc      = roc_auc_score(y_test, y_proba)
brier    = brier_score_loss(y_test, y_proba)

print(f"\n{'='*44}")
print(f"  Accuracy   : {accuracy:.4f}")
print(f"  AUC-ROC    : {auc:.4f}")
print(f"  Brier score: {brier:.4f}  (lower = better calibrated)")
print(f"  CV mean AUC: {np.mean(cv_scores):.4f}")
print(f"{'='*44}")

# ── 10. Save model ────────────────────────────────────────────────────────────
# Saved as a dict so evaluate.py and prediction_tab.py can load both
# the calibrated model (for predict_proba) and the raw XGBoost (for SHAP).

os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
artifact = {"model": model, "xgb_for_shap": xgb_for_shap}
pickle.dump(artifact, open(MODEL_PATH, "wb"))
print(f"\n✅  Model saved → {MODEL_PATH}")
