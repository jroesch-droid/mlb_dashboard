"""
model/train_model.py
====================
Train the XGBoost game-prediction model and save it to model/model.pkl.

Usage:
    python model\train_model.py
"""

import os
import pickle
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data.fetch_data import fetch_pitching_stats, fetch_schedule
from data.feature_eng import FEATURE_COLS, TEAM_CODES, build_training_dataset

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")

# ── Enable pybaseball cache (adds rate limiting, prevents blocks) ──────────────
try:
    from pybaseball import cache
    cache.enable()
    print("✓  pybaseball cache enabled")
except Exception:
    pass

# ── 1. Fetch schedules with delays between requests ───────────────────────────

YEAR = 2024

print(f"\nFetching schedules for {len(TEAM_CODES)} teams ({YEAR})...")
print("(Pausing 3s between requests to avoid being blocked by Baseball Reference)\n")

schedules = {}
for i, code in enumerate(TEAM_CODES):
    try:
        schedules[code] = fetch_schedule(YEAR, code)
        print(f"  ✓  {code}  ({len(schedules[code])} games)")
    except Exception as e:
        print(f"  ✗  {code}: skipped")

    # Pause between requests to be polite to Baseball Reference
    if i < len(TEAM_CODES) - 1:
        time.sleep(3)

print(f"\nSuccessfully loaded {len(schedules)} / {len(TEAM_CODES)} teams.")

# ── 2. If all requests failed, generate synthetic training data ───────────────

if len(schedules) == 0:
    print("\n⚠️  No schedule data retrieved from Baseball Reference.")
    print("    Generating synthetic training data so the model can still be built.")
    print("    Re-run after a few minutes to try fetching real data.\n")

    np.random.seed(42)
    n = 2000

    # Simulate plausible MLB game features
    home_win_pct  = np.random.beta(5, 5, n)
    away_win_pct  = np.random.beta(5, 5, n)
    home_run_diff = np.random.normal(0, 1.5, n)
    away_run_diff = np.random.normal(0, 1.5, n)
    home_sp_era   = np.random.normal(4.0, 0.8, n).clip(2.0, 7.0)
    away_sp_era   = np.random.normal(4.0, 0.8, n).clip(2.0, 7.0)
    home_sp_whip  = np.random.normal(1.25, 0.2, n).clip(0.8, 2.0)
    away_sp_whip  = np.random.normal(1.25, 0.2, n).clip(0.8, 2.0)
    h2h_win_pct   = np.random.beta(3, 3, n)

    # Win probability shaped by features (realistic signal + noise)
    logit = (
        0.15
        + 1.2 * (home_win_pct - away_win_pct)
        + 0.3 * (home_run_diff - away_run_diff)
        - 0.2 * (home_sp_era - away_sp_era)
        - 0.15 * (home_sp_whip - away_sp_whip)
        + 0.4 * (h2h_win_pct - 0.5)
        + np.random.normal(0, 0.5, n)
    )
    prob_home_win = 1 / (1 + np.exp(-logit))
    labels = (np.random.uniform(size=n) < prob_home_win).astype(int)

    train_df = pd.DataFrame({
        "home_rolling_win_pct_10g": home_win_pct,
        "away_rolling_win_pct_10g": away_win_pct,
        "home_run_diff_15g":        home_run_diff,
        "away_run_diff_15g":        away_run_diff,
        "home_sp_era":              home_sp_era,
        "home_sp_whip":             home_sp_whip,
        "away_sp_era":              away_sp_era,
        "away_sp_whip":             away_sp_whip,
        "h2h_win_pct":              h2h_win_pct,
        "home_advantage":           np.ones(n),
        "label":                    labels,
    })

else:
    # ── 3. Fetch pitching stats and engineer features ─────────────────────────
    print("\nFetching pitching stats...")
    try:
        pitching_df = fetch_pitching_stats(YEAR)
    except Exception as e:
        print(f"  ✗  Pitching stats failed ({e}), using empty DataFrame")
        pitching_df = pd.DataFrame()

    print("\nEngineering features...")
    train_df = build_training_dataset(schedules, pitching_df)

# ── 4. Validate dataset ───────────────────────────────────────────────────────

print(f"\n  Dataset shape  : {train_df.shape}")

if train_df.empty or len(train_df) < 50:
    print("\n❌  Not enough data to train. Try again in a few minutes.")
    print("    Baseball Reference may be rate-limiting requests.")
    sys.exit(1)

print(f"  Class balance  : {train_df['label'].value_counts().to_dict()}")

X = train_df[FEATURE_COLS]
y = train_df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ── 5. Train model ────────────────────────────────────────────────────────────

print("\nTraining XGBoost model...")
model = XGBClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric="logloss",
    verbosity=0,
)

model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

# ── 6. Evaluate ───────────────────────────────────────────────────────────────

y_pred  = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
auc      = roc_auc_score(y_test, y_proba)

print(f"\n{'='*40}")
print(f"  Accuracy : {accuracy:.4f}")
print(f"  AUC-ROC  : {auc:.4f}")
print(f"{'='*40}")

# ── 7. Save model ─────────────────────────────────────────────────────────────

os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
pickle.dump(model, open(MODEL_PATH, "wb"))
print(f"\n✅  Model saved → {MODEL_PATH}")
