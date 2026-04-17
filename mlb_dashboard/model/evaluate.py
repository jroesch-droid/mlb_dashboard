"""
model/evaluate.py
=================
Load the saved model and produce a full evaluation report:
  - Accuracy, Precision, Recall, F1, AUC-ROC, Brier score
  - Confusion matrix
  - ROC curve
  - Calibration curve  (verifies isotonic calibration quality)
  - SHAP feature-importance bar chart

Matches the exact data path and temporal split used in train_model.py so
the test set here is identical to the one seen during training evaluation.

Usage:
    python model/evaluate.py
"""

import os
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data.fetch_data import fetch_mlb_games
from data.feature_eng import FEATURE_COLS, build_training_dataset_mlb

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")
OUT_DIR    = os.path.join(os.path.dirname(__file__), "eval_outputs")
os.makedirs(OUT_DIR, exist_ok=True)

YEARS = list(range(2019, 2027))

# ── Load model artifact ───────────────────────────────────────────────────────

artifact = pickle.load(open(MODEL_PATH, "rb"))
if isinstance(artifact, dict):
    model        = artifact["model"]
    xgb_for_shap = artifact.get("xgb_for_shap", artifact["model"])
elif isinstance(artifact, tuple):
    model        = artifact[0]
    xgb_for_shap = artifact[1] if len(artifact) > 1 else artifact[0]
else:
    # Legacy: plain XGBoost saved directly
    model        = artifact
    xgb_for_shap = artifact

print(f"Model loaded: {type(model).__name__}")

# ── Rebuild dataset (same MLB API path as train_model.py) ────────────────────

print(f"\nFetching game data ({YEARS[0]}–{YEARS[-1]})...")
mlb_frames = []
for year in YEARS:
    try:
        df = fetch_mlb_games(year)
        if not df.empty:
            mlb_frames.append(df)
            print(f"  ✓  {year}: {len(df):,} games")
        else:
            print(f"  ✗  {year}: no data returned")
    except Exception as e:
        print(f"  ✗  {year}: {e}")

if not mlb_frames:
    print("❌  No game data available. Cannot evaluate.")
    sys.exit(1)

all_games = pd.concat(mlb_frames, ignore_index=True)
print(f"\nBuilding features from {len(all_games):,} games...")
train_df = build_training_dataset_mlb(all_games, pitching_df=pd.DataFrame())
print(f"  ✓  Feature rows: {len(train_df):,}")

X = train_df[FEATURE_COLS]
y = train_df["label"]

# Temporal split — must match train_model.py exactly
split_idx = int(len(X) * 0.8)
X_test = X.iloc[split_idx:]
y_test = y.iloc[split_idx:]
print(f"  Test set: {len(X_test):,} games (last 20% chronologically)\n")

# ── Predict ───────────────────────────────────────────────────────────────────

y_pred  = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

# ── Metrics ───────────────────────────────────────────────────────────────────

print("=" * 52)
print(classification_report(y_test, y_pred, target_names=["Away Win", "Home Win"]))
print(f"AUC-ROC    : {roc_auc_score(y_test, y_proba):.4f}")
print(f"Brier score: {brier_score_loss(y_test, y_proba):.4f}  (lower = better calibrated)")
print("=" * 52)

# ── Confusion matrix ──────────────────────────────────────────────────────────

cm   = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=["Away Win", "Home Win"])
fig, ax = plt.subplots(figsize=(5, 4))
disp.plot(ax=ax, colorbar=False, cmap="Blues")
ax.set_title("Confusion Matrix")
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "confusion_matrix.png"), dpi=150)
plt.close()
print("Saved confusion_matrix.png")

# ── ROC curve ─────────────────────────────────────────────────────────────────

fpr, tpr, _ = roc_curve(y_test, y_proba)
auc = roc_auc_score(y_test, y_proba)
fig, ax = plt.subplots(figsize=(5, 4))
ax.plot(fpr, tpr, label=f"AUC = {auc:.3f}", color="#2563eb", lw=2)
ax.plot([0, 1], [0, 1], "k--", lw=1)
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve")
ax.legend(loc="lower right")
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "roc_curve.png"), dpi=150)
plt.close()
print("Saved roc_curve.png")

# ── Calibration curve ─────────────────────────────────────────────────────────
# A well-calibrated model follows the diagonal: predicted 60% → wins 60% of the time.

brier = brier_score_loss(y_test, y_proba)
prob_true, prob_pred = calibration_curve(y_test, y_proba, n_bins=10, strategy="quantile")
fig, ax = plt.subplots(figsize=(5, 4))
ax.plot(prob_pred, prob_true, "s-", color="#2563eb", lw=2,
        label=f"Model (Brier = {brier:.4f})")
ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect calibration")
ax.set_xlabel("Mean Predicted Probability")
ax.set_ylabel("Fraction of Positives")
ax.set_title("Calibration Curve")
ax.legend(loc="upper left")
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "calibration_curve.png"), dpi=150)
plt.close()
print("Saved calibration_curve.png")

# ── SHAP feature importance ───────────────────────────────────────────────────
# Uses xgb_for_shap (plain XGBoost) — CalibratedClassifierCV is not tree-native.

explainer   = shap.TreeExplainer(xgb_for_shap)
shap_values = explainer.shap_values(X_test)
fig, _ax = plt.subplots(figsize=(7, 5))
shap.summary_plot(shap_values, X_test, plot_type="bar", show=False, max_display=len(FEATURE_COLS))
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "shap_summary.png"), dpi=150)
plt.close()
print("Saved shap_summary.png")

print(f"\nAll outputs written to {OUT_DIR}/")
