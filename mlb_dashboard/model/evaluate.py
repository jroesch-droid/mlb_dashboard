"""
model/evaluate.py
=================
Load the saved model and produce a full evaluation report:
  - Accuracy, Precision, Recall, F1, AUC-ROC
  - Confusion matrix
  - ROC curve (saved as PNG)
  - SHAP summary plot (saved as PNG)

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
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data.fetch_data import fetch_pitching_stats, fetch_schedule
from data.feature_eng import FEATURE_COLS, TEAM_CODES, build_training_dataset
from sklearn.model_selection import train_test_split

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")
OUT_DIR = os.path.join(os.path.dirname(__file__), "eval_outputs")
os.makedirs(OUT_DIR, exist_ok=True)

YEAR = 2024

# ── Load model ────────────────────────────────────────────────────────────────
model = pickle.load(open(MODEL_PATH, "rb"))
print("Model loaded.")

# ── Rebuild test set ──────────────────────────────────────────────────────────
schedules = {code: fetch_schedule(YEAR, code) for code in TEAM_CODES}
pitching_df = fetch_pitching_stats(YEAR)
train_df = build_training_dataset(schedules, pitching_df)

X = train_df[FEATURE_COLS]
y = train_df["label"]
_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

# ── Metrics ───────────────────────────────────────────────────────────────────
print("\n" + "="*50)
print(classification_report(y_test, y_pred, target_names=["Away Win", "Home Win"]))
print(f"AUC-ROC: {roc_auc_score(y_test, y_proba):.4f}")
print("="*50)

# ── Confusion matrix ──────────────────────────────────────────────────────────
cm = confusion_matrix(y_test, y_pred)
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

# ── SHAP summary ──────────────────────────────────────────────────────────────
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)
fig, ax = plt.subplots(figsize=(7, 4))
shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "shap_summary.png"), dpi=150)
plt.close()
print("Saved shap_summary.png")
