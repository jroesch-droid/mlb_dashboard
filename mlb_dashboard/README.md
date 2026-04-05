# ⚾ MLB Analytics Dashboard

> **Live demo:** `https://your-app-name.onrender.com` ← replace after deployment

An interactive Plotly Dash web app combining MLB player performance tracking with an ML-powered game prediction engine.

---

## Features

| Tab | What it does |
|-----|-------------|
| 📊 **Player Performance** | Search any MLB player → view rolling BA/OBP/SLG, exit velocity scatter, and a comparison against league average |
| 🔮 **Game Prediction** | Pick home + away teams → get a win-probability gauge powered by XGBoost + SHAP feature importance |

---

## Tech Stack

| Layer | Tool |
|-------|------|
| UI Framework | Plotly Dash |
| Data | pybaseball (Statcast + FanGraphs) |
| ML Model | XGBoost |
| Explainability | SHAP |
| Deployment | Render.com |

---

## Quickstart

```bash
# 1. Clone and set up
git clone https://github.com/yourusername/mlb-dashboard.git
cd mlb-dashboard
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train the model (fetches ~2024 season data, takes 5–10 min first run)
python model/train_model.py

# 4. Run the app
python app.py
# → Open http://127.0.0.1:8050
```

---

## Model Performance

| Metric | Score |
|--------|-------|
| Accuracy | ~0.63 |
| AUC-ROC | ~0.69 |
| Precision | ~0.64 |
| Recall | ~0.62 |

Run `python model/evaluate.py` to regenerate full metrics, confusion matrix, and ROC curve.

---

## Project Structure

```
mlb-dashboard/
├── app.py                   # Dash app entry point
├── requirements.txt
├── data/
│   ├── fetch_data.py        # pybaseball wrappers with local caching
│   ├── feature_eng.py       # Feature engineering pipeline
│   └── cache/               # Auto-generated CSVs (.gitignored)
├── model/
│   ├── train_model.py       # XGBoost training script
│   ├── evaluate.py          # Metrics + confusion matrix + SHAP plots
│   └── model.pkl            # Saved model (.gitignored)
├── components/
│   ├── player_tab.py        # Player Performance tab
│   └── prediction_tab.py    # Game Prediction tab
└── assets/
    └── style.css            # Dark-theme custom styles
```

---

## Deployment (Render.com — free tier)

1. Push to a public GitHub repo
2. Go to [render.com](https://render.com) → **New Web Service**
3. Connect your repo and set:
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `gunicorn app:server`
   - **Python Version:** `3.11`
4. Click **Deploy** — you'll get a live `https://` URL in ~3 minutes

---

## Key Learnings

- **Statcast data is rich but noisy** — rolling windows smooth out small-sample variance and make features much more predictive
- **Home field advantage is real but small** — approximately 54% historical win rate for home teams; the model captures this via the `home_advantage` feature but learns that recent form matters more
- **SHAP adds real value** — seeing which features drove each individual prediction makes the model debuggable and trustworthy, not just a black box
- **Data caching is essential** — pybaseball fetches can be slow; writing CSVs locally cut development iteration time dramatically

---

## Demo Video

[📹 Watch on Loom](#) ← add link after recording

---

*Built with ❤️ and a lot of baseball data*
