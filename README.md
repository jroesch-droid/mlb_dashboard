# mlb_dashboard
MLB analytics dashboard with XGBoost game prediction, Statcast player stats, and live 2019–2026 season data.

An interactive MLB analytics dashboard built with Python and Plotly Dash. Features a game prediction model trained on 13,000+ real games (2019–2026) using the official MLB Stats API, with an XGBoost classifier and 16 engineered features including rolling win percentage, run differential, win streak, rest days, and head-to-head history. The prediction tab displays win probability via a gauge chart, SHAP feature contribution breakdown, and each team's last 10 games. Current-season data auto-refreshes on every run so predictions stay up to date throughout the season. A second tab surfaces individual player Statcast and FanGraphs data for deeper performance analysis.


