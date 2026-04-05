"""
MLB Analytics Dashboard
=======================
Main entry point for the Plotly Dash application.
Run locally with:  python app.py
Deploy with:       gunicorn app:server
"""

import dash
from dash import dcc, html

from components.player_tab import player_tab_layout, register_player_callbacks
from components.prediction_tab import prediction_tab_layout, register_prediction_callbacks

# ── App init ──────────────────────────────────────────────────────────────────
app = dash.Dash(
    __name__,
    title="MLB Analytics Dashboard",
    suppress_callback_exceptions=True,
)
server = app.server  # Required for gunicorn / Render deployment

# ── Layout ────────────────────────────────────────────────────────────────────
app.layout = html.Div(
    className="app-wrapper",
    children=[
        # ── Header ────────────────────────────────────────────────────────────
        html.Div(
            className="header",
            children=[
                html.Span("⚾", className="header-icon"),
                html.H1("MLB Analytics Dashboard", className="header-title"),
                html.P(
                    "Player Performance Tracker + Game Win Predictor",
                    className="header-subtitle",
                ),
            ],
        ),

        # ── Tabs ──────────────────────────────────────────────────────────────
        dcc.Tabs(
            id="main-tabs",
            value="tab-player",
            className="main-tabs",
            children=[
                dcc.Tab(
                    label="📊  Player Performance",
                    value="tab-player",
                    className="tab",
                    selected_className="tab--selected",
                    children=player_tab_layout(),
                ),
                dcc.Tab(
                    label="🔮  Game Prediction",
                    value="tab-prediction",
                    className="tab",
                    selected_className="tab--selected",
                    children=prediction_tab_layout(),
                ),
            ],
        ),

        # ── Footer ────────────────────────────────────────────────────────────
        html.Footer(
            "Built with Plotly Dash · pybaseball · XGBoost · SHAP",
            className="footer",
        ),
    ],
)

# ── Register callbacks ────────────────────────────────────────────────────────
register_player_callbacks(app)
register_prediction_callbacks(app)

# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True)
