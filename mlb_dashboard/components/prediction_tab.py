"""
components/prediction_tab.py
=============================
Tab 2 — Game Prediction
  • Home / Away team dropdowns
  • Win-probability gauge chart
  • SHAP feature-contribution bar chart
  • Recent form table (last 10 games)
"""

import os
import pickle
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import shap
from dash import Input, Output, dcc, html

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "model", "model.pkl")

# MLB team options
TEAM_OPTIONS = [
    {"label": "Arizona Diamondbacks", "value": "ARI"},
    {"label": "Atlanta Braves",       "value": "ATL"},
    {"label": "Baltimore Orioles",    "value": "BAL"},
    {"label": "Boston Red Sox",       "value": "BOS"},
    {"label": "Chicago Cubs",         "value": "CHC"},
    {"label": "Chicago White Sox",    "value": "CWS"},
    {"label": "Cincinnati Reds",      "value": "CIN"},
    {"label": "Cleveland Guardians",  "value": "CLE"},
    {"label": "Colorado Rockies",     "value": "COL"},
    {"label": "Detroit Tigers",       "value": "DET"},
    {"label": "Houston Astros",       "value": "HOU"},
    {"label": "Kansas City Royals",   "value": "KCR"},
    {"label": "Los Angeles Angels",   "value": "LAA"},
    {"label": "Los Angeles Dodgers",  "value": "LAD"},
    {"label": "Miami Marlins",        "value": "MIA"},
    {"label": "Milwaukee Brewers",    "value": "MIL"},
    {"label": "Minnesota Twins",      "value": "MIN"},
    {"label": "New York Mets",        "value": "NYM"},
    {"label": "New York Yankees",     "value": "NYY"},
    {"label": "Oakland Athletics",    "value": "OAK"},
    {"label": "Philadelphia Phillies","value": "PHI"},
    {"label": "Pittsburgh Pirates",   "value": "PIT"},
    {"label": "San Diego Padres",     "value": "SDP"},
    {"label": "San Francisco Giants", "value": "SFG"},
    {"label": "Seattle Mariners",     "value": "SEA"},
    {"label": "St. Louis Cardinals",  "value": "STL"},
    {"label": "Tampa Bay Rays",       "value": "TBR"},
    {"label": "Texas Rangers",        "value": "TEX"},
    {"label": "Toronto Blue Jays",    "value": "TOR"},
    {"label": "Washington Nationals", "value": "WSN"},
]

FEATURE_LABELS = {
    "home_rolling_win_pct_10g": "Home Win % (L10)",
    "away_rolling_win_pct_10g": "Away Win % (L10)",
    "home_run_diff_15g":        "Home Run Diff (L15)",
    "away_run_diff_15g":        "Away Run Diff (L15)",
    "home_sp_era":              "Home SP ERA",
    "home_sp_whip":             "Home SP WHIP",
    "away_sp_era":              "Away SP ERA",
    "away_sp_whip":             "Away SP WHIP",
    "h2h_win_pct":              "Head-to-Head Win %",
    "home_advantage":           "Home Advantage",
}


# ── Layout ────────────────────────────────────────────────────────────────────

def prediction_tab_layout():
    return html.Div(
        className="tab-content",
        children=[
            # ── Controls ──────────────────────────────────────────────────────
            html.Div(
                className="controls-row",
                children=[
                    html.Div([
                        html.Label("🏠  Home Team", className="control-label"),
                        dcc.Dropdown(
                            id="home-team",
                            options=TEAM_OPTIONS,
                            placeholder="Select home team...",
                            className="dropdown",
                        ),
                    ], className="control-group control-group--lg"),

                    html.Div(
                        html.Span("vs", className="vs-badge"),
                        className="control-group control-group--vs",
                    ),

                    html.Div([
                        html.Label("✈️  Away Team", className="control-label"),
                        dcc.Dropdown(
                            id="away-team",
                            options=TEAM_OPTIONS,
                            placeholder="Select away team...",
                            className="dropdown",
                        ),
                    ], className="control-group control-group--lg"),

                    html.Div([
                        html.Label("Season", className="control-label"),
                        dcc.Dropdown(
                            id="pred-season",
                            options=[{"label": str(y), "value": y} for y in range(2024, 2018, -1)],
                            value=2024,
                            clearable=False,
                            className="dropdown",
                        ),
                    ], className="control-group"),

                    html.Div([
                        html.Button(
                            "Predict",
                            id="predict-btn",
                            n_clicks=0,
                            className="btn-primary",
                        ),
                    ], className="control-group control-group--btn"),
                ],
            ),

            # ── Status ────────────────────────────────────────────────────────
            html.Div(id="pred-status", className="status-msg"),

            # ── Charts ────────────────────────────────────────────────────────
            html.Div(
                className="charts-grid",
                children=[
                    html.Div([
                        html.H3("Win Probability", className="chart-title"),
                        dcc.Graph(id="prediction-gauge", className="chart"),
                    ], className="chart-card"),

                    html.Div([
                        html.H3("SHAP Feature Contributions", className="chart-title"),
                        dcc.Graph(id="shap-bar", className="chart"),
                    ], className="chart-card"),

                    html.Div([
                        html.H3("Home Team — Last 10 Games", className="chart-title"),
                        html.Div(id="home-form-table"),
                    ], className="chart-card"),

                    html.Div([
                        html.H3("Away Team — Last 10 Games", className="chart-title"),
                        html.Div(id="away-form-table"),
                    ], className="chart-card"),
                ],
            ),

            # ── Model info ────────────────────────────────────────────────────
            html.Div(
                className="model-info",
                children=[
                    html.P("Model: XGBoost · Features: 10 · Training data: 2019–2024 seasons"),
                    html.P("Expected accuracy ~63% · AUC-ROC ~0.69"),
                ],
            ),
        ],
    )


# ── Callbacks ─────────────────────────────────────────────────────────────────

def register_prediction_callbacks(app):

    @app.callback(
        Output("prediction-gauge", "figure"),
        Output("shap-bar", "figure"),
        Output("home-form-table", "children"),
        Output("away-form-table", "children"),
        Output("pred-status", "children"),
        Input("predict-btn", "n_clicks"),
        Input("home-team", "value"),
        Input("away-team", "value"),
        Input("pred-season", "value"),
        prevent_initial_call=True,
    )
    def update_prediction(n_clicks, home_code, away_code, season):
        from data.fetch_data import fetch_schedule

        if not home_code or not away_code:
            return _empty_fig("Select both teams"), _empty_fig(""), _no_table(), _no_table(), "⚠️  Select both teams."
        if home_code == away_code:
            return _empty_fig("Teams must differ"), _empty_fig(""), _no_table(), _no_table(), "⚠️  Home and away teams must be different."

        # Load model
        if not os.path.exists(MODEL_PATH):
            msg = "❌  model.pkl not found. Run: python model/train_model.py"
            return _empty_fig("Model not trained"), _empty_fig(""), _no_table(), _no_table(), msg

        model = pickle.load(open(MODEL_PATH, "rb"))

        # Fetch schedules
        try:
            home_sched = fetch_schedule(season, home_code)
            away_sched = fetch_schedule(season, away_code)
        except Exception as e:
            msg = f"❌  Could not fetch schedule data: {e}"
            return _empty_fig("Data error"), _empty_fig(""), _no_table(), _no_table(), msg

        # Build features
        from data.feature_eng import build_game_features, FEATURE_COLS
        features_df = build_game_features(
            home_schedule=home_sched,
            away_schedule=away_sched,
            pitching_df=pd.DataFrame(),   # no SP lookup in quick mode
            home_team_name=home_code,
            away_team_name=away_code,
        )

        X = features_df[FEATURE_COLS]

        # Predict
        win_prob = float(model.predict_proba(X)[0, 1])

        # SHAP
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(X)

        # Charts
        gauge_fig = _build_gauge(win_prob, home_code, away_code)
        shap_fig  = _build_shap_bar(shap_vals[0], X.columns.tolist())
        home_table = _build_form_table(home_sched)
        away_table = _build_form_table(away_sched)

        home_name = next((t["label"] for t in TEAM_OPTIONS if t["value"] == home_code), home_code)
        away_name = next((t["label"] for t in TEAM_OPTIONS if t["value"] == away_code), away_code)
        status = (
            f"✅  {home_name} win probability: {win_prob*100:.1f}%  |  "
            f"{away_name} win probability: {(1-win_prob)*100:.1f}%"
        )

        return gauge_fig, shap_fig, home_table, away_table, status


# ── Chart builders ────────────────────────────────────────────────────────────

def _empty_fig(msg: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=msg, xref="paper", yref="paper",
                       x=0.5, y=0.5, showarrow=False,
                       font=dict(size=14, color="#94a3b8"))
    fig.update_layout(**_base_layout())
    return fig


def _base_layout(height: int = 340) -> dict:
    return dict(
        height=height,
        margin=dict(l=40, r=20, t=30, b=40),
        plot_bgcolor="#0f172a",
        paper_bgcolor="#0f172a",
        font=dict(color="#e2e8f0"),
    )


def _build_gauge(win_prob: float, home_code: str, away_code: str) -> go.Figure:
    """Gauge chart showing home-team win probability."""
    pct = round(win_prob * 100, 1)
    color = "#22c55e" if pct >= 60 else "#f59e0b" if pct >= 45 else "#ef4444"

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=pct,
        number={"suffix": "%", "font": {"size": 40, "color": "#e2e8f0"}},
        delta={"reference": 50, "suffix": "%", "font": {"size": 16}},
        title={"text": f"{home_code} Win Probability", "font": {"size": 16, "color": "#e2e8f0"}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#475569"},
            "bar":  {"color": color, "thickness": 0.3},
            "bgcolor": "#1e293b",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 40],  "color": "#1e293b"},
                {"range": [40, 60], "color": "#1e293b"},
                {"range": [60, 100],"color": "#1e293b"},
            ],
            "threshold": {
                "line": {"color": "#e2e8f0", "width": 2},
                "thickness": 0.8,
                "value": 50,
            },
        },
    ))
    fig.update_layout(**_base_layout(height=300))
    return fig


def _build_shap_bar(shap_values: np.ndarray, feature_names: list) -> go.Figure:
    """Horizontal bar chart of SHAP values for a single prediction."""
    labels = [FEATURE_LABELS.get(f, f) for f in feature_names]
    colors = ["#22c55e" if v > 0 else "#ef4444" for v in shap_values]

    # Sort by absolute value
    order = np.argsort(np.abs(shap_values))
    labels_sorted = [labels[i] for i in order]
    vals_sorted   = [shap_values[i] for i in order]
    colors_sorted = [colors[i] for i in order]

    fig = go.Figure(go.Bar(
        x=vals_sorted,
        y=labels_sorted,
        orientation="h",
        marker_color=colors_sorted,
        hovertemplate="<b>%{y}</b><br>SHAP: %{x:.4f}<extra></extra>",
    ))

    layout = _base_layout()
    layout.update(
        xaxis_title="SHAP value (impact on win probability)",
        xaxis=dict(gridcolor="#1e293b", zeroline=True, zerolinecolor="#475569"),
        yaxis=dict(gridcolor="#1e293b"),
        shapes=[{
            "type": "line", "x0": 0, "x1": 0,
            "y0": -0.5, "y1": len(labels) - 0.5,
            "line": {"color": "#475569", "width": 1},
        }],
    )
    fig.update_layout(**layout)
    return fig


def _build_form_table(schedule_df: pd.DataFrame) -> html.Div:
    """HTML table of last 10 games."""
    if schedule_df is None or schedule_df.empty:
        return _no_table()

    recent = schedule_df.tail(10).copy()
    # Normalize column names
    col_map = {"W/L": "W/L", "Opp": "Opp", "R": "R", "RA": "RA", "Date": "Date"}
    available = {c: c for c in recent.columns if c in col_map}
    display_cols = [c for c in ["Date", "Opp", "R", "RA", "W/L"] if c in recent.columns]

    if not display_cols:
        return _no_table()

    recent = recent[display_cols].tail(10)

    rows = []
    for _, row in recent.iterrows():
        result = str(row.get("W/L", ""))
        row_class = "table-row--win" if result.startswith("W") else "table-row--loss"
        cells = [html.Td(str(row[c])) for c in display_cols]
        rows.append(html.Tr(cells, className=row_class))

    return html.Table(
        className="form-table",
        children=[
            html.Thead(html.Tr([html.Th(c) for c in display_cols])),
            html.Tbody(rows),
        ],
    )


def _no_table() -> html.Div:
    return html.Div("No data available.", className="no-data")
