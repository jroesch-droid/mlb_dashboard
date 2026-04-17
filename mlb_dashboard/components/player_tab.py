"""
components/player_tab.py
========================
Tab 1 — Player Performance
  • Player search by name
  • Season selector
  • Rolling batting avg / OBP / SLG line chart
  • Exit velocity vs. launch angle scatter
  • Stat comparison bar chart vs. league average
"""

import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Input, Output, dcc, html

# ── Layout ────────────────────────────────────────────────────────────────────

def player_tab_layout():
    return html.Div(
        className="tab-content",
        children=[
            # ── Controls ──────────────────────────────────────────────────────
            html.Div(
                className="controls-row",
                children=[
                    html.Div([
                        html.Label("Player Last Name", className="control-label"),
                        dcc.Input(
                            id="player-last",
                            type="text",
                            placeholder="e.g. judge",
                            debounce=True,
                            className="text-input",
                        ),
                    ], className="control-group"),

                    html.Div([
                        html.Label("Player First Name", className="control-label"),
                        dcc.Input(
                            id="player-first",
                            type="text",
                            placeholder="e.g. aaron",
                            debounce=True,
                            className="text-input",
                        ),
                    ], className="control-group"),

                    html.Div([
                        html.Label("Season", className="control-label"),
                        dcc.Dropdown(
                            id="player-season",
                            options=[{"label": str(y), "value": y} for y in range(datetime.date.today().year, 2018, -1)],
                            value=datetime.date.today().year,
                            clearable=False,
                            className="dropdown",
                        ),
                    ], className="control-group"),

                    html.Div([
                        html.Button(
                            "Load Player",
                            id="load-player-btn",
                            n_clicks=0,
                            className="btn-primary",
                        ),
                    ], className="control-group control-group--btn"),
                ],
            ),

            # ── Status message ────────────────────────────────────────────────
            html.Div(id="player-status", className="status-msg"),

            # ── Charts ────────────────────────────────────────────────────────
            html.Div(
                className="charts-grid",
                children=[
                    html.Div([
                        html.H3("Rolling Stats Over Time", className="chart-title"),
                        dcc.Graph(id="rolling-stats-chart", className="chart"),
                    ], className="chart-card chart-card--wide"),

                    html.Div([
                        html.H3("Exit Velocity vs. Launch Angle", className="chart-title"),
                        dcc.Graph(id="ev-la-scatter", className="chart"),
                    ], className="chart-card"),

                    html.Div([
                        html.H3("vs. League Average", className="chart-title"),
                        dcc.Graph(id="league-avg-bar", className="chart"),
                    ], className="chart-card"),
                ],
            ),
        ],
    )


# ── Callbacks ─────────────────────────────────────────────────────────────────

def register_player_callbacks(app):

    @app.callback(
        Output("rolling-stats-chart", "figure"),
        Output("ev-la-scatter", "figure"),
        Output("league-avg-bar", "figure"),
        Output("player-status", "children"),
        Input("load-player-btn", "n_clicks"),
        Input("player-last", "value"),
        Input("player-first", "value"),
        Input("player-season", "value"),
        prevent_initial_call=True,
    )
    def update_player_charts(n_clicks, last, first, season):
        """Fetch Statcast data and build the three player charts."""
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
        from data.fetch_data import fetch_batter_statcast, get_player_id, fetch_batting_stats

        if not last or not first:
            empty = _empty_fig("Enter a player name above")
            return empty, empty, empty, "⚠️  Please enter both first and last name."

        player_id = get_player_id(last.strip(), first.strip())
        if player_id is None:
            empty = _empty_fig("Player not found")
            return empty, empty, empty, f"❌  Player '{first} {last}' not found."

        start = f"{season}-03-28"
        end   = f"{season}-10-01"

        try:
            df = fetch_batter_statcast(player_id, start, end)
        except Exception as e:
            empty = _empty_fig(str(e))
            return empty, empty, empty, f"❌  Error fetching data: {e}"

        if df is None or df.empty:
            empty = _empty_fig("No Statcast data found")
            return empty, empty, empty, "⚠️  No Statcast data found for this player / season."

        status = f"✅  Loaded {len(df):,} plate appearances for {first.title()} {last.title()} ({season})"

        # ── Chart 1: Rolling stats ─────────────────────────────────────────
        rolling_fig = _build_rolling_stats(df, first, last, season)

        # ── Chart 2: EV vs LA scatter ──────────────────────────────────────
        ev_la_fig = _build_ev_la_scatter(df, first, last)

        # ── Chart 3: vs League avg ─────────────────────────────────────────
        try:
            batting_df = fetch_batting_stats(season)
            bar_fig = _build_league_avg_bar(batting_df, last, first)
        except Exception as e:
            bar_fig = _empty_fig("FanGraphs stats unavailable")
            status += f"  ⚠️  League comparison unavailable ({e})."

        return rolling_fig, ev_la_fig, bar_fig, status


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
        xaxis=dict(gridcolor="#1e293b", zeroline=False),
        yaxis=dict(gridcolor="#1e293b", zeroline=False),
    )


def _build_rolling_stats(df: pd.DataFrame, first: str, last: str, season: int) -> go.Figure:
    """Rolling 20-PA batting average, OBP proxy, and SLG proxy."""
    df = df.copy()
    df["game_date"] = pd.to_datetime(df["game_date"])
    df = df.sort_values("game_date")

    # Only keep balls put in play (events that ended the PA)
    pa = df[df["events"].notna()].copy()
    if pa.empty:
        return _empty_fig("No plate-appearance events found")

    pa["hit"] = pa["events"].isin(["single", "double", "triple", "home_run"]).astype(int)
    pa["ab"]  = (~pa["events"].isin(["walk", "hit_by_pitch", "sac_fly", "sac_bunt"])).astype(int)
    pa["tb"]  = pa["events"].map({"single": 1, "double": 2, "triple": 3, "home_run": 4}).fillna(0)
    pa["obp_num"] = pa["events"].isin(["single","double","triple","home_run","walk","hit_by_pitch"]).astype(int)

    window = 20
    pa["rolling_ba"]  = pa["hit"].rolling(window, min_periods=5).sum() / pa["ab"].rolling(window, min_periods=5).sum()
    pa["rolling_obp"] = pa["obp_num"].rolling(window, min_periods=5).mean()
    pa["rolling_slg"] = pa["tb"].rolling(window, min_periods=5).sum() / pa["ab"].rolling(window, min_periods=5).sum()

    pa = pa.dropna(subset=["rolling_ba"])

    fig = go.Figure()
    colors = {"BA": "#3b82f6", "OBP": "#10b981", "SLG": "#f59e0b"}
    for col, label, color in [
        ("rolling_ba", "BA", "#3b82f6"),
        ("rolling_obp", "OBP", "#10b981"),
        ("rolling_slg", "SLG", "#f59e0b"),
    ]:
        fig.add_trace(go.Scatter(
            x=pa["game_date"], y=pa[col],
            mode="lines", name=label,
            line=dict(color=color, width=2),
            hovertemplate=f"<b>{label}</b>: %{{y:.3f}}<br>%{{x|%b %d}}<extra></extra>",
        ))

    layout = _base_layout(height=350)
    layout.update(
        title=f"{first.title()} {last.title()} — Rolling {window}-PA Stats ({season})",
        legend=dict(orientation="h", y=1.1),
    )
    fig.update_layout(**layout)
    return fig


def _build_ev_la_scatter(df: pd.DataFrame, first: str, last: str) -> go.Figure:
    """Exit velocity vs launch angle colored by hit result."""
    df = df.copy()
    df = df.dropna(subset=["launch_speed", "launch_angle"])
    if df.empty:
        return _empty_fig("No Statcast exit-velocity data")

    color_map = {
        "single":   "#3b82f6",
        "double":   "#10b981",
        "triple":   "#f59e0b",
        "home_run": "#ef4444",
        "field_out":"#64748b",
        "strikeout":"#475569",
    }
    df["result"] = df["events"].fillna("other")
    df["color"]  = df["result"].map(color_map).fillna("#334155")

    fig = go.Figure()
    for result, color in color_map.items():
        subset = df[df["result"] == result]
        if subset.empty:
            continue
        fig.add_trace(go.Scatter(
            x=subset["launch_speed"],
            y=subset["launch_angle"],
            mode="markers",
            name=result.replace("_", " ").title(),
            marker=dict(color=color, size=5, opacity=0.7),
            hovertemplate=(
                "<b>%{text}</b><br>"
                "Exit Velo: %{x:.1f} mph<br>"
                "Launch Angle: %{y:.1f}°<extra></extra>"
            ),
            text=subset["result"].str.replace("_", " ").str.title(),
        ))

    layout = _base_layout()
    layout.update(
        xaxis_title="Exit Velocity (mph)",
        yaxis_title="Launch Angle (°)",
        legend=dict(orientation="h", y=-0.25, font=dict(size=10)),
    )
    fig.update_layout(**layout)
    return fig


def _build_league_avg_bar(batting_df: pd.DataFrame, last: str, first: str) -> go.Figure:
    """Grouped bar: player vs league average for BA, OBP, SLG, wOBA."""
    stats = ["AVG", "OBP", "SLG", "wOBA"]

    player_row = batting_df[batting_df["Name"].str.lower().str.contains(last.lower(), na=False)]
    if player_row.empty:
        return _empty_fig("Player not found in FanGraphs data")

    player_row = player_row.iloc[0]
    player_vals = [player_row.get(s, np.nan) for s in stats]

    league_vals = [batting_df[s].median() for s in stats]

    fig = go.Figure(data=[
        go.Bar(name=f"{first.title()} {last.title()}", x=stats, y=player_vals,
               marker_color="#3b82f6"),
        go.Bar(name="League Median", x=stats, y=league_vals,
               marker_color="#64748b"),
    ])

    layout = _base_layout()
    layout.update(
        barmode="group",
        yaxis_title="Rate",
        legend=dict(orientation="h", y=1.1),
    )
    fig.update_layout(**layout)
    return fig
