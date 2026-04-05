"""
data/feature_eng.py
===================
Feature engineering pipeline for the XGBoost game-prediction model.
Transforms raw schedule + team-stats DataFrames into ML-ready features.
"""

import numpy as np
import pandas as pd


# ── Rolling team features ─────────────────────────────────────────────────────

def add_rolling_win_pct(schedule_df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    """
    Add a rolling win-percentage column over the last `window` games.
    Expects a column 'W/L' with values 'W' or 'L'.
    """
    df = schedule_df.copy()
    df["win"] = (df["W/L"].str.startswith("W")).astype(int)
    df["rolling_win_pct"] = (
        df["win"].rolling(window, min_periods=1).mean()
    )
    return df


def add_run_differential(schedule_df: pd.DataFrame, window: int = 15) -> pd.DataFrame:
    """
    Add a rolling run-differential column over the last `window` games.
    Expects 'R' (runs scored) and 'RA' (runs allowed) columns.
    """
    df = schedule_df.copy()
    df["run_diff"] = df["R"] - df["RA"]
    df["rolling_run_diff"] = (
        df["run_diff"].rolling(window, min_periods=1).mean()
    )
    return df


# ── Pitcher features ──────────────────────────────────────────────────────────

def get_sp_features(pitching_df: pd.DataFrame, player_name: str) -> dict:
    """
    Extract starting-pitcher ERA and WHIP for a given player name.
    Returns a dict: {'sp_era': float, 'sp_whip': float}
    """
    row = pitching_df[pitching_df["Name"].str.contains(player_name, na=False)]
    if row.empty:
        return {"sp_era": np.nan, "sp_whip": np.nan}
    return {
        "sp_era": float(row.iloc[0].get("ERA", np.nan)),
        "sp_whip": float(row.iloc[0].get("WHIP", np.nan)),
    }


# ── Head-to-head record ───────────────────────────────────────────────────────

def compute_h2h(home_schedule: pd.DataFrame, away_team_name: str) -> float:
    """
    Compute head-to-head win % of home team vs. away team in current season.
    `home_schedule` must have 'Opp' (opponent) and 'W/L' columns.
    Returns win % as a float, or 0.5 if no games played yet.
    """
    h2h = home_schedule[home_schedule["Opp"] == away_team_name]
    if h2h.empty:
        return 0.5
    wins = h2h["W/L"].str.startswith("W").sum()
    return wins / len(h2h)


# ── Master feature builder ────────────────────────────────────────────────────

def build_game_features(
    home_schedule: pd.DataFrame,
    away_schedule: pd.DataFrame,
    pitching_df: pd.DataFrame,
    home_sp: str = "",
    away_sp: str = "",
    home_team_name: str = "",
    away_team_name: str = "",
) -> pd.DataFrame:
    """
    Build a single-row feature DataFrame for one matchup.

    Args:
        home_schedule: schedule_and_record DataFrame for home team (full season so far)
        away_schedule: schedule_and_record DataFrame for away team
        pitching_df:   FanGraphs pitching stats DataFrame
        home_sp:       Home starting pitcher name (partial match OK)
        away_sp:       Away starting pitcher name
        home_team_name: Opponent column value for head-to-head lookup
        away_team_name: Opponent column value for head-to-head lookup

    Returns:
        Single-row pd.DataFrame ready to pass to model.predict()
    """
    home_sched = add_rolling_win_pct(home_schedule)
    home_sched = add_run_differential(home_sched)
    away_sched = add_rolling_win_pct(away_schedule)
    away_sched = add_run_differential(away_sched)

    home_sp_feats = get_sp_features(pitching_df, home_sp) if home_sp else {"sp_era": np.nan, "sp_whip": np.nan}
    away_sp_feats = get_sp_features(pitching_df, away_sp) if away_sp else {"sp_era": np.nan, "sp_whip": np.nan}

    h2h = compute_h2h(home_schedule, away_team_name) if away_team_name else 0.5

    features = {
        "home_rolling_win_pct_10g": home_sched["rolling_win_pct"].iloc[-1] if not home_sched.empty else 0.5,
        "away_rolling_win_pct_10g": away_sched["rolling_win_pct"].iloc[-1] if not away_sched.empty else 0.5,
        "home_run_diff_15g":        home_sched["rolling_run_diff"].iloc[-1] if not home_sched.empty else 0.0,
        "away_run_diff_15g":        away_sched["rolling_run_diff"].iloc[-1] if not away_sched.empty else 0.0,
        "home_sp_era":              home_sp_feats["sp_era"],
        "home_sp_whip":             home_sp_feats["sp_whip"],
        "away_sp_era":              away_sp_feats["sp_era"],
        "away_sp_whip":             away_sp_feats["sp_whip"],
        "h2h_win_pct":              h2h,
        "home_advantage":           1,
    }

    return pd.DataFrame([features])


# ── Training-data builder ─────────────────────────────────────────────────────

TEAM_CODES = [
    "NYY", "BOS", "LAD", "HOU", "ATL", "CHC", "NYM", "SFG",
    "PHI", "TOR", "MIL", "ARI", "TEX", "SEA", "BAL", "MIN",
    "CLE", "TBR", "SDP", "CIN",
]


def build_training_dataset(
    schedules: dict[str, pd.DataFrame],
    pitching_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build a multi-game training DataFrame from a dict of schedules.

    Args:
        schedules:   {team_code: schedule_DataFrame}
        pitching_df: FanGraphs pitching stats

    Returns:
        DataFrame with feature columns + 'label' (1 = home team won)
    """
    rows = []

    for home_code, home_sched in schedules.items():
        home_sched = add_rolling_win_pct(home_sched)
        home_sched = add_run_differential(home_sched)

        for _, game in home_sched.iterrows():
            opp_code = str(game.get("Opp", "")).strip()
            if opp_code not in schedules:
                continue

            away_sched = schedules[opp_code]
            away_sched = add_rolling_win_pct(away_sched)
            away_sched = add_run_differential(away_sched)

            # Only use home games (avoid duplicate games)
            if str(game.get("Home_Away", "")).strip() == "@":
                continue

            label = 1 if str(game.get("W/L", "")).startswith("W") else 0

            row = {
                "home_rolling_win_pct_10g": game.get("rolling_win_pct", 0.5),
                "away_rolling_win_pct_10g": away_sched["rolling_win_pct"].iloc[-1] if not away_sched.empty else 0.5,
                "home_run_diff_15g":        game.get("rolling_run_diff", 0.0),
                "away_run_diff_15g":        away_sched["rolling_run_diff"].iloc[-1] if not away_sched.empty else 0.0,
                "home_sp_era":              np.nan,
                "home_sp_whip":             np.nan,
                "away_sp_era":              np.nan,
                "away_sp_whip":             np.nan,
                "h2h_win_pct":              compute_h2h(home_sched, opp_code),
                "home_advantage":           1,
                "label":                    label,
            }
            rows.append(row)

    df = pd.DataFrame(rows)
    # Fill NaN pitcher stats with league medians
    for col in ["home_sp_era", "home_sp_whip", "away_sp_era", "away_sp_whip"]:
        df[col] = df[col].fillna(df[col].median())
    return df


FEATURE_COLS = [
    "home_rolling_win_pct_10g",
    "away_rolling_win_pct_10g",
    "home_run_diff_15g",
    "away_run_diff_15g",
    "home_sp_era",
    "home_sp_whip",
    "away_sp_era",
    "away_sp_whip",
    "h2h_win_pct",
    "home_advantage",
]
