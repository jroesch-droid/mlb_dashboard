"""
data/feature_eng.py
===================
Feature engineering pipeline for the XGBoost game-prediction model.
Transforms raw schedule / MLB API DataFrames into ML-ready features.
"""

import numpy as np
import pandas as pd


# ── Feature column list (shared by training and prediction) ──────────────────

FEATURE_COLS = [
    "home_win_pct_10",    # rolling win % — last 10 games
    "home_win_pct_20",    # rolling win % — last 20 games
    "away_win_pct_10",
    "away_win_pct_20",
    "home_run_diff_15",   # rolling avg run differential — last 15 games
    "away_run_diff_15",
    "home_streak",        # current win streak (+) / loss streak (−)
    "away_streak",
    "home_rest_days",     # days since last game
    "away_rest_days",
    "h2h_win_pct",        # home team win % vs this opponent (season so far)
    "home_sp_era",        # starting pitcher ERA
    "home_sp_whip",
    "away_sp_era",
    "away_sp_whip",
    "home_advantage",     # always 1
]

TEAM_CODES = [
    "NYY", "BOS", "LAD", "HOU", "ATL", "CHC", "NYM", "SFG",
    "PHI", "TOR", "MIL", "ARI", "TEX", "SEA", "BAL", "MIN",
    "CLE", "TBR", "SDP", "CIN",
]


# ── Helpers for pybaseball schedule DataFrames ───────────────────────────────

def add_rolling_win_pct(schedule_df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    """Add rolling_win_pct (10g) and rolling_win_pct_20 columns."""
    df = schedule_df.copy()
    if "win" not in df.columns:
        df["win"] = (df["W/L"].str.startswith("W")).astype(int)
    df["rolling_win_pct"]    = df["win"].rolling(10, min_periods=1).mean()
    df["rolling_win_pct_20"] = df["win"].rolling(20, min_periods=1).mean()
    return df


def add_run_differential(schedule_df: pd.DataFrame, window: int = 15) -> pd.DataFrame:
    """Add rolling run-differential column."""
    df = schedule_df.copy()
    df["run_diff"]        = df["R"] - df["RA"]
    df["rolling_run_diff"] = df["run_diff"].rolling(window, min_periods=1).mean()
    return df


def add_win_streak(schedule_df: pd.DataFrame) -> pd.DataFrame:
    """Add win_streak column: +N = N-game win streak, -N = N-game loss streak."""
    df = schedule_df.copy()
    if "win" not in df.columns:
        df["win"] = (df["W/L"].str.startswith("W")).astype(int)
    streak, streaks = 0, []
    for won in df["win"]:
        streaks.append(streak)          # streak BEFORE this game
        if won == 1:
            streak = streak + 1 if streak > 0 else 1
        else:
            streak = streak - 1 if streak < 0 else -1
    df["win_streak"] = streaks
    return df


def get_sp_features(pitching_df: pd.DataFrame, player_name: str) -> dict:
    row = pitching_df[pitching_df["Name"].str.contains(player_name, na=False)]
    if row.empty:
        return {"sp_era": np.nan, "sp_whip": np.nan}
    return {
        "sp_era":  float(row.iloc[0].get("ERA",  np.nan)),
        "sp_whip": float(row.iloc[0].get("WHIP", np.nan)),
    }


def compute_h2h(home_schedule: pd.DataFrame, away_team_name: str) -> float:
    """Win % of home team vs away team in current season (pybaseball path)."""
    h2h = home_schedule[home_schedule["Opp"] == away_team_name]
    if h2h.empty:
        return 0.5
    wins = h2h["W/L"].str.startswith("W").sum()
    return wins / len(h2h)


# ── MLB Stats API feature engineering ────────────────────────────────────────

def compute_team_rolling_stats_mlb(games_df: pd.DataFrame) -> dict:
    """
    Pre-compute rolling stats for every (team_id, game_date) in games_df.
    Stats use ONLY games played BEFORE each game (no data leakage).

    Returns: {(team_id: int, date: str) -> stats_dict}
    where stats_dict has keys: win_pct_10, win_pct_20, run_diff_15,
                                streak, rest_days
    """
    games_df = games_df.copy()
    games_df["game_date"] = pd.to_datetime(games_df["game_date"])

    # Expand each game into two rows — one per team
    home = games_df[["game_date", "home_id", "away_id", "home_score", "away_score"]].copy()
    home.columns = ["game_date", "team_id", "opp_id", "runs_scored", "runs_allowed"]

    away = games_df[["game_date", "away_id", "home_id", "away_score", "home_score"]].copy()
    away.columns = ["game_date", "team_id", "opp_id", "runs_scored", "runs_allowed"]

    tg = pd.concat([home, away], ignore_index=True)
    tg["won"]      = (tg["runs_scored"] > tg["runs_allowed"]).astype(int)
    tg["run_diff"] = tg["runs_scored"] - tg["runs_allowed"]
    tg = tg.sort_values(["team_id", "game_date"]).reset_index(drop=True)

    result = {}
    for team_id, grp in tg.groupby("team_id"):
        grp = grp.sort_values("game_date").reset_index(drop=True)

        # shift(1) so rolling stats at position i are computed from games [0..i-1]
        won_s  = grp["won"].shift(1)
        diff_s = grp["run_diff"].shift(1)

        win_pct_10  = won_s.rolling(10, min_periods=3).mean()
        win_pct_20  = won_s.rolling(20, min_periods=5).mean()
        run_diff_15 = diff_s.rolling(15, min_periods=3).mean()

        # Streak: based on games BEFORE the current one
        streak, streaks = 0, [0]
        for i in range(1, len(grp)):
            prev = grp["won"].iloc[i - 1]
            if prev == 1:
                streak = streak + 1 if streak > 0 else 1
            else:
                streak = streak - 1 if streak < 0 else -1
            streaks.append(streak)

        rest = (grp["game_date"] - grp["game_date"].shift(1)).dt.days.fillna(1)

        for i in range(len(grp)):
            key = (int(team_id), str(grp["game_date"].iloc[i].date()))
            result[key] = {
                "win_pct_10":  float(win_pct_10.iloc[i])  if not pd.isna(win_pct_10.iloc[i])  else 0.5,
                "win_pct_20":  float(win_pct_20.iloc[i])  if not pd.isna(win_pct_20.iloc[i])  else 0.5,
                "run_diff_15": float(run_diff_15.iloc[i]) if not pd.isna(run_diff_15.iloc[i]) else 0.0,
                "streak":      float(streaks[i]),
                "rest_days":   float(rest.iloc[i]),
            }

    return result


def build_training_dataset_mlb(games_df: pd.DataFrame, pitching_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a training DataFrame from MLB Stats API game data.
    Each row is one game; features are computed from strictly prior games only.
    """
    if games_df.empty:
        return pd.DataFrame()

    games_df = games_df.copy()
    games_df["game_date"] = pd.to_datetime(games_df["game_date"])
    games_df = games_df.sort_values("game_date").reset_index(drop=True)

    team_stats = compute_team_rolling_stats_mlb(games_df)

    # Running H2H tracker: wins[(home_id, away_id)] = home team wins vs that opp
    h2h_wins  = {}
    h2h_total = {}

    rows = []
    for _, game in games_df.iterrows():
        home_id  = int(game["home_id"])
        away_id  = int(game["away_id"])
        date_str = str(game["game_date"].date())

        h = team_stats.get((home_id, date_str))
        a = team_stats.get((away_id, date_str))

        home_won = int(game["home_score"]) > int(game["away_score"])

        # Update H2H before checking (so early games still update tracker)
        h2h_wins[(home_id, away_id)]  = h2h_wins.get((home_id, away_id), 0) + (1 if home_won else 0)
        h2h_total[(home_id, away_id)] = h2h_total.get((home_id, away_id), 0) + 1

        if h is None or a is None:
            continue  # skip games with insufficient history

        total_h2h = h2h_total.get((home_id, away_id), 0) - 1  # exclude current game
        wins_h2h  = h2h_wins.get((home_id, away_id), 0)  - (1 if home_won else 0)
        h2h_pct   = wins_h2h / total_h2h if total_h2h > 0 else 0.5

        rows.append({
            "home_win_pct_10":  h["win_pct_10"],
            "home_win_pct_20":  h["win_pct_20"],
            "away_win_pct_10":  a["win_pct_10"],
            "away_win_pct_20":  a["win_pct_20"],
            "home_run_diff_15": h["run_diff_15"],
            "away_run_diff_15": a["run_diff_15"],
            "home_streak":      h["streak"],
            "away_streak":      a["streak"],
            "home_rest_days":   h["rest_days"],
            "away_rest_days":   a["rest_days"],
            "h2h_win_pct":      h2h_pct,
            "home_sp_era":      np.nan,
            "home_sp_whip":     np.nan,
            "away_sp_era":      np.nan,
            "away_sp_whip":     np.nan,
            "home_advantage":   1,
            "label":            int(home_won),
        })

    df = pd.DataFrame(rows)
    for col in ["home_sp_era", "home_sp_whip", "away_sp_era", "away_sp_whip"]:
        df[col] = df[col].fillna(4.0)   # league-average ERA/WHIP defaults
    df["away_sp_whip"] = df["away_sp_whip"].fillna(1.25)
    df["home_sp_whip"] = df["home_sp_whip"].fillna(1.25)
    return df


def build_prediction_features_mlb(
    home_id: int,
    away_id: int,
    games_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build a single-row feature DataFrame for a matchup using MLB API data.
    Uses the most recent available stats for each team.
    """
    team_stats = compute_team_rolling_stats_mlb(games_df)

    home_dates = sorted(d for (tid, d) in team_stats if tid == home_id)
    away_dates = sorted(d for (tid, d) in team_stats if tid == away_id)

    if not home_dates or not away_dates:
        raise ValueError("Not enough game history for one or both teams.")

    h = team_stats[(home_id, home_dates[-1])]
    a = team_stats[(away_id, away_dates[-1])]

    # H2H this season
    h2h_games  = games_df[
        ((games_df["home_id"] == home_id) & (games_df["away_id"] == away_id)) |
        ((games_df["home_id"] == away_id) & (games_df["away_id"] == home_id))
    ]
    if h2h_games.empty:
        h2h_pct = 0.5
    else:
        home_wins = (
            ((h2h_games["home_id"] == home_id) & (h2h_games["home_score"] > h2h_games["away_score"])) |
            ((h2h_games["away_id"] == home_id) & (h2h_games["away_score"] > h2h_games["home_score"]))
        ).sum()
        h2h_pct = home_wins / len(h2h_games)

    features = {
        "home_win_pct_10":  h["win_pct_10"],
        "home_win_pct_20":  h["win_pct_20"],
        "away_win_pct_10":  a["win_pct_10"],
        "away_win_pct_20":  a["win_pct_20"],
        "home_run_diff_15": h["run_diff_15"],
        "away_run_diff_15": a["run_diff_15"],
        "home_streak":      h["streak"],
        "away_streak":      a["streak"],
        "home_rest_days":   h["rest_days"],
        "away_rest_days":   a["rest_days"],
        "h2h_win_pct":      h2h_pct,
        "home_sp_era":      4.00,   # no SP selected — use league average
        "home_sp_whip":     1.25,
        "away_sp_era":      4.00,
        "away_sp_whip":     1.25,
        "home_advantage":   1,
    }
    return pd.DataFrame([features])


# ── Pybaseball schedule path (fallback) ──────────────────────────────────────

def build_game_features(
    home_schedule: pd.DataFrame,
    away_schedule: pd.DataFrame,
    pitching_df: pd.DataFrame,
    home_sp: str = "",
    away_sp: str = "",
    home_team_name: str = "",
    away_team_name: str = "",
) -> pd.DataFrame:
    """Build a single-row feature DataFrame from pybaseball schedule data."""
    home_s = add_rolling_win_pct(home_schedule)
    home_s = add_run_differential(home_s)
    home_s = add_win_streak(home_s)

    away_s = add_rolling_win_pct(away_schedule)
    away_s = add_run_differential(away_s)
    away_s = add_win_streak(away_s)

    home_sp_f = get_sp_features(pitching_df, home_sp) if home_sp else {"sp_era": 4.0, "sp_whip": 1.25}
    away_sp_f = get_sp_features(pitching_df, away_sp) if away_sp else {"sp_era": 4.0, "sp_whip": 1.25}
    h2h = compute_h2h(home_schedule, away_team_name) if away_team_name else 0.5

    features = {
        "home_win_pct_10":  home_s["rolling_win_pct"].iloc[-1]    if not home_s.empty else 0.5,
        "home_win_pct_20":  home_s["rolling_win_pct_20"].iloc[-1] if not home_s.empty else 0.5,
        "away_win_pct_10":  away_s["rolling_win_pct"].iloc[-1]    if not away_s.empty else 0.5,
        "away_win_pct_20":  away_s["rolling_win_pct_20"].iloc[-1] if not away_s.empty else 0.5,
        "home_run_diff_15": home_s["rolling_run_diff"].iloc[-1]   if not home_s.empty else 0.0,
        "away_run_diff_15": away_s["rolling_run_diff"].iloc[-1]   if not away_s.empty else 0.0,
        "home_streak":      home_s["win_streak"].iloc[-1]         if not home_s.empty else 0.0,
        "away_streak":      away_s["win_streak"].iloc[-1]         if not away_s.empty else 0.0,
        "home_rest_days":   1.0,
        "away_rest_days":   1.0,
        "h2h_win_pct":      h2h,
        "home_sp_era":      home_sp_f["sp_era"],
        "home_sp_whip":     home_sp_f["sp_whip"],
        "away_sp_era":      away_sp_f["sp_era"],
        "away_sp_whip":     away_sp_f["sp_whip"],
        "home_advantage":   1,
    }
    return pd.DataFrame([features])


def build_training_dataset(
    schedules: dict,
    pitching_df: pd.DataFrame,
) -> pd.DataFrame:
    """Build training data from pybaseball schedule dict (fallback path)."""
    rows = []

    for home_code, home_sched in schedules.items():
        home_sched = add_rolling_win_pct(home_sched)
        home_sched = add_run_differential(home_sched)
        home_sched = add_win_streak(home_sched)

        for _, game in home_sched.iterrows():
            opp_code = str(game.get("Opp", "")).strip()
            if opp_code not in schedules:
                continue
            if str(game.get("Home_Away", "")).strip() == "@":
                continue

            away_sched = add_rolling_win_pct(schedules[opp_code])
            away_sched = add_run_differential(away_sched)
            away_sched = add_win_streak(away_sched)

            label = 1 if str(game.get("W/L", "")).startswith("W") else 0

            rows.append({
                "home_win_pct_10":  game.get("rolling_win_pct", 0.5),
                "home_win_pct_20":  game.get("rolling_win_pct_20", 0.5),
                "away_win_pct_10":  away_sched["rolling_win_pct"].iloc[-1]    if not away_sched.empty else 0.5,
                "away_win_pct_20":  away_sched["rolling_win_pct_20"].iloc[-1] if not away_sched.empty else 0.5,
                "home_run_diff_15": game.get("rolling_run_diff", 0.0),
                "away_run_diff_15": away_sched["rolling_run_diff"].iloc[-1]   if not away_sched.empty else 0.0,
                "home_streak":      game.get("win_streak", 0),
                "away_streak":      away_sched["win_streak"].iloc[-1]         if not away_sched.empty else 0.0,
                "home_rest_days":   1.0,
                "away_rest_days":   1.0,
                "h2h_win_pct":      compute_h2h(home_sched, opp_code),
                "home_sp_era":      np.nan,
                "home_sp_whip":     np.nan,
                "away_sp_era":      np.nan,
                "away_sp_whip":     np.nan,
                "home_advantage":   1,
                "label":            label,
            })

    df = pd.DataFrame(rows)
    for col in ["home_sp_era", "home_sp_whip", "away_sp_era", "away_sp_whip"]:
        df[col] = df[col].fillna(4.0)
    df["home_sp_whip"] = df["home_sp_whip"].fillna(1.25)
    df["away_sp_whip"] = df["away_sp_whip"].fillna(1.25)
    return df
