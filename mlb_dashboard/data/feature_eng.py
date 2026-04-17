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
    # Rolling win percentages (3 windows)
    "home_win_pct_5",    # last 5 games — hot/cold form
    "home_win_pct_10",   # last 10 games
    "home_win_pct_20",   # last 20 games
    "away_win_pct_5",
    "away_win_pct_10",
    "away_win_pct_20",
    # Rolling run differential (last 15 games)
    "home_run_diff_15",
    "away_run_diff_15",
    # Pythagorean win% (expected W% from runs scored/allowed — more stable than actual W%)
    "home_pyth_wp",
    "away_pyth_wp",
    # Win/loss streaks
    "home_streak",
    "away_streak",
    # Days rest since last game
    "home_rest_days",
    "away_rest_days",
    # Head-to-head win % (home team vs this opponent, season so far)
    "h2h_win_pct",
    # Starting pitcher quality
    "home_sp_era",
    "home_sp_whip",
    "away_sp_era",
    "away_sp_whip",
    # Elo ratings — continuous team-quality tracker, updated after every game
    "home_elo",
    "away_elo",
    "elo_diff",          # home_elo − away_elo (+ve = home team stronger)
    # Park run-scoring factor for the home venue
    "park_factor",
    # Differential features — give XGBoost direct relative advantage signals
    "win_pct_diff_10",   # home_win_pct_10 − away_win_pct_10
    "run_diff_diff",     # home_run_diff_15 − away_run_diff_15
    "streak_diff",       # home_streak − away_streak
    "era_diff",          # away_sp_era − home_sp_era (+ve = home pitcher advantage)
]


# ── Elo constants ─────────────────────────────────────────────────────────────

ELO_K     = 6      # conservative K-factor (baseball has high game-to-game variance)
ELO_START = 1500   # default rating
ELO_RESET = 0.5    # fraction of deviation from 1500 that carries over to next season


# ── Park factors (multi-year FanGraphs averages, 1.00 = league neutral) ──────
# > 1.00: hitter-friendly  |  < 1.00: pitcher-friendly
# Indexed by MLB Stats API team ID (matches TEAM_ID_MAP in fetch_data.py)

PARK_FACTORS: dict[int, float] = {
    109: 1.02,  # ARI - Chase Field
    144: 0.94,  # ATL - Truist Park
    110: 0.97,  # BAL - Camden Yards
    111: 1.04,  # BOS - Fenway Park
    112: 0.98,  # CHC - Wrigley Field
    145: 0.98,  # CWS - Guaranteed Rate Field
    113: 1.00,  # CIN - Great American Ball Park
    114: 0.97,  # CLE - Progressive Field
    115: 1.13,  # COL - Coors Field (altitude — single biggest factor in MLB)
    116: 0.95,  # DET - Comerica Park
    117: 0.99,  # HOU - Minute Maid Park
    118: 0.99,  # KCR - Kauffman Stadium
    108: 0.99,  # LAA - Angel Stadium
    119: 0.97,  # LAD - Dodger Stadium
    146: 0.95,  # MIA - loanDepot park
    158: 1.01,  # MIL - American Family Field
    142: 1.01,  # MIN - Target Field
    121: 0.97,  # NYM - Citi Field
    147: 1.04,  # NYY - Yankee Stadium
    133: 0.94,  # OAK - Oakland Coliseum
    143: 1.00,  # PHI - Citizens Bank Park
    134: 0.96,  # PIT - PNC Park
    135: 0.94,  # SDP - Petco Park
    137: 0.93,  # SFG - Oracle Park
    136: 0.92,  # SEA - T-Mobile Park
    138: 0.97,  # STL - Busch Stadium
    139: 0.94,  # TBR - Tropicana Field
    140: 1.01,  # TEX - Globe Life Field
    141: 1.00,  # TOR - Rogers Centre
    120: 1.01,  # WSN - Nationals Park
}


TEAM_CODES = [
    "NYY", "BOS", "LAD", "HOU", "ATL", "CHC", "NYM", "SFG",
    "PHI", "TOR", "MIL", "ARI", "TEX", "SEA", "BAL", "MIN",
    "CLE", "TBR", "SDP", "CIN",
]


# ── SP stat lookup ────────────────────────────────────────────────────────────

def build_sp_lookup(pitching_df: pd.DataFrame) -> dict:
    """
    Build {normalized_name: {"era": float, "whip": float}} from a FanGraphs pitching DataFrame.
    Used to look up starting pitcher stats by name at prediction time.
    """
    lookup = {}
    if pitching_df is None or pitching_df.empty:
        return lookup
    for _, row in pitching_df.iterrows():
        name = str(row.get("Name", "")).strip().lower()
        if not name:
            continue
        era  = float(row["ERA"])  if "ERA"  in row and pd.notna(row["ERA"])  else 4.00
        whip = float(row["WHIP"]) if "WHIP" in row and pd.notna(row["WHIP"]) else 1.25
        lookup[name] = {"era": era, "whip": whip}
    return lookup


def lookup_sp_stats(name: str, sp_lookup: dict) -> tuple[float, float]:
    """
    Return (ERA, WHIP) for a pitcher name using the sp_lookup dict.
    Tries exact match first, then substring match for partial names (min 4 chars).
    Returns league-average defaults (4.00, 1.25) if not found.
    """
    if not name or not sp_lookup:
        return 4.00, 1.25
    normalized = name.strip().lower()
    if normalized in sp_lookup:
        s = sp_lookup[normalized]
        return s["era"], s["whip"]
    if len(normalized) >= 4:
        for key, s in sp_lookup.items():
            if normalized in key or key in normalized:
                return s["era"], s["whip"]
    return 4.00, 1.25


# ── Elo simulation ────────────────────────────────────────────────────────────

def _run_elo_simulation(games_df: pd.DataFrame) -> tuple[dict, dict]:
    """
    Run a full Elo simulation over games_df (must be sorted by game_date).

    Returns:
        elo_before  — {(team_id, date_str): elo_before_game}  used for training features
        final_elo   — {team_id: elo_after_last_game}           used for live prediction
    """
    games = games_df.sort_values("game_date").copy()
    elo: dict[int, float] = {}
    elo_before: dict[tuple, float] = {}
    current_year: int | None = None

    for _, g in games.iterrows():
        year     = pd.to_datetime(g["game_date"]).year
        home_id  = int(g["home_id"])
        away_id  = int(g["away_id"])
        date_str = str(pd.to_datetime(g["game_date"]).date())

        # Season reset: pull ratings 50% toward 1500 at the start of each new year
        if year != current_year:
            current_year = year
            for tid in list(elo.keys()):
                elo[tid] = ELO_START + ELO_RESET * (elo[tid] - ELO_START)

        h_elo = elo.get(home_id, ELO_START)
        a_elo = elo.get(away_id, ELO_START)

        # Record pre-game ratings for the training feature lookup
        elo_before[(home_id, date_str)] = h_elo
        elo_before[(away_id, date_str)] = a_elo

        h_expected  = 1.0 / (1.0 + 10.0 ** ((a_elo - h_elo) / 400.0))
        home_actual = 1 if int(g["home_score"]) > int(g["away_score"]) else 0

        elo[home_id] = h_elo + ELO_K * (home_actual       - h_expected)
        elo[away_id] = a_elo + ELO_K * ((1 - home_actual) - (1 - h_expected))

    return elo_before, elo


def compute_elo_ratings(games_df: pd.DataFrame) -> dict:
    """Return {(team_id, date_str): elo_before_game} for training feature lookup."""
    elo_before, _ = _run_elo_simulation(games_df)
    return elo_before


def get_current_elo(games_df: pd.DataFrame) -> dict:
    """Return {team_id: current_elo} after all games — for live predictions."""
    _, final_elo = _run_elo_simulation(games_df)
    return final_elo


# ── Pythagorean win% ──────────────────────────────────────────────────────────

def _pythagorean_wp(rs: float, ra: float, exp: float = 1.83) -> float:
    """Bill James Pythagorean win% — more stable than actual W/L over small samples."""
    denom = rs**exp + ra**exp
    return rs**exp / denom if denom > 0 else 0.5


# ── Helpers for pybaseball schedule DataFrames ───────────────────────────────

def add_rolling_win_pct(schedule_df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    """Add rolling_win_pct_5/10/20 columns."""
    df = schedule_df.copy()
    if "win" not in df.columns:
        df["win"] = (df["W/L"].str.startswith("W")).astype(int)
    df["rolling_win_pct_5"]  = df["win"].rolling(5,  min_periods=1).mean()
    df["rolling_win_pct"]    = df["win"].rolling(10, min_periods=1).mean()
    df["rolling_win_pct_20"] = df["win"].rolling(20, min_periods=1).mean()
    return df


def add_run_differential(schedule_df: pd.DataFrame, window: int = 15) -> pd.DataFrame:
    """Add rolling run-differential column."""
    df = schedule_df.copy()
    df["run_diff"]         = df["R"] - df["RA"]
    df["rolling_run_diff"] = df["run_diff"].rolling(window, min_periods=1).mean()
    return df


def add_win_streak(schedule_df: pd.DataFrame) -> pd.DataFrame:
    """Add win_streak column: +N = N-game win streak, -N = N-game loss streak."""
    df = schedule_df.copy()
    if "win" not in df.columns:
        df["win"] = (df["W/L"].str.startswith("W")).astype(int)
    streak, streaks = 0, []
    for won in df["win"]:
        streaks.append(streak)
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
    All stats use ONLY games played BEFORE each game to prevent data leakage.

    Returns: {(team_id: int, date: str) -> stats_dict}
    stats_dict keys: win_pct_5, win_pct_10, win_pct_20, run_diff_15,
                     pyth_wp, streak, rest_days
    """
    games_df = games_df.copy()
    games_df["game_date"] = pd.to_datetime(games_df["game_date"])

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

        # shift(1) ensures each position uses only prior games
        won_s = grp["won"].shift(1)
        rd_s  = grp["run_diff"].shift(1)
        rs_s  = grp["runs_scored"].shift(1)
        ra_s  = grp["runs_allowed"].shift(1)

        win_pct_5   = won_s.rolling(5,  min_periods=2).mean()
        win_pct_10  = won_s.rolling(10, min_periods=3).mean()
        win_pct_20  = won_s.rolling(20, min_periods=5).mean()
        run_diff_15 = rd_s.rolling(15,  min_periods=3).mean()
        rs_20       = rs_s.rolling(20,  min_periods=5).sum()
        ra_20       = ra_s.rolling(20,  min_periods=5).sum()

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
            rs  = rs_20.iloc[i]
            ra  = ra_20.iloc[i]
            result[key] = {
                "win_pct_5":   float(win_pct_5.iloc[i])   if not pd.isna(win_pct_5.iloc[i])   else 0.5,
                "win_pct_10":  float(win_pct_10.iloc[i])  if not pd.isna(win_pct_10.iloc[i])  else 0.5,
                "win_pct_20":  float(win_pct_20.iloc[i])  if not pd.isna(win_pct_20.iloc[i])  else 0.5,
                "run_diff_15": float(run_diff_15.iloc[i]) if not pd.isna(run_diff_15.iloc[i]) else 0.0,
                "pyth_wp":     _pythagorean_wp(
                                   float(rs) if not pd.isna(rs) else 0.0,
                                   float(ra) if not pd.isna(ra) else 0.0,
                               ),
                "streak":      float(streaks[i]),
                "rest_days":   float(rest.iloc[i]),
            }

    return result


def build_training_dataset_mlb(
    games_df: pd.DataFrame,
    pitching_df: pd.DataFrame = None,
    pitching_by_year: dict | None = None,
) -> pd.DataFrame:
    """
    Build a training DataFrame from MLB Stats API game data.
    Each row is one game with features derived strictly from prior games only.

    pitching_by_year: {year: sp_lookup_dict} — if provided, real SP ERA/WHIP is used
                      per game instead of league-average defaults.
    """
    if games_df.empty:
        return pd.DataFrame()

    games_df = games_df.copy()
    games_df["game_date"] = pd.to_datetime(games_df["game_date"])
    games_df = games_df.sort_values("game_date").reset_index(drop=True)

    team_stats = compute_team_rolling_stats_mlb(games_df)
    elo_map    = compute_elo_ratings(games_df)

    h2h_wins:  dict[tuple, int]   = {}
    h2h_total: dict[tuple, int]   = {}

    rows = []
    for _, game in games_df.iterrows():
        home_id  = int(game["home_id"])
        away_id  = int(game["away_id"])
        date_str = str(game["game_date"].date())

        h = team_stats.get((home_id, date_str))
        a = team_stats.get((away_id, date_str))

        home_won = int(game["home_score"]) > int(game["away_score"])

        # Update H2H tracker before reading, then subtract current game below
        h2h_wins[(home_id, away_id)]  = h2h_wins.get((home_id, away_id), 0) + (1 if home_won else 0)
        h2h_total[(home_id, away_id)] = h2h_total.get((home_id, away_id), 0) + 1

        if h is None or a is None:
            continue

        total_h2h = h2h_total.get((home_id, away_id), 0) - 1
        wins_h2h  = h2h_wins.get((home_id, away_id), 0)  - (1 if home_won else 0)
        h2h_pct   = wins_h2h / total_h2h if total_h2h > 0 else 0.5

        home_elo = elo_map.get((home_id, date_str), ELO_START)
        away_elo = elo_map.get((away_id, date_str), ELO_START)

        year_lookup = pitching_by_year.get(game["game_date"].year, {}) if pitching_by_year else {}
        h_era, h_whip = lookup_sp_stats(str(game.get("home_sp", "")), year_lookup)
        a_era, a_whip = lookup_sp_stats(str(game.get("away_sp", "")), year_lookup)

        rows.append({
            "home_win_pct_5":   h["win_pct_5"],
            "home_win_pct_10":  h["win_pct_10"],
            "home_win_pct_20":  h["win_pct_20"],
            "away_win_pct_5":   a["win_pct_5"],
            "away_win_pct_10":  a["win_pct_10"],
            "away_win_pct_20":  a["win_pct_20"],
            "home_run_diff_15": h["run_diff_15"],
            "away_run_diff_15": a["run_diff_15"],
            "home_pyth_wp":     h["pyth_wp"],
            "away_pyth_wp":     a["pyth_wp"],
            "home_streak":      h["streak"],
            "away_streak":      a["streak"],
            "home_rest_days":   h["rest_days"],
            "away_rest_days":   a["rest_days"],
            "h2h_win_pct":      h2h_pct,
            "home_sp_era":      h_era,
            "home_sp_whip":     h_whip,
            "away_sp_era":      a_era,
            "away_sp_whip":     a_whip,
            "home_elo":         home_elo,
            "away_elo":         away_elo,
            "elo_diff":         home_elo - away_elo,
            "park_factor":      PARK_FACTORS.get(home_id, 1.00),
            "win_pct_diff_10":  h["win_pct_10"] - a["win_pct_10"],
            "run_diff_diff":    h["run_diff_15"] - a["run_diff_15"],
            "streak_diff":      h["streak"] - a["streak"],
            "era_diff":         np.nan,
            "label":            int(home_won),
        })

    df = pd.DataFrame(rows)
    df["era_diff"] = df["away_sp_era"] - df["home_sp_era"]
    return df


def build_prediction_features_mlb(
    home_id: int,
    away_id: int,
    games_df: pd.DataFrame,
    home_sp: str = "",
    away_sp: str = "",
    sp_lookup: dict | None = None,
) -> pd.DataFrame:
    """
    Build a single-row feature DataFrame for a matchup using MLB API data.
    Uses the most recent available stats for each team.
    Elo uses post-game ratings (current team strength, not pre-game from last matchup).

    home_sp / away_sp: probable pitcher names (e.g. "Gerrit Cole").
    sp_lookup: output of build_sp_lookup() for the current season.
    """
    team_stats  = compute_team_rolling_stats_mlb(games_df)
    current_elo = get_current_elo(games_df)

    home_dates = sorted(d for (tid, d) in team_stats if tid == home_id)
    away_dates = sorted(d for (tid, d) in team_stats if tid == away_id)

    if not home_dates or not away_dates:
        raise ValueError("Not enough game history for one or both teams.")

    h = team_stats[(home_id, home_dates[-1])]
    a = team_stats[(away_id, away_dates[-1])]

    home_elo = current_elo.get(home_id, ELO_START)
    away_elo = current_elo.get(away_id, ELO_START)

    h_era, h_whip = lookup_sp_stats(home_sp, sp_lookup or {})
    a_era, a_whip = lookup_sp_stats(away_sp, sp_lookup or {})

    h2h_games = games_df[
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
        "home_win_pct_5":   h["win_pct_5"],
        "home_win_pct_10":  h["win_pct_10"],
        "home_win_pct_20":  h["win_pct_20"],
        "away_win_pct_5":   a["win_pct_5"],
        "away_win_pct_10":  a["win_pct_10"],
        "away_win_pct_20":  a["win_pct_20"],
        "home_run_diff_15": h["run_diff_15"],
        "away_run_diff_15": a["run_diff_15"],
        "home_pyth_wp":     h["pyth_wp"],
        "away_pyth_wp":     a["pyth_wp"],
        "home_streak":      h["streak"],
        "away_streak":      a["streak"],
        "home_rest_days":   h["rest_days"],
        "away_rest_days":   a["rest_days"],
        "h2h_win_pct":      h2h_pct,
        "home_sp_era":      h_era,
        "home_sp_whip":     h_whip,
        "away_sp_era":      a_era,
        "away_sp_whip":     a_whip,
        "home_elo":         home_elo,
        "away_elo":         away_elo,
        "elo_diff":         home_elo - away_elo,
        "park_factor":      PARK_FACTORS.get(home_id, 1.00),
        "win_pct_diff_10":  h["win_pct_10"] - a["win_pct_10"],
        "run_diff_diff":    h["run_diff_15"] - a["run_diff_15"],
        "streak_diff":      h["streak"] - a["streak"],
        "era_diff":         0.0,  # symmetric default when SP data unavailable
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

    home_wp10 = home_s["rolling_win_pct"].iloc[-1]    if not home_s.empty else 0.5
    away_wp10 = away_s["rolling_win_pct"].iloc[-1]    if not away_s.empty else 0.5
    home_rd   = home_s["rolling_run_diff"].iloc[-1]   if not home_s.empty else 0.0
    away_rd   = away_s["rolling_run_diff"].iloc[-1]   if not away_s.empty else 0.0
    home_str  = home_s["win_streak"].iloc[-1]         if not home_s.empty else 0.0
    away_str  = away_s["win_streak"].iloc[-1]         if not away_s.empty else 0.0
    home_era  = home_sp_f["sp_era"]  if not np.isnan(home_sp_f["sp_era"])  else 4.0
    away_era  = away_sp_f["sp_era"]  if not np.isnan(away_sp_f["sp_era"])  else 4.0

    features = {
        "home_win_pct_5":   home_s["rolling_win_pct_5"].iloc[-1]  if not home_s.empty else 0.5,
        "home_win_pct_10":  home_wp10,
        "home_win_pct_20":  home_s["rolling_win_pct_20"].iloc[-1] if not home_s.empty else 0.5,
        "away_win_pct_5":   away_s["rolling_win_pct_5"].iloc[-1]  if not away_s.empty else 0.5,
        "away_win_pct_10":  away_wp10,
        "away_win_pct_20":  away_s["rolling_win_pct_20"].iloc[-1] if not away_s.empty else 0.5,
        "home_run_diff_15": home_rd,
        "away_run_diff_15": away_rd,
        "home_pyth_wp":     0.5,  # not computable without raw run totals from schedule
        "away_pyth_wp":     0.5,
        "home_streak":      home_str,
        "away_streak":      away_str,
        "home_rest_days":   1.0,
        "away_rest_days":   1.0,
        "h2h_win_pct":      h2h,
        "home_sp_era":      home_era,
        "home_sp_whip":     home_sp_f["sp_whip"] if not np.isnan(home_sp_f["sp_whip"]) else 1.25,
        "away_sp_era":      away_era,
        "away_sp_whip":     away_sp_f["sp_whip"] if not np.isnan(away_sp_f["sp_whip"]) else 1.25,
        "home_elo":         float(ELO_START),
        "away_elo":         float(ELO_START),
        "elo_diff":         0.0,
        "park_factor":      1.00,
        "win_pct_diff_10":  home_wp10 - away_wp10,
        "run_diff_diff":    home_rd - away_rd,
        "streak_diff":      home_str - away_str,
        "era_diff":         away_era - home_era,
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

            home_wp10 = game.get("rolling_win_pct", 0.5)
            away_wp10 = away_sched["rolling_win_pct"].iloc[-1] if not away_sched.empty else 0.5
            home_rd   = game.get("rolling_run_diff", 0.0)
            away_rd   = away_sched["rolling_run_diff"].iloc[-1] if not away_sched.empty else 0.0
            home_str  = game.get("win_streak", 0)
            away_str  = away_sched["win_streak"].iloc[-1] if not away_sched.empty else 0.0

            rows.append({
                "home_win_pct_5":   game.get("rolling_win_pct_5", 0.5),
                "home_win_pct_10":  home_wp10,
                "home_win_pct_20":  game.get("rolling_win_pct_20", 0.5),
                "away_win_pct_5":   away_sched["rolling_win_pct_5"].iloc[-1]  if not away_sched.empty else 0.5,
                "away_win_pct_10":  away_wp10,
                "away_win_pct_20":  away_sched["rolling_win_pct_20"].iloc[-1] if not away_sched.empty else 0.5,
                "home_run_diff_15": home_rd,
                "away_run_diff_15": away_rd,
                "home_pyth_wp":     0.5,
                "away_pyth_wp":     0.5,
                "home_streak":      home_str,
                "away_streak":      away_str,
                "home_rest_days":   1.0,
                "away_rest_days":   1.0,
                "h2h_win_pct":      compute_h2h(home_sched, opp_code),
                "home_sp_era":      np.nan,
                "home_sp_whip":     np.nan,
                "away_sp_era":      np.nan,
                "away_sp_whip":     np.nan,
                "home_elo":         float(ELO_START),
                "away_elo":         float(ELO_START),
                "elo_diff":         0.0,
                "park_factor":      1.00,
                "win_pct_diff_10":  home_wp10 - away_wp10,
                "run_diff_diff":    home_rd - away_rd,
                "streak_diff":      float(home_str) - float(away_str),
                "era_diff":         np.nan,
                "label":            label,
            })

    df = pd.DataFrame(rows)
    for col in ["home_sp_era", "away_sp_era"]:
        df[col] = df[col].fillna(4.00)
    for col in ["home_sp_whip", "away_sp_whip"]:
        df[col] = df[col].fillna(1.25)
    df["era_diff"] = df["away_sp_era"] - df["home_sp_era"]
    return df
