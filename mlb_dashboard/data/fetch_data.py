"""
data/fetch_data.py
==================
All pybaseball data-fetching helpers.
Results are cached as CSVs in data/cache/ to avoid repeated API calls.
"""

import os
import pandas as pd

# MLB Stats API team ID lookup (code → numeric ID used by statsapi)
TEAM_ID_MAP = {
    "ARI": 109, "ATL": 144, "BAL": 110, "BOS": 111, "CHC": 112,
    "CWS": 145, "CIN": 113, "CLE": 114, "COL": 115, "DET": 116,
    "HOU": 117, "KCR": 118, "LAA": 108, "LAD": 119, "MIA": 146,
    "MIL": 158, "MIN": 142, "NYM": 121, "NYY": 147, "OAK": 133,
    "PHI": 143, "PIT": 134, "SDP": 135, "SFG": 137, "SEA": 136,
    "STL": 138, "TBR": 139, "TEX": 140, "TOR": 141, "WSN": 120,
}
from pybaseball import (
    playerid_lookup,
    statcast_batter,
    statcast_pitcher,
    batting_stats,
    pitching_stats,
    schedule_and_record,
    team_batting,
    team_pitching,
)

CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")
os.makedirs(CACHE_DIR, exist_ok=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _cache_path(name: str) -> str:
    return os.path.join(CACHE_DIR, f"{name}.csv")


def _load_or_fetch(cache_name: str, fetch_fn, *args, **kwargs) -> pd.DataFrame:
    """Return cached CSV if it exists, otherwise call fetch_fn and cache the result."""
    path = _cache_path(cache_name)
    if os.path.exists(path):
        return pd.read_csv(path)
    df = fetch_fn(*args, **kwargs)
    df.to_csv(path, index=False)
    return df


# ── Player lookup ─────────────────────────────────────────────────────────────

def get_player_id(last: str, first: str) -> int | None:
    """
    Return the MLBAM player ID for a given player name.
    Returns None if the player is not found.
    """
    result = playerid_lookup(last, first)
    if result.empty:
        return None
    return int(result.iloc[0]["key_mlbam"])


def search_players(last: str) -> pd.DataFrame:
    """Look up all players with a given last name."""
    return playerid_lookup(last)


# ── Statcast data ─────────────────────────────────────────────────────────────

def fetch_batter_statcast(
    player_id: int,
    start_date: str,
    end_date: str,
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Fetch pitch-by-pitch Statcast data for a batter.

    Args:
        player_id: MLBAM player ID (int)
        start_date: 'YYYY-MM-DD'
        end_date:   'YYYY-MM-DD'
        use_cache:  Load from local CSV if available

    Returns:
        DataFrame with columns including exit_velocity, launch_angle,
        estimated_ba_using_speedangle, events, etc.
    """
    cache_name = f"batter_{player_id}_{start_date}_{end_date}"
    if use_cache:
        return _load_or_fetch(
            cache_name, statcast_batter, start_date, end_date, player_id=player_id
        )
    return statcast_batter(start_date, end_date, player_id=player_id)


def fetch_pitcher_statcast(
    player_id: int,
    start_date: str,
    end_date: str,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Fetch pitch-by-pitch Statcast data for a pitcher."""
    cache_name = f"pitcher_{player_id}_{start_date}_{end_date}"
    if use_cache:
        return _load_or_fetch(
            cache_name, statcast_pitcher, start_date, end_date, player_id=player_id
        )
    return statcast_pitcher(start_date, end_date, player_id=player_id)


# ── Season stats ──────────────────────────────────────────────────────────────

def fetch_batting_stats(year: int, use_cache: bool = True) -> pd.DataFrame:
    """
    Full FanGraphs season batting stats for all qualified players.
    Includes BA, OBP, SLG, wOBA, wRC+, WAR, etc.
    """
    cache_name = f"batting_stats_{year}"
    if use_cache:
        return _load_or_fetch(cache_name, batting_stats, year)
    return batting_stats(year)


def fetch_pitching_stats(year: int, use_cache: bool = True) -> pd.DataFrame:
    """Full FanGraphs season pitching stats for all qualified pitchers."""
    cache_name = f"pitching_stats_{year}"
    if use_cache:
        return _load_or_fetch(cache_name, pitching_stats, year)
    return pitching_stats(year)


# ── Team data ─────────────────────────────────────────────────────────────────

def fetch_schedule(year: int, team_code: str, use_cache: bool = True) -> pd.DataFrame:
    """
    Fetch full season schedule and W/L record for a team.
    team_code examples: 'NYY', 'LAD', 'BOS'
    """
    cache_name = f"schedule_{year}_{team_code}"
    if use_cache:
        return _load_or_fetch(
            cache_name, schedule_and_record, year, team_code
        )
    return schedule_and_record(year, team_code)


def fetch_team_batting(year: int, use_cache: bool = True) -> pd.DataFrame:
    """Aggregated team batting stats."""
    cache_name = f"team_batting_{year}"
    if use_cache:
        return _load_or_fetch(cache_name, team_batting, year)
    return team_batting(year)


def fetch_team_pitching(year: int, use_cache: bool = True) -> pd.DataFrame:
    """Aggregated team pitching stats."""
    cache_name = f"team_pitching_{year}"
    if use_cache:
        return _load_or_fetch(cache_name, team_pitching, year)
    return team_pitching(year)


# ── Convenience wrapper ───────────────────────────────────────────────────────

def fetch_player_season_stats(last: str, first: str, year: int) -> dict:
    """
    Return a dict with:
      - 'statcast': pitch-by-pitch Statcast DataFrame
      - 'batting':  FanGraphs batting row for this player
    """
    player_id = get_player_id(last, first)
    if player_id is None:
        raise ValueError(f"Player '{first} {last}' not found.")

    start = f"{year}-03-28"
    end = f"{year}-10-01"

    statcast_df = fetch_batter_statcast(player_id, start, end)
    batting_df = fetch_batting_stats(year)

    # Filter batting row to this player
    player_name = f"{first.title()} {last.title()}"
    batting_row = batting_df[batting_df["Name"].str.contains(last.title(), na=False)]

    return {
        "player_id": player_id,
        "statcast": statcast_df,
        "batting": batting_row,
    }


# ── MLB Stats API (official, no scraping, no rate-limiting) ──────────────────

def fetch_mlb_games(year: int, use_cache: bool = True) -> pd.DataFrame:
    """
    Fetch all regular-season final games for a year from the official MLB Stats API.
    Returns columns: game_id, game_date, home_id, away_id, home_name, away_name,
                     home_score, away_score, home_sp, away_sp, venue.
    Much more reliable than scraping Baseball Reference.
    The current calendar year always bypasses the cache so live season data stays fresh.
    """
    import datetime
    import statsapi

    current_year = datetime.date.today().year
    cache_name = f"mlb_games_{year}"
    path = _cache_path(cache_name)
    if use_cache and year != current_year and os.path.exists(path):
        df = pd.read_csv(path)
        df["game_date"] = pd.to_datetime(df["game_date"])
        return df

    month_ranges = [
        (f"{year}-03-20", f"{year}-03-31"),
        (f"{year}-04-01", f"{year}-04-30"),
        (f"{year}-05-01", f"{year}-05-31"),
        (f"{year}-06-01", f"{year}-06-30"),
        (f"{year}-07-01", f"{year}-07-31"),
        (f"{year}-08-01", f"{year}-08-31"),
        (f"{year}-09-01", f"{year}-09-30"),
        (f"{year}-10-01", f"{year}-10-31"),
    ]

    all_games, seen_ids = [], set()
    for start, end in month_ranges:
        try:
            games = statsapi.schedule(start_date=start, end_date=end, sportId=1)
            for g in games:
                if (g.get("game_type") == "R"
                        and g.get("status") == "Final"
                        and g["game_id"] not in seen_ids):
                    seen_ids.add(g["game_id"])
                    all_games.append({
                        "game_id":    g["game_id"],
                        "game_date":  g["game_date"],
                        "home_id":    g["home_id"],
                        "away_id":    g["away_id"],
                        "home_name":  g["home_name"],
                        "away_name":  g["away_name"],
                        "home_score": int(g.get("home_score") or 0),
                        "away_score": int(g.get("away_score") or 0),
                        "home_sp":    g.get("home_probable_pitcher", ""),
                        "away_sp":    g.get("away_probable_pitcher", ""),
                        "venue":      g.get("venue_name", ""),
                    })
        except Exception as e:
            print(f"  MLB API: {start}–{end} failed ({e})")

    df = pd.DataFrame(all_games)
    if not df.empty:
        df["game_date"] = pd.to_datetime(df["game_date"])
        df = df.sort_values("game_date").reset_index(drop=True)
        df.to_csv(path, index=False)
    return df
