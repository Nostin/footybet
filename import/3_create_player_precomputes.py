# build_player_precomputes_basic.py
import pandas as pd
import numpy as np
from sqlalchemy import text
from db_connect import get_engine

engine = get_engine()

# Home-ground lookup (keep whatever naming your data actually uses)
HOME_GROUNDS = {
    'Adelaide': ['Adelaide'],
    'Brisbane': ['Brisbane'],
    'Carlton': ['MCG', 'Docklands'],
    'Collingwood': ['MCG'],
    'Dees': ['MCG'],
    'Essendon': ['Docklands'],
    'Fremantle': ['Perth'],
    'Geelong': ['Geelong'],
    'Suns': ['Gold Coast', 'Darwin'],
    'GWS': ['Engie', 'Canberra'],
    'Hawthorn': ['MCG', 'Launceston'],
    'Norf': ['Docklands', 'Hobart'],
    'Port': ['Adelaide'],
    'Richmond': ['MCG'],
    'Saints': ['Docklands'],
    'Sydney': ['SCG'],
    'Eagles': ['Perth'],
    'Bulldogs': ['Docklands', 'Ballarat'],
}

# Optional: map canonical names -> keys used in HOME_GROUNDS (helps if your DB uses official names)
ALIASES = {
    'North Melbourne': 'Norf',
    'Port Adelaide': 'Port',
    'West Coast': 'Eagles',
    'West Coast Eagles': 'Eagles',
    'Gold Coast': 'Suns',
    'Gold Coast Suns': 'Suns',
    'Melbourne': 'Dees',
    'Western Bulldogs': 'Bulldogs',
    'St Kilda': 'Saints',
    'GWS Giants': 'GWS',
    'Greater Western Sydney': 'GWS',
}

def normalize_team_key(team: str) -> str:
    t = (team or "").strip()
    return t if t in HOME_GROUNDS else ALIASES.get(t, t)

# --- Load upcoming games and normalise strings ---
upcoming_games_df = pd.read_sql('SELECT * FROM upcoming_games', engine)
for col in ["Home Team", "Away Team", "Venue", "Timeslot"]:
    if col in upcoming_games_df.columns:
        upcoming_games_df[col] = upcoming_games_df[col].astype(str).str.strip()
upcoming_games_df["Date"] = pd.to_datetime(upcoming_games_df["Date"], errors="coerce")

# --- Load full player stats ---
df = pd.read_sql('SELECT * FROM player_stats ORDER BY "Date" DESC', engine)
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

# Pick the latest year present as "this season"
season_year = int(df['Date'].dt.year.max())
df_season = df[df['Date'].dt.year == season_year].copy()

# Convenience
TOG_COL = 'Time on Ground %'

precomputes = []

for player in df['Player'].dropna().unique():
    row = {
        "Player": player,
        "Team": "",
        "Games_This_Season": 0,
        "Next_Opponent": "",
        "Next_Venue": "",
        "Next_Venue_Home": 0,
        "Next_Timeslot": "",
        "Days_since_last_game": np.nan,
        "ToG_Season_Avg": np.nan,
        "ToG_Last": np.nan,
        "Missed_Game_Time": False,
    }

    # Per-player slices
    p_all = df[df['Player'] == player].copy().sort_values('Date')
    p_season = df_season[df_season['Player'] == player].copy().sort_values('Date')

    # Team: prefer this season, else latest ever
    if not p_season.empty:
        team = str(p_season['Team'].iloc[-1]).strip()
    elif not p_all.empty:
        team = str(p_all['Team'].iloc[-1]).strip()
    else:
        team = ""
    row["Team"] = team
    team_key = normalize_team_key(team)

    # Games this season
    row["Games_This_Season"] = int(len(p_season))

    # Next game + days since last game
    if team:
        last_game_date = p_all['Date'].max() if not p_all.empty else pd.NaT
        mask_team_next = (upcoming_games_df["Home Team"] == team) | (upcoming_games_df["Away Team"] == team)
        next_games = upcoming_games_df[mask_team_next].copy()
        if pd.notna(last_game_date):
            next_games = next_games[next_games["Date"] > last_game_date]
        next_games = next_games.sort_values("Date")

        if not next_games.empty:
            ng = next_games.iloc[0]
            row["Next_Venue"] = str(ng["Venue"])
            row["Next_Timeslot"] = str(ng["Timeslot"])
            row["Next_Opponent"] = str(ng["Away Team"] if ng["Home Team"] == team else ng["Home Team"])

            if pd.notna(last_game_date):
                row["Days_since_last_game"] = int((ng["Date"] - last_game_date).days)

    # Next_Venue_Home flag
    if row["Next_Venue"]:
        row["Next_Venue_Home"] = int(row["Next_Venue"] in HOME_GROUNDS.get(team_key, []))

    # Time-on-ground (this season)
    if not p_season.empty and TOG_COL in p_season.columns:
        row["ToG_Season_Avg"] = float(p_season[TOG_COL].mean())
        row["ToG_Last"] = float(p_season[TOG_COL].iloc[-1]) if not p_season[TOG_COL].empty else np.nan

    # Missed game time in last 4 team matches (absent OR ToG < 50)
    if team:
        team_games = (
            df[df['Team'] == team]
            .drop_duplicates(subset='Date')
            .sort_values('Date')
            .tail(4)
        )
        missed = False
        for gdate in team_games['Date']:
            played_mask = (p_all['Date'] == gdate) & (p_all['Team'] == team)
            if not played_mask.any():
                missed = True
                break
            tog_val = p_all.loc[played_mask, TOG_COL].iloc[0] if TOG_COL in p_all.columns else np.nan
            if pd.isna(tog_val) or float(tog_val) < 50:
                missed = True
                break
        row["Missed_Game_Time"] = bool(missed)

    precomputes.append(row)

# --- Build DataFrame with ONLY the requested columns ---
cols = [
    "Player",
    "Team",
    "Games_This_Season",
    "Next_Opponent",
    "Next_Venue",
    "Next_Venue_Home",
    "Next_Timeslot",
    "Days_since_last_game",
    "ToG_Season_Avg",
    "ToG_Last",
    "Missed_Game_Time",
]
df_pre = pd.DataFrame(precomputes, columns=cols)

# --- Write/replace table and add helpful indexes ---
df_pre.to_sql("player_precomputes", engine, if_exists="replace", index=False)
with engine.connect() as conn:
    conn.execute(text('CREATE INDEX IF NOT EXISTS idx_player_precomputes_player ON player_precomputes ("Player")'))
    conn.execute(text('CREATE INDEX IF NOT EXISTS idx_player_precomputes_team ON player_precomputes ("Team")'))

print(f"✅ player_precomputes table written for season {season_year} with {len(df_pre)} players")
