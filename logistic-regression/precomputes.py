import pandas as pd
from sqlalchemy import create_engine, text
from db_connect import get_engine
import numpy as np

engine = get_engine()

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

# need this to work out days since last game
upcoming_games_df = pd.read_sql('SELECT * FROM upcoming_games', engine)
for col in ["Home Team", "Away Team", "Venue", "Timeslot"]:
    if col in upcoming_games_df.columns:
        upcoming_games_df[col] = upcoming_games_df[col].astype(str).str.strip()
upcoming_games_df["Date"] = pd.to_datetime(upcoming_games_df["Date"], errors="coerce")

df = pd.read_sql('SELECT * FROM player_stats ORDER BY "Date" DESC', engine)
df['Date'] = pd.to_datetime(df['Date'])
df_2025 = df[df['Date'] >= '2025-01-01']

def summary_stats(s):
    # avg = np.mean(s)
    # var = np.var(s, ddof=0)
    # cv = var / avg if avg > 0 else 0  # Coefficient of Variation

    # Cap disposals at 35 to reduce impact of outlier highs
    capped = s.clip(upper=35)

    avg = np.mean(capped)
    med = np.median(capped)
    high = np.max(capped)
    low = np.min(capped)
    var = np.var(capped, ddof=0)

    # Coefficient of Variation (unitless dispersion metric)
    cv = round(var / avg, 3) if avg > 0 else 0

    return pd.Series({
        'Avg': avg,
        'Median': np.median(s),
        'High': np.max(s),
        'Low': np.min(s),
        # 'Variance': var,
        'Variance': cv
    })

def compute_for_split(df, mask, prefix):
    sub = df[mask]
    if sub.empty:
        return {f"{prefix}_{k}": np.nan for k in ['Avg', 'Median', 'High', 'Low', 'Variance']}
    stats = summary_stats(sub['Disposals'])
    return {f"{prefix}_{k}": v for k, v in stats.items()}

def is_home_game(row):
    home_venues = HOME_GROUNDS.get(row['Team'], [])
    return row['Venue'] in home_venues

def home_away_mask(df):
    return df.apply(is_home_game, axis=1)

def last_n_by_condition(df, N, col, val):
    mask = df[col].str.lower() == val
    return df[mask].sort_values('Date').tail(N)

def last_n_by_timeslot(df, N, val):
    mask = df['Timeslot'].str.lower() == val
    return df[mask].sort_values('Date').tail(N)

def last_n_home_away(df, N, home=True):
    mask = df.apply(is_home_game, axis=1)
    if not home:
        mask = ~mask
    return df[mask].sort_values('Date').tail(N)

precomputes = []

for player in df['Player'].unique():
    row = {'Player': player}  # ✅ Start row dict early

    # 2025 and all games
    pgroup_2025 = df_2025[df_2025['Player'] == player]
    pgroup_all = df[df['Player'] == player]
    row["Games_This_Season"] = len(pgroup_2025)

    # Days between last game and next game
    # --- Find Next Opponent / Venue / Timeslot ---
    if not pgroup_all.empty:
        last_game_date = pgroup_all['Date'].max()

        # Get team from latest game
        team = (pgroup_2025['Team'].iloc[-1] if not pgroup_2025.empty else pgroup_all['Team'].iloc[-1]).strip()


        # Find next game where this team is home or away
        mask = (upcoming_games_df["Home Team"] == team) | (upcoming_games_df["Away Team"] == team)
        next_games = upcoming_games_df[mask]
        next_games = next_games[next_games["Date"] > last_game_date].sort_values("Date")

        if not next_games.empty:
            next_game = next_games.iloc[0]
            row["Next_Venue"] = next_game["Venue"]
            row["Next_Timeslot"] = next_game["Timeslot"]

            # Work out opponent
            if next_game["Home Team"] == team:
                row["Next_Opponent"] = next_game["Away Team"]
            else:
                row["Next_Opponent"] = next_game["Home Team"]

            # Also set days since last game (keep your existing behaviour)
            row["Days_since_last_game"] = (next_game["Date"] - last_game_date).days
        else:
            row["Next_Opponent"] = ''
            row["Next_Venue"] = ''
            row["Next_Timeslot"] = ''
            row["Days_since_last_game"] = np.nan
    else:
        row["Next_Opponent"] = ''
        row["Next_Venue"] = ''
        row["Next_Timeslot"] = ''
        row["Days_since_last_game"] = np.nan

    # ToG
    tog_col = 'Time on Ground %'
    if not pgroup_2025.empty:
        row['ToG_Season_Avg'] = pgroup_2025[tog_col].mean()
        row['ToG_Last'] = pgroup_2025.sort_values('Date')[tog_col].iloc[-1]
    else:
        row['ToG_Season_Avg'] = np.nan
        row['ToG_Last'] = np.nan

    # Missed game time
    # Get team from latest game (2025 preferred)
    team = pgroup_2025['Team'].iloc[-1] if not pgroup_2025.empty else pgroup_all['Team'].iloc[-1]

    # Get last 4 matches that the team played
    team_games = df[df['Team'] == team].drop_duplicates(subset='Date').sort_values('Date').tail(4)
    missed_time = False

    for game_date in team_games['Date']:
        # Check if player played on that date for this team
        played = (
            (pgroup_all['Date'] == game_date) &
            (pgroup_all['Team'] == team)
        )
        if not played.any():
            missed_time = True
            break
        else:
            tog = pgroup_all.loc[played, tog_col].iloc[0]
            if pd.isna(tog) or tog < 50:
                missed_time = True
                break

    row['Missed_Game_Time'] = missed_time

    # Team (assign to row, don't re-create row)
    team = pgroup_2025['Team'].iloc[-1] if not pgroup_2025.empty else pgroup_all['Team'].iloc[-1]
    row['Team'] = team


    # --- 2025 season stats ---
    row.update(compute_for_split(pgroup_2025, pgroup_2025['Disposals'].notnull(), "Disposal_Season"))
    row.update(compute_for_split(pgroup_2025, (pgroup_2025['Conditions'].str.lower() == 'dry'), "Disposal_Season_Dry"))
    row.update(compute_for_split(pgroup_2025, (pgroup_2025['Conditions'].str.lower() == 'wet'), "Disposal_Season_Wet"))
    row.update(compute_for_split(pgroup_2025, (pgroup_2025['Timeslot'].str.lower() == 'day'), "Disposal_Season_Day"))
    row.update(compute_for_split(pgroup_2025, (pgroup_2025['Timeslot'].str.lower() == 'night'), "Disposal_Season_Night"))
    row.update(compute_for_split(pgroup_2025, (pgroup_2025['Timeslot'].str.lower() == 'twilight'), "Disposal_Season_Twilight"))
    row.update(compute_for_split(pgroup_2025, home_away_mask(pgroup_2025), "Disposal_Season_Home"))
    row.update(compute_for_split(pgroup_2025, ~home_away_mask(pgroup_2025), "Disposal_Season_Away"))

    # --- Recent N games: ALL games, not just 2025 ---
    for N in [3, 6, 10, 22]:
        sorted_games = pgroup_all.sort_values('Date').tail(N)
        row.update(compute_for_split(sorted_games, sorted_games['Disposals'].notnull(), f"Disposal_{N}"))

        # Take last N DRY games
        last_n_dry = last_n_by_condition(pgroup_all, N, 'Conditions', 'dry')
        row.update(compute_for_split(last_n_dry, last_n_dry['Disposals'].notnull(), f"Disposal_{N}_Dry"))

        # Take last N WET games
        last_n_wet = last_n_by_condition(pgroup_all, N, 'Conditions', 'wet')
        row.update(compute_for_split(last_n_wet, last_n_wet['Disposals'].notnull(), f"Disposal_{N}_Wet"))

        # Take last N DAY games
        last_n_day = last_n_by_timeslot(pgroup_all, N, 'day')
        row.update(compute_for_split(last_n_day, last_n_day['Disposals'].notnull(), f"Disposal_{N}_Day"))

        # Take last N NIGHT games
        last_n_night = last_n_by_timeslot(pgroup_all, N, 'night')
        row.update(compute_for_split(last_n_night, last_n_night['Disposals'].notnull(), f"Disposal_{N}_Night"))

        # Take last N TWILIGHT games
        last_n_twilight = last_n_by_timeslot(pgroup_all, N, 'twilight')
        row.update(compute_for_split(last_n_twilight, last_n_twilight['Disposals'].notnull(), f"Disposal_{N}_Twilight"))

        # Last N home games
        last_n_home = last_n_home_away(pgroup_all, N, home=True)
        row.update(compute_for_split(last_n_home, last_n_home['Disposals'].notnull(), f"Disposal_{N}_Home"))

        # Last N away games
        last_n_away = last_n_home_away(pgroup_all, N, home=False)
        row.update(compute_for_split(last_n_away, last_n_away['Disposals'].notnull(), f"Disposal_{N}_Away"))

    # Add empty prediction fields (placeholders for now)
    row["Prob_20_Disposals"] = np.nan
    row["Prob_25_Disposals"] = np.nan
    row["Prob_30_Disposals"] = np.nan

    # Full 2025 season FloorScore
    if not pgroup_2025.empty:
        med = pgroup_2025['Disposals'].median()
        floor_drops = med - pgroup_2025['Disposals']
        floor_drops = floor_drops[floor_drops > 0]
        mean_drop = floor_drops.mean() if not floor_drops.empty else 0
        score = round(1 - (mean_drop / med), 3) if med > 0 else np.nan
    else:
        score = np.nan

    row['Disposal_Season_DropConsistency'] = score

    # --- Floor score: consistency against the median ---
    for N in [3, 6, 10, 22]:
        recent_games = pgroup_all.sort_values("Date").tail(N)
        if not recent_games.empty:
            med = recent_games['Disposals'].median()
            floor_drops = med - recent_games['Disposals']
            floor_drops = floor_drops[floor_drops > 0]
            mean_drop = floor_drops.mean() if not floor_drops.empty else 0
            score = round(1 - (mean_drop / med), 3) if med > 0 else np.nan
        else:
            score = np.nan

        row[f'Disposal_{N}_DropConsistency'] = score



    precomputes.append(row)

df_pre = pd.DataFrame(precomputes)
df_pre.to_sql("player_precomputes", engine, if_exists="replace", index=False)
with engine.connect() as conn:
    conn.execute(text('CREATE INDEX IF NOT EXISTS idx_player_precomputes_player ON player_precomputes ("Player")'))
    conn.execute(text('CREATE INDEX IF NOT EXISTS idx_player_precomputes_team ON player_precomputes ("Team")'))

print("✅ player_precomputes table written (2025 season splits + rolling windows from all years)")
