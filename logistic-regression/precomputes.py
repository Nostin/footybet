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
    'North Melbourne': ['Docklands', 'Hobart'],
    'Port Adelaide': ['Adelaide'],
    'Richmond': ['MCG'],
    'Saints': ['Docklands'],
    'Sydney': ['SCG'],
    'West Coast': ['Perth'],
    'Western Bulldogs': ['Docklands', 'Ballarat'],
}

df = pd.read_sql('SELECT * FROM player_stats ORDER BY "Date" DESC', engine)
df['Date'] = pd.to_datetime(df['Date'])
df_2025 = df[df['Date'] >= '2025-01-01']

def summary_stats(s):
    return pd.Series({
        'Avg': np.mean(s),
        'Median': np.median(s),
        'High': np.max(s),
        'Low': np.min(s),
        'Variance': np.var(s, ddof=0)
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

    # Days since last game
    if not pgroup_all.empty:
        last_game_date = pgroup_all['Date'].max()
        row['Days_since_last_game'] = (pd.Timestamp.now().normalize() - last_game_date).days
    else:
        row['Days_since_last_game'] = np.nan

    # ToG
    tog_col = 'Time on Ground %'
    if not pgroup_2025.empty:
        row['ToG_Season_Avg'] = pgroup_2025[tog_col].mean()
        row['ToG_Last'] = pgroup_2025.sort_values('Date')[tog_col].iloc[-1]
    else:
        row['ToG_Season_Avg'] = np.nan
        row['ToG_Last'] = np.nan

    # Missed game time
    last_4 = pgroup_all.sort_values('Date').tail(4)
    row['Missed_Game_Time'] = (
        last_4.empty or last_4[tog_col].isnull().any() or (last_4[tog_col] < 50).any()
    )

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
    row["Prob_20_Disp_Dry"] = np.nan
    row["Prob_20_Disp_Wet"] = np.nan
    row["Prob_25_Disp_Dry"] = np.nan
    row["Prob_25_Disp_Wet"] = np.nan
    row["Prob_30_Disp_Dry"] = np.nan
    row["Prob_30_Disp_Wet"] = np.nan

    precomputes.append(row)

df_pre = pd.DataFrame(precomputes)
df_pre.to_sql("player_precomputes", engine, if_exists="replace", index=False)
with engine.connect() as conn:
    conn.execute(text('CREATE INDEX IF NOT EXISTS idx_player_precomputes_player ON player_precomputes ("Player")'))

print("✅ player_precomputes table written (2025 season splits + rolling windows from all years)")
