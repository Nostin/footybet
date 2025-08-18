import pandas as pd
from sqlalchemy import text
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

# --- Load upcoming games and normalise strings ---
upcoming_games_df = pd.read_sql('SELECT * FROM upcoming_games', engine)
for col in ["Home Team", "Away Team", "Venue", "Timeslot"]:
    if col in upcoming_games_df.columns:
        upcoming_games_df[col] = upcoming_games_df[col].astype(str).str.strip()
upcoming_games_df["Date"] = pd.to_datetime(upcoming_games_df["Date"], errors="coerce")

# --- Load full player stats ---
df = pd.read_sql('SELECT * FROM player_stats ORDER BY "Date" DESC', engine)
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df_2025 = df[df['Date'] >= '2025-01-01'].copy()

def summary_stats(s: pd.Series) -> pd.Series:
    # Cap disposals at 35 to reduce outlier impact
    capped = s.clip(upper=35)
    avg = np.mean(capped)
    med = np.median(capped)
    high = np.max(capped)
    low = np.min(capped)
    var = np.var(capped, ddof=0)
    cv = round(var / avg, 3) if avg > 0 else 0
    return pd.Series({
        'Avg': avg,
        'Median': np.median(s),      # report uncapped median for interpretability
        'High': np.max(s),
        'Low': np.min(s),
        'Variance': cv               # using CV as variance-like dispersion
    })

def compute_for_split(df_in: pd.DataFrame, mask, prefix: str):
    sub = df_in[mask]
    if sub.empty:
        return {f"{prefix}_{k}": np.nan for k in ['Avg','Median','High','Low','Variance']}
    stats = summary_stats(sub['Disposals'])
    return {f"{prefix}_{k}": v for k, v in stats.items()}

def is_home_game_row(row):
    home_venues = HOME_GROUNDS.get(row['Team'], [])
    return row['Venue'] in home_venues

def home_away_mask(df_in):
    return df_in.apply(is_home_game_row, axis=1)

def last_n_by_condition(df_in, N, col, val):
    mask = df_in[col].astype(str).str.lower() == val
    return df_in[mask].sort_values('Date').tail(N)

def last_n_by_timeslot(df_in, N, val):
    mask = df_in['Timeslot'].astype(str).str.lower() == val
    return df_in[mask].sort_values('Date').tail(N)

def last_n_home_away(df_in, N, home=True):
    mask = df_in.apply(is_home_game_row, axis=1)
    if not home:
        mask = ~mask
    return df_in[mask].sort_values('Date').tail(N)

def last5_derived(disposals: pd.Series) -> dict:
    """
    Compute the five model-required last-5 features from a chronological
    series of past disposals (strictly pre-next-game if possible).
    """
    out = {
        'disposals_trend_last_5': np.nan,
        'disposals_delta_5': np.nan,
        'disposals_max_last_5': np.nan,
        'disposals_min_last_5': np.nan,
        'disposals_std_last_5': np.nan,
    }
    if disposals is None or disposals.empty:
        return out

    vals = disposals.dropna().values
    if vals.size >= 5:
        last5 = vals[-5:]
        # slope of linear fit over indices 0..4
        try:
            slope = np.polyfit(np.arange(len(last5)), last5, 1)[0]
        except Exception:
            slope = np.nan
        out['disposals_trend_last_5'] = slope
        out['disposals_max_last_5'] = float(np.max(last5))
        out['disposals_min_last_5'] = float(np.min(last5))
        out['disposals_std_last_5'] = float(np.std(last5, ddof=0))
    # delta over a 5-game span uses last and the one 5 games earlier -> need >=6
    if vals.size >= 6:
        out['disposals_delta_5'] = float(vals[-1] - vals[-6])

    return out

precomputes = []

# Precompute team totals by date once (helps with pace and concessions)
team_totals_all = (df.groupby(['Date','Team'], as_index=False)['Disposals']
                     .sum()
                     .rename(columns={'Disposals':'team_total'}))

# If Opponent column exists, precompute allowed-by-team using true pairings
has_opponent = 'Opponent' in df.columns
if has_opponent:
    # Sum by Date-Team-Opponent (player rows -> team totals)
    dt_team_opp = (df.groupby(['Date','Team','Opponent'], as_index=False)['Disposals'].sum()
                     .rename(columns={'Disposals':'team_total_vs_opp'}))
    # "Allowed by Team" = what their opponents put up against them on those dates
    allowed_by_team = (dt_team_opp.rename(columns={'Team':'OppFaced','Opponent':'Team','team_total_vs_opp':'disposals_allowed'})
                                 .sort_values(['Team','Date']))
else:
    allowed_by_team = None

for player in df['Player'].unique():
    row = {'Player': player}

    # Slice per-player
    pgroup_all = df[df['Player'] == player].copy()
    pgroup_all.sort_values('Date', inplace=True)
    pgroup_2025 = df_2025[df_2025['Player'] == player].copy()
    row["Games_This_Season"] = int(len(pgroup_2025))

    # --- Identify team (prefer 2025, else all-time latest) ---
    if not pgroup_2025.empty:
        team = str(pgroup_2025['Team'].iloc[-1]).strip()
    elif not pgroup_all.empty:
        team = str(pgroup_all['Team'].iloc[-1]).strip()
    else:
        team = None
    row['Team'] = team if team is not None else ''

    # --- Next game lookup & Days_since_last_game ---
    row["Next_Opponent"] = ''
    row["Next_Venue"] = ''
    row["Next_Timeslot"] = ''
    row["Days_since_last_game"] = np.nan

    next_game_date = None
    if team:
        last_game_date = pgroup_all['Date'].max() if not pgroup_all.empty else pd.NaT
        mask_team_next = (upcoming_games_df["Home Team"] == team) | (upcoming_games_df["Away Team"] == team)
        next_games = upcoming_games_df[mask_team_next]
        if pd.notna(last_game_date):
            next_games = next_games[next_games["Date"] > last_game_date]
        next_games = next_games.sort_values("Date")

        if not next_games.empty:
            next_game = next_games.iloc[0]
            next_game_date = next_game["Date"]
            row["Next_Venue"] = str(next_game["Venue"])
            row["Next_Timeslot"] = str(next_game["Timeslot"])
            if next_game["Home Team"] == team:
                row["Next_Opponent"] = str(next_game["Away Team"])
            else:
                row["Next_Opponent"] = str(next_game["Home Team"])
            if pd.notna(last_game_date):
                row["Days_since_last_game"] = int((next_game_date - last_game_date).days)

    # --- ToG (this season) ---
    tog_col = 'Time on Ground %'
    if not pgroup_2025.empty:
        row['ToG_Season_Avg'] = float(pgroup_2025[tog_col].mean())
        row['ToG_Last'] = float(pgroup_2025.sort_values('Date')[tog_col].iloc[-1])
    else:
        row['ToG_Season_Avg'] = np.nan
        row['ToG_Last'] = np.nan

    # --- Missed game time in last 4 team matches (absent or ToG < 50) ---
    missed_time = False
    if team:
        team_games = df[df['Team'] == team].drop_duplicates(subset='Date').sort_values('Date').tail(4)
        for game_date in team_games['Date']:
            played = ((pgroup_all['Date'] == game_date) & (pgroup_all['Team'] == team))
            if not played.any():
                missed_time = True
                break
            else:
                tog = pgroup_all.loc[played, tog_col].iloc[0]
                if pd.isna(tog) or tog < 50:
                    missed_time = True
                    break
    row['Missed_Game_Time'] = bool(missed_time)

    # --- 2025 season stats (by conditions/timeslot/home/away) ---
    if not pgroup_2025.empty:
        row.update(compute_for_split(pgroup_2025, pgroup_2025['Disposals'].notnull(), "Disposal_Season"))
        row.update(compute_for_split(pgroup_2025, (pgroup_2025['Conditions'].astype(str).str.lower() == 'dry'), "Disposal_Season_Dry"))
        row.update(compute_for_split(pgroup_2025, (pgroup_2025['Conditions'].astype(str).str.lower() == 'wet'), "Disposal_Season_Wet"))
        row.update(compute_for_split(pgroup_2025, (pgroup_2025['Timeslot'].astype(str).str.lower() == 'day'), "Disposal_Season_Day"))
        row.update(compute_for_split(pgroup_2025, (pgroup_2025['Timeslot'].astype(str).str.lower() == 'night'), "Disposal_Season_Night"))
        row.update(compute_for_split(pgroup_2025, (pgroup_2025['Timeslot'].astype(str).str.lower() == 'twilight'), "Disposal_Season_Twilight"))
        hamask_2025 = home_away_mask(pgroup_2025)
        row.update(compute_for_split(pgroup_2025, hamask_2025, "Disposal_Season_Home"))
        row.update(compute_for_split(pgroup_2025, ~hamask_2025, "Disposal_Season_Away"))
    else:
        for p in ["Disposal_Season","Disposal_Season_Dry","Disposal_Season_Wet","Disposal_Season_Day","Disposal_Season_Night","Disposal_Season_Twilight","Disposal_Season_Home","Disposal_Season_Away"]:
            for k in ['Avg','Median','High','Low','Variance']:
                row[f"{p}_{k}"] = np.nan

    # --- Recent windows from ALL years ---
    for N in [3, 6, 10, 22]:
        sorted_games = pgroup_all.sort_values('Date').tail(N)
        row.update(compute_for_split(sorted_games, sorted_games['Disposals'].notnull(), f"Disposal_{N}"))

        last_n_dry = last_n_by_condition(pgroup_all, N, 'Conditions', 'dry')
        row.update(compute_for_split(last_n_dry, last_n_dry['Disposals'].notnull(), f"Disposal_{N}_Dry"))

        last_n_wet = last_n_by_condition(pgroup_all, N, 'Conditions', 'wet')
        row.update(compute_for_split(last_n_wet, last_n_wet['Disposals'].notnull(), f"Disposal_{N}_Wet"))

        last_n_day = last_n_by_timeslot(pgroup_all, N, 'day')
        row.update(compute_for_split(last_n_day, last_n_day['Disposals'].notnull(), f"Disposal_{N}_Day"))

        last_n_night = last_n_by_timeslot(pgroup_all, N, 'night')
        row.update(compute_for_split(last_n_night, last_n_night['Disposals'].notnull(), f"Disposal_{N}_Night"))

        last_n_twilight = last_n_by_timeslot(pgroup_all, N, 'twilight')
        row.update(compute_for_split(last_n_twilight, last_n_twilight['Disposals'].notnull(), f"Disposal_{N}_Twilight"))

        last_n_home = last_n_home_away(pgroup_all, N, home=True)
        row.update(compute_for_split(last_n_home, last_n_home['Disposals'].notnull(), f"Disposal_{N}_Home"))

        last_n_away = last_n_home_away(pgroup_all, N, home=False)
        row.update(compute_for_split(last_n_away, last_n_away['Disposals'].notnull(), f"Disposal_{N}_Away"))

    # --- Prediction placeholders ---
    row["Prob_20_Disposals"] = np.nan
    row["Prob_25_Disposals"] = np.nan
    row["Prob_30_Disposals"] = np.nan

    # --- Season-long floor score (drop consistency vs season median) ---
    if not pgroup_2025.empty:
        med = pgroup_2025['Disposals'].median()
        floor_drops = med - pgroup_2025['Disposals']
        floor_drops = floor_drops[floor_drops > 0]
        score = round(1 - (floor_drops.mean() / med), 3) if med > 0 and not floor_drops.empty else (1.0 if med > 0 else np.nan)
    else:
        score = np.nan
    row['Disposal_Season_DropConsistency'] = score

    # --- Floor score for N in {3,6,10,22} ---
    for N in [3, 6, 10, 22]:
        recent_games = pgroup_all.sort_values("Date").tail(N)
        if not recent_games.empty:
            med = recent_games['Disposals'].median()
            floor_drops = med - recent_games['Disposals']
            floor_drops = floor_drops[floor_drops > 0]
            scoreN = round(1 - (floor_drops.mean() / med), 3) if med > 0 and not floor_drops.empty else (1.0 if med > 0 else np.nan)
        else:
            scoreN = np.nan
        row[f'Disposal_{N}_DropConsistency'] = scoreN

    # =============================
    # NEW: Model-friendly deriveds
    # =============================

    # 1) form_minus_season_med_last_3 = last 3 avg (capped) - season median
    row['form_minus_season_med_last_3'] = (
        row.get('Disposal_3_Avg', np.nan) - row.get('Disposal_Season_Median', np.nan)
    )

    # 2) is_home_game for the NEXT match (based on Next_Venue)
    if team and row.get('Next_Venue'):
        row['is_home_game'] = int(row['Next_Venue'] in HOME_GROUNDS.get(team, []))
    else:
        row['is_home_game'] = 0

    # 3) avg_team_disposals_last_4 — team's four most recent completed matches BEFORE next game
    if team:
        team_totals = team_totals_all[team_totals_all['Team'] == team].sort_values('Date')
        ref_date = next_game_date if next_game_date is not None else (pgroup_all['Date'].max() if not pgroup_all.empty else pd.NaT)
        if pd.notna(ref_date):
            recent4 = team_totals[team_totals['Date'] < ref_date].tail(4)['team_total']
        else:
            recent4 = team_totals.tail(4)['team_total']
        row['avg_team_disposals_last_4'] = float(recent4.mean()) if not recent4.empty else np.nan
    else:
        row['avg_team_disposals_last_4'] = np.nan

    # 4) opp_concessions_last_5 — how many disposals the NEXT opponent usually concedes
    if row.get('Next_Opponent'):
        opp = row['Next_Opponent']
        if has_opponent:
            opp_allowed = allowed_by_team[allowed_by_team['Team'] == opp]
            ref_date = next_game_date
            if ref_date is None and not pgroup_all.empty:
                ref_date = pgroup_all['Date'].max()
            if ref_date is not None and pd.notna(ref_date):
                opp_hist = opp_allowed[opp_allowed['Date'] < ref_date].tail(5)
            else:
                opp_hist = opp_allowed.tail(5)
            row['opp_concessions_last_5'] = float(opp_hist['disposals_allowed'].mean()) if not opp_hist.empty else np.nan
        else:
            row['opp_concessions_last_5'] = np.nan
    else:
        row['opp_concessions_last_5'] = np.nan

    # 5) Wet/Dry last-3 (for parity with training names)
    row['wet_disposals_last_3'] = row.get('Disposal_3_Wet_Avg', np.nan)
    row['dry_disposals_last_3'] = row.get('Disposal_3_Dry_Avg', np.nan)

    # 6) The missing last-5 features (trend/max/min/std/delta)
    # Use only games STRICTLY before the next match if we know that date
    if next_game_date is not None:
        past_games = pgroup_all[pgroup_all['Date'] < next_game_date]
    else:
        past_games = pgroup_all
    d5 = last5_derived(past_games['Disposals'] if not past_games.empty else pd.Series([], dtype=float))
    row.update(d5)

    precomputes.append(row)

# --- Write table ---
df_pre = pd.DataFrame(precomputes)

# Ensure new model columns exist even if some rows were empty
must_have = [
    'form_minus_season_med_last_3', 'is_home_game',
    'avg_team_disposals_last_4', 'opp_concessions_last_5',
    'wet_disposals_last_3', 'dry_disposals_last_3',
    'disposals_trend_last_5', 'disposals_delta_5',
    'disposals_max_last_5', 'disposals_min_last_5', 'disposals_std_last_5'
]
for c in must_have:
    if c not in df_pre.columns:
        df_pre[c] = np.nan

df_pre.to_sql("player_precomputes", engine, if_exists="replace", index=False)
with engine.connect() as conn:
    conn.execute(text('CREATE INDEX IF NOT EXISTS idx_player_precomputes_player ON player_precomputes ("Player")'))
    conn.execute(text('CREATE INDEX IF NOT EXISTS idx_player_precomputes_team ON player_precomputes ("Team")'))

print("✅ player_precomputes table written (incl. Next_* fields + model-friendly features + last-5 metrics)")
