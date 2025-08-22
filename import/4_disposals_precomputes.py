# augment_player_precomputes_disposals.py
import pandas as pd
import numpy as np
from sqlalchemy import text
from db_connect import get_engine

engine = get_engine()

# ---------- Home/Away helpers ----------
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

def is_home_game_row(row) -> bool:
    team_key = normalize_team_key(row.get('Team', ''))
    return str(row.get('Venue', '')).strip() in HOME_GROUNDS.get(team_key, [])

# ---------- Numeric coercion (fix psycopg2/NumPy issues) ----------
def to_python_scalar(v):
    # Convert numpy / pandas scalars & NaNs to native Python types
    if v is None:
        return None
    try:
        if pd.isna(v):
            return None
    except Exception:
        pass
    if isinstance(v, (np.floating, np.float32, np.float64)):
        return float(v)
    if isinstance(v, (np.integer, np.int32, np.int64)):
        return int(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    return v

# ---------- Disposals-only stat helpers ----------
def summary_stats(s: pd.Series) -> pd.Series:
    # Cap at 35 to tame outliers; keep median/high/low uncapped for interpretability
    if s is None or s.empty:
        return pd.Series({'Avg': np.nan, 'Median': np.nan, 'High': np.nan, 'Low': np.nan, 'Variance': np.nan})
    capped = s.clip(upper=35)
    avg = float(np.mean(capped)) if not capped.empty else np.nan
    med = float(np.median(s))
    high = float(np.max(s))
    low  = float(np.min(s))
    var  = float(np.var(capped, ddof=0)) if not capped.empty else np.nan
    cv   = round(var / avg, 3) if (avg is not np.nan and avg and avg > 0) else np.nan   # CV-as-variance-like
    return pd.Series({'Avg': avg, 'Median': med, 'High': high, 'Low': low, 'Variance': cv})

def compute_for_split(df_in: pd.DataFrame, mask, prefix: str):
    sub = df_in[mask]
    if sub.empty or 'Disposals' not in sub.columns:
        return {f"{prefix}_{k}": np.nan for k in ['Avg','Median','High','Low','Variance']}
    stats = summary_stats(sub['Disposals'])
    return {f"{prefix}_{k}": to_python_scalar(stats[k]) for k in ['Avg','Median','High','Low','Variance']}

def last_n_by(df_in: pd.DataFrame, N: int, col: str, val: str):
    mask = df_in[col].astype(str).str.lower() == val
    return df_in[mask].sort_values('Date').tail(N)

def last_n_home_away(df_in: pd.DataFrame, N: int, home=True):
    mask = df_in.apply(is_home_game_row, axis=1)
    if not home:
        mask = ~mask
    return df_in[mask].sort_values('Date').tail(N)

# ---------- Load source data ----------
df = pd.read_sql('SELECT * FROM player_stats ORDER BY "Date" ASC', engine)
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
if not df['Date'].notna().any():
    raise RuntimeError("player_stats has no valid dates")

season_year = int(df['Date'].dt.year.max())
df_season = df[df['Date'].dt.year == season_year].copy()

# Only update players already in base table
base_players = pd.read_sql('SELECT "Player" FROM player_precomputes', engine)['Player'].dropna().unique().tolist()
if not base_players:
    raise RuntimeError("player_precomputes is empty. Run the basic precomputes builder first.")

# ---------- Decide columns to add (disposals only) ----------
stat_suffixes = ['Avg','Median','High','Low','Variance']

season_prefixes = [
    "Disposal_Season",
    "Disposal_Season_Dry",
    "Disposal_Season_Wet",
    "Disposal_Season_Day",
    "Disposal_Season_Night",
    "Disposal_Season_Twilight",
    "Disposal_Season_Home",
    "Disposal_Season_Away",
]

recent_prefixes = []
for N in [3, 6, 10, 22]:
    recent_prefixes += [
        f"Disposal_{N}",
        f"Disposal_{N}_Dry",
        f"Disposal_{N}_Wet",
        f"Disposal_{N}_Day",
        f"Disposal_{N}_Night",
        f"Disposal_{N}_Twilight",
        f"Disposal_{N}_Home",
        f"Disposal_{N}_Away",
    ]

dropconsistency_cols = [f"Disposal_{N}_DropConsistency" for N in [3, 6, 10, 22]]

# NOTE: no last5_derived columns here
base_cols = (
    [f"{p}_{s}" for p in season_prefixes for s in stat_suffixes] +
    [f"{p}_{s}" for p in recent_prefixes for s in stat_suffixes] +
    [
        "Disposal_Season_DropConsistency",
        *dropconsistency_cols,
        "Prob_20_Disposals", "Prob_25_Disposals", "Prob_30_Disposals",
    ]
)

# ---------- ALTER TABLE: add disposals columns if missing ----------
with engine.begin() as conn:
    for col in base_cols:
        conn.exec_driver_sql(f'ALTER TABLE player_precomputes ADD COLUMN IF NOT EXISTS "{col}" DOUBLE PRECISION')

# ---------- Compute per-player values ----------
def compute_player_row(player: str) -> dict:
    out = {"Player": player}
    p_all = df[df['Player'] == player].copy().sort_values('Date')
    p_season = df_season[df_season['Player'] == player].copy().sort_values('Date')

    # Season splits
    if not p_season.empty:
        out.update(compute_for_split(p_season, p_season['Disposals'].notnull(), "Disposal_Season"))
        out.update(compute_for_split(p_season, p_season['Conditions'].astype(str).str.lower() == 'dry', "Disposal_Season_Dry"))
        out.update(compute_for_split(p_season, p_season['Conditions'].astype(str).str.lower() == 'wet', "Disposal_Season_Wet"))
        out.update(compute_for_split(p_season, p_season['Timeslot'].astype(str).str.lower() == 'day', "Disposal_Season_Day"))
        out.update(compute_for_split(p_season, p_season['Timeslot'].astype(str).str.lower() == 'night', "Disposal_Season_Night"))
        out.update(compute_for_split(p_season, p_season['Timeslot'].astype(str).str.lower() == 'twilight', "Disposal_Season_Twilight"))
        hamask_season = p_season.apply(is_home_game_row, axis=1)
        out.update(compute_for_split(p_season, hamask_season, "Disposal_Season_Home"))
        out.update(compute_for_split(p_season, ~hamask_season, "Disposal_Season_Away"))
    else:
        for pfx in season_prefixes:
            for sfx in stat_suffixes:
                out[f"{pfx}_{sfx}"] = np.nan

    # Recent windows (all years)
    for N in [3, 6, 10, 22]:
        recent = p_all.sort_values('Date').tail(N)
        out.update(compute_for_split(recent, recent['Disposals'].notnull(), f"Disposal_{N}"))

        last_dry = last_n_by(p_all, N, 'Conditions', 'dry')
        out.update(compute_for_split(last_dry, last_dry['Disposals'].notnull(), f"Disposal_{N}_Dry"))

        last_wet = last_n_by(p_all, N, 'Conditions', 'wet')
        out.update(compute_for_split(last_wet, last_wet['Disposals'].notnull(), f"Disposal_{N}_Wet"))

        last_day = last_n_by(p_all, N, 'Timeslot', 'day')
        out.update(compute_for_split(last_day, last_day['Disposals'].notnull(), f"Disposal_{N}_Day"))

        last_night = last_n_by(p_all, N, 'Timeslot', 'night')
        out.update(compute_for_split(last_night, last_night['Disposals'].notnull(), f"Disposal_{N}_Night"))

        last_twilight = last_n_by(p_all, N, 'Timeslot', 'twilight')
        out.update(compute_for_split(last_twilight, last_twilight['Disposals'].notnull(), f"Disposal_{N}_Twilight"))

        last_home = last_n_home_away(p_all, N, home=True)
        out.update(compute_for_split(last_home, last_home['Disposals'].notnull(), f"Disposal_{N}_Home"))

        last_away = last_n_home_away(p_all, N, home=False)
        out.update(compute_for_split(last_away, last_away['Disposals'].notnull(), f"Disposal_{N}_Away"))

    # Placeholders for model probabilities (left as NULL)
    out["Prob_20_Disposals"] = np.nan
    out["Prob_25_Disposals"] = np.nan
    out["Prob_30_Disposals"] = np.nan

    # Drop-consistency (season + windows)
    if not p_season.empty:
        med = p_season['Disposals'].median()
        drops = med - p_season['Disposals']
        drops = drops[drops > 0]
        score = round(1 - (drops.mean() / med), 3) if med > 0 and not drops.empty else (1.0 if med > 0 else np.nan)
    else:
        score = np.nan
    out["Disposal_Season_DropConsistency"] = score

    for N in [3, 6, 10, 22]:
        recent = p_all.sort_values('Date').tail(N)
        if not recent.empty:
            med = recent['Disposals'].median()
            drops = med - recent['Disposals']
            drops = drops[drops > 0]
            sc = round(1 - (drops.mean() / med), 3) if med > 0 and not drops.empty else (1.0 if med > 0 else np.nan)
        else:
            sc = np.nan
        out[f"Disposal_{N}_DropConsistency"] = sc

    return out

updates = [compute_player_row(p) for p in base_players]

# ---------- Bulk UPDATE ----------
with engine.begin() as conn:
    for row in updates:
        player = row.pop("Player")
        # Coerce values to native Python types / NULLs
        clean = {k: to_python_scalar(v) for k, v in row.items()}
        clean["player"] = player

        set_fragments = [f'"{col}" = :{col}' for col in row.keys()]
        sql = f'UPDATE player_precomputes SET {", ".join(set_fragments)} WHERE "Player" = :player'
        conn.execute(text(sql), clean)

print(f"✅ Added/updated Disposals features for {len(updates)} players in player_precomputes (season {season_year})")
