from pathlib import Path
import sys
import pandas as pd
import numpy as np
from sqlalchemy import text as sqtext

ROOT = Path(__file__).resolve().parents[1]   # /.../root
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from db_connect import get_engine
from util import HOME_GROUNDS, TEAM_ALIASES

engine = get_engine()

def normalize_team_key(team: str) -> str:
    t = (team or "").strip()
    return t if t in HOME_GROUNDS else TEAM_ALIASES.get(t, t)

def is_home_game_row(row) -> bool:
    team_key = normalize_team_key(row.get('Team', ''))
    return str(row.get('Venue', '')).strip() in HOME_GROUNDS.get(team_key, [])

def to_python_scalar(v):
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

# ---------- Handballs helpers ----------
METRIC = 'Handballs'
TOG_COL = 'Time on Ground %'

def summary_stats(s: pd.Series) -> pd.Series:
    if s is None or s.empty:
        return pd.Series({'Avg': np.nan, 'Median': np.nan, 'High': np.nan, 'Low': np.nan, 'Variance': np.nan})
    capped = s.clip(upper=25)
    avg = float(np.mean(capped)) if not capped.empty else np.nan
    med = float(np.median(s))
    high = float(np.max(s))
    low  = float(np.min(s))
    var  = float(np.var(capped, ddof=0)) if not capped.empty else np.nan
    cv   = round(var / avg, 3) if (avg and avg > 0) else np.nan
    return pd.Series({'Avg': avg, 'Median': med, 'High': high, 'Low': low, 'Variance': cv})

def compute_for_split_metric(df_in: pd.DataFrame, mask, prefix: str):
    sub = df_in[mask]
    if sub.empty or METRIC not in sub.columns:
        return {f"{prefix}_{k}": np.nan for k in ['Avg','Median','High','Low','Variance']}
    stats = summary_stats(sub[METRIC])
    return {f"{prefix}_{k}": to_python_scalar(stats[k]) for k in ['Avg','Median','High','Low','Variance']}

def last_n_by(df_in: pd.DataFrame, N: int, col: str, val: str):
    mask = df_in[col].astype(str).str.lower() == val
    return df_in[mask].sort_values('Date').tail(N)

def last_n_home_away(df_in: pd.DataFrame, N: int, home=True):
    mask = df_in.apply(is_home_game_row, axis=1)
    if not home:
        mask = ~mask
    return df_in[mask].sort_values('Date').tail(N)

def zero_rate(series: pd.Series) -> float:
    if series is None or series.empty:
        return np.nan
    n = series.size
    if n == 0:
        return np.nan
    return float((series == 0).sum()) / float(n)

def handballs_per100_tog(series_vals: pd.Series, series_tog: pd.Series) -> float:
    if series_vals is None or series_vals.empty or series_tog is None or series_tog.empty:
        return np.nan
    df = pd.DataFrame({'x': series_vals, 'tog': series_tog}).dropna()
    if df.empty:
        return np.nan
    df = df[df['tog'] > 0]
    if df.empty:
        return np.nan
    rates = df['x'] * 100.0 / df['tog']
    return float(rates.mean())

# ---------- Load ----------
df = pd.read_sql('SELECT * FROM player_stats ORDER BY "Date" ASC', engine)
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
if not df['Date'].notna().any():
    raise RuntimeError("player_stats has no valid dates")

season_year = int(df['Date'].dt.year.max())
df_season = df[df['Date'].dt.year == season_year].copy()

base_players = pd.read_sql('SELECT "Player" FROM player_precomputes', engine)['Player'].dropna().unique().tolist()
if not base_players:
    raise RuntimeError("player_precomputes is empty. Run the basic precomputes builder first.")

# ---------- Columns ----------
stat_suffixes = ['Avg','Median','High','Low','Variance']

season_prefixes = [
    "Handball_Season", "Handball_Season_Dry", "Handball_Season_Wet",
    "Handball_Season_Home", "Handball_Season_Away",
]
WINDOWS = [6, 10, 22]
recent_prefixes = []
for N in WINDOWS:
    recent_prefixes += [f"Handball_{N}", f"Handball_{N}_Dry", f"Handball_{N}_Wet", f"Handball_{N}_Home", f"Handball_{N}_Away"]

zero_cols = ["Handball_Season_ZeroRate"] + [f"Handball_{N}_ZeroRate" for N in WINDOWS]
rate_cols = ["Handballs_per100ToG_Season"] + [f"Handballs_per100ToG_{N}" for N in WINDOWS]

handball_cols = (
    [f"{p}_{s}" for p in season_prefixes for s in stat_suffixes] +
    [f"{p}_{s}" for p in recent_prefixes for s in stat_suffixes] +
    zero_cols + rate_cols
)

with engine.begin() as conn:
    for col in handball_cols:
        conn.exec_driver_sql(f'ALTER TABLE player_precomputes ADD COLUMN IF NOT EXISTS "{col}" DOUBLE PRECISION')

# ---------- Compute ----------
def compute_player_row(player: str) -> dict:
    out = {"Player": player}
    p_all = df[df['Player'] == player].copy().sort_values('Date')
    p_season = df_season[df_season['Player'] == player].copy().sort_values('Date')

    if not p_season.empty:
        out.update(compute_for_split_metric(p_season, p_season[METRIC].notnull(), "Handball_Season"))
        out.update(compute_for_split_metric(p_season, p_season['Conditions'].astype(str).str.lower() == 'dry', "Handball_Season_Dry"))
        out.update(compute_for_split_metric(p_season, p_season['Conditions'].astype(str).str.lower() == 'wet', "Handball_Season_Wet"))
        hamask = p_season.apply(is_home_game_row, axis=1)
        out.update(compute_for_split_metric(p_season, hamask, "Handball_Season_Home"))
        out.update(compute_for_split_metric(p_season, ~hamask, "Handball_Season_Away"))
        out["Handball_Season_ZeroRate"] = to_python_scalar(zero_rate(p_season[METRIC]))
        out["Handballs_per100ToG_Season"] = to_python_scalar(handballs_per100_tog(
            p_season[METRIC], p_season.get(TOG_COL, pd.Series(dtype=float))
        ))
    else:
        for pfx in season_prefixes:
            for sfx in stat_suffixes:
                out[f"{pfx}_{sfx}"] = np.nan
        out["Handball_Season_ZeroRate"] = np.nan
        out["Handballs_per100ToG_Season"] = np.nan

    for N in WINDOWS:
        recent = p_all.sort_values('Date').tail(N)
        out.update(compute_for_split_metric(recent, recent[METRIC].notnull(), f"Handball_{N}"))
        last_dry = last_n_by(p_all, N, 'Conditions', 'dry')
        out.update(compute_for_split_metric(last_dry, last_dry[METRIC].notnull(), f"Handball_{N}_Dry"))
        last_wet = last_n_by(p_all, N, 'Conditions', 'wet')
        out.update(compute_for_split_metric(last_wet, last_wet[METRIC].notnull(), f"Handball_{N}_Wet"))
        last_home = last_n_home_away(p_all, N, home=True)
        out.update(compute_for_split_metric(last_home, last_home[METRIC].notnull(), f"Handball_{N}_Home"))
        last_away = last_n_home_away(p_all, N, home=False)
        out.update(compute_for_split_metric(last_away, last_away[METRIC].notnull(), f"Handball_{N}_Away"))

        out[f"Handball_{N}_ZeroRate"] = to_python_scalar(zero_rate(recent[METRIC]))
        out[f"Handballs_per100ToG_{N}"] = to_python_scalar(handballs_per100_tog(
            recent[METRIC], recent.get(TOG_COL, pd.Series(dtype=float))
        ))

    return out

updates = [compute_player_row(p) for p in base_players]

with engine.begin() as conn:
    for row in updates:
        player = row.pop("Player")
        clean = {k: to_python_scalar(v) for k, v in row.items()}
        clean["player"] = player
        set_fragments = [f'"{col}" = :{col}' for col in row.keys()]
        sql = f'UPDATE player_precomputes SET {", ".join(set_fragments)} WHERE "Player" = :player'
        conn.execute(sqtext(sql), clean)

print(f"✅ Added/updated Handball features for {len(updates)} players in player_precomputes (season {season_year})")
