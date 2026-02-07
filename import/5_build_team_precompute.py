from pathlib import Path
import sys
import argparse
import pandas as pd
import numpy as np
from sqlalchemy import text

ROOT = Path(__file__).resolve().parents[1]   # /.../root
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from db_connect import get_engine
from util import HOME_GROUNDS, SECONDARY_HOME_GROUNDS, TEAM_STATE, VENUE_STATE, TEAM_ALIASES

ap = argparse.ArgumentParser()
ap.add_argument("--source", default="player_stats")
ap.add_argument("--dest",   default="team_precompute")
ap.add_argument("--windows", default="3,5,10")
ap.add_argument("--exclude-finals-from-rolling", action="store_true")
ap.add_argument("--stats-mode", choices=["inclusive","pre"], default="inclusive")
ap.add_argument("--only-after", default=None)  # YYYY-MM-DD
ap.add_argument("--chunksize", type=int, default=20000)
args = ap.parse_args()

engine = get_engine()

FINALS = {"EF","QF","SF","PF","GF"}
def is_finals_round(x):
    if x is None: return False
    return str(x).strip().upper() in FINALS

def normalize_team_key(team: str) -> str:
    t = (team or "").strip()
    # normalize to the keys used in HOME_GROUNDS / TEAM_STATE
    if t in HOME_GROUNDS or t in TEAM_STATE:
        return t
    return TEAM_ALIASES.get(t, t)

# --- Primary/Secondary home-ground helpers ------------------------------

# Clean SECONDARY_HOME_GROUNDS (original mapping may contain empty strings)
SECONDARY_CLEAN = {
    k: [v.strip() for v in vs if v and str(v).strip()]
    for k, vs in SECONDARY_HOME_GROUNDS.items()
}

def compute_is_primary_home(team: str, venue: str) -> int:
    if pd.isna(team) or pd.isna(venue):
        return 0
    tkey = normalize_team_key(team)
    return int(str(venue).strip() in HOME_GROUNDS.get(tkey, []))

def compute_is_home(team: str, venue: str) -> int:
    """Home = primary ∪ secondary."""
    if pd.isna(team) or pd.isna(venue):
        return 0
    tkey = normalize_team_key(team)
    v = str(venue).strip()
    return int(
        (v in HOME_GROUNDS.get(tkey, [])) or
        (v in SECONDARY_CLEAN.get(tkey, []))
    )

def team_state(team: str) -> str:
    return TEAM_STATE.get((team or "").strip(), "")

def venue_state(venue: str) -> str:
    return VENUE_STATE.get((venue or "").strip(), "")

def is_interstate(team: str, venue: str) -> int:
    ts, vs = team_state(team), venue_state(venue)
    return int(bool(ts) and bool(vs) and ts != vs)

def is_secondary_home(team: str, venue: str) -> int:
    """Explicit secondary home grounds per mapping (can be same-state)."""
    if pd.isna(team) or pd.isna(venue):
        return 0
    tkey = normalize_team_key(team)
    return int(str(venue).strip() in SECONDARY_CLEAN.get(tkey, []))

# ------------ Discover columns safely ------------
sample = pd.read_sql(text(f'SELECT * FROM {args.source} LIMIT 0'), engine)
cols_available = set(sample.columns)

need_base = ["Date","Team","Round","Venue","Timeslot"]
# Opponent is optional in source; we’ll add later from team_games if missing
maybe_opp = "Opponent" if "Opponent" in cols_available else None

# player-level sums we want if present
player_sum_candidates = {
    "Disposals":   "team_disposals",
    "Goals":       "team_goals",
    "Clearances":  "team_clearances",
    "Tackles":     "team_tackles",
    "Marks":       "team_marks",           # optional
}
PLAYER_SUM_COLS = {k:v for k,v in player_sum_candidates.items() if k in cols_available}

team_first_candidates = {
    "Team Score":      "team_score",
    "Team Inside 50":  "team_inside50",
    "Team Turnovers":  "team_turnovers",
    "Team Free Kicks": "team_free_kicks",
}
TEAM_FIRST_COLS = {k:v for k,v in team_first_candidates.items() if k in cols_available}

# Build SELECT list dynamically
select_cols = need_base + list(PLAYER_SUM_COLS.keys()) + list(TEAM_FIRST_COLS.keys())
if maybe_opp: select_cols.append("Opponent")
sel = ", ".join([f'"{c}"' for c in select_cols if c in cols_available])

sql = f"SELECT {sel} FROM {args.source}"
params = {}
if args.only_after:
    sql += ' WHERE "Date" > :after'
    params["after"] = args.only_after

df = pd.read_sql(text(sql), engine, params=params)
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df = df.dropna(subset=["Date"]).copy()

for c in ["Team","Round","Venue","Timeslot"]:
    if c in df.columns:
        df[c] = df[c].astype(str).str.strip()
if "Opponent" in df.columns:
    df["Opponent"] = df["Opponent"].astype(str).str.strip()

# numeric coercion
for c in PLAYER_SUM_COLS:
    df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
for c in TEAM_FIRST_COLS:
    df[c] = pd.to_numeric(df[c], errors="coerce")

# ------------ Collapse to one row per (Date, Team) ------------
agg_dict = {**{k:"sum" for k in PLAYER_SUM_COLS},
            **{k:"first" for k in TEAM_FIRST_COLS},
            "Round":"first","Venue":"first","Timeslot":"first"}
tg = (df.sort_values(["Date","Team"])
        .groupby(["Date","Team"], as_index=False)
        .agg(agg_dict)
        .rename(columns={**PLAYER_SUM_COLS, **TEAM_FIRST_COLS}))

tg["season"]  = tg["Date"].dt.year.astype(int)

# New: explicit primary/secondary, and home as their union
tg["is_primary_home"]   = tg.apply(lambda r: compute_is_primary_home(r.get("Team"), r.get("Venue")), axis=1).astype(int)
tg["is_secondary_home"] = tg.apply(lambda r: is_secondary_home(r.get("Team"), r.get("Venue")), axis=1).astype(int)
tg["is_home"]           = tg.apply(lambda r: compute_is_home(r.get("Team"), r.get("Venue")), axis=1).astype(int)
tg["is_away"]           = 1 - tg["is_home"]

tg["points_for"] = pd.to_numeric(tg.get("team_score"), errors="coerce")

tg["Team_State"]   = tg["Team"].map(team_state)
tg["Venue_State"]  = tg["Venue"].map(venue_state)
tg["is_interstate"] = tg.apply(lambda r: is_interstate(r.get("Team"), r.get("Venue")), axis=1).astype(int)

# Optional Yes/No string (if you want it for easy display)
tg["interstate"] = np.where(tg["is_interstate"] == 1, "Yes", "No")

# Ensure we have Opponent on each row:
if "Opponent" not in df.columns:
    # pull from team_games produced by 501
    try:
        tg_opp = pd.read_sql(text('SELECT "Date","Team","Opponent" FROM team_games'), engine)
        tg_opp["Date"] = pd.to_datetime(tg_opp["Date"], errors="coerce")
        tg_opp = tg_opp.dropna(subset=["Date"])
        tg_opp["Team"] = tg_opp["Team"].astype(str).str.strip()
        tg_opp["Opponent"] = tg_opp["Opponent"].astype(str).str.strip()
        tg = tg.merge(tg_opp, on=["Date","Team"], how="left")
    except Exception:
        tg["Opponent"] = None  # concede_* will be NaN in this fallback
else:
    # We aggregated; keep a single opponent per team-date by first value in raw
    raw_opp = (df.sort_values(["Date","Team","Opponent"])
                 .dropna(subset=["Opponent"])
                 .drop_duplicates(subset=["Date","Team"])[["Date","Team","Opponent"]])
    tg = tg.merge(raw_opp, on=["Date","Team"], how="left")

# ------------ Build concede_* for the same game ------------
# Using the opponent's team totals on that date.
concede_map = {}
for src, dst in [
    ("team_disposals",  "concede_disposals"),
    ("team_tackles",    "concede_tackles"),
    ("team_marks",      "concede_marks"),
    ("team_clearances", "concede_clearances"),
]:
    if src in tg.columns:
        concede_map[src] = dst

if len(concede_map):
    opp_view = tg[["Date","Team"] + list(concede_map.keys())].copy()
    opp_view = opp_view.rename(columns={"Team":"Opponent", **concede_map})
    tg = tg.merge(opp_view, on=["Date","Opponent"], how="left")

# ------------ Feature computation ------------
WIN_SIZES = [int(w.strip()) for w in args.windows.split(",") if w.strip()]

# All metrics to compute windows for (team + concede)
METRICS = {}
for name, col in [
    ("score",          "points_for"),
    ("disposals",      "team_disposals"),
    ("goals",          "team_goals"),
    ("clearances",     "team_clearances"),
    ("tackles",        "team_tackles"),
    ("marks",          "team_marks"),
    ("inside50",       "team_inside50"),
    ("turnovers",      "team_turnovers"),
    ("free_kicks",     "team_free_kicks"),
    ("concede_disposals",  "concede_disposals"),
    ("concede_tackles",    "concede_tackles"),
    ("concede_marks",      "concede_marks"),
    ("concede_clearances", "concede_clearances"),
]:
    if col in tg.columns:
        METRICS[name] = col

tg = tg.sort_values(["season","Team","Date"]).reset_index(drop=True)

def _series_for_mode(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    return s if args.stats_mode == "inclusive" else s.shift(1)

def _season_to_date_subset(s: pd.Series, mask: pd.Series, how: str):
    x = _series_for_mode(s)
    masked = x[mask]
    if how == "mean":
        r = masked.expanding(min_periods=1).mean()
    elif how == "min":
        r = masked.expanding(min_periods=1).min()
    elif how == "max":
        r = masked.expanding(min_periods=1).max()
    else:
        raise ValueError(how)
    return r.reindex(s.index).ffill()

def _rolling_last_n_subset(s: pd.Series, mask: pd.Series, w: int, how: str):
    x = _series_for_mode(s)
    masked = x[mask]
    roll = masked.rolling(w, min_periods=1)
    if how == "mean":
        r = roll.mean()
    elif how == "min":
        r = roll.min()
    elif how == "max":
        r = roll.max()
    else:
        raise ValueError(how)
    return r.reindex(s.index).ffill()

def compute_features(g: pd.DataFrame) -> pd.DataFrame:
    # if we pass include_groups=False (below), reattach keys here
    if "season" not in g.columns or "Team" not in g.columns:
        season, team = g.name
        g = g.copy()
        g["season"], g["Team"] = season, team

    if args.exclude_finals_from_rolling:
        finals_mask = g["Round"].apply(is_finals_round)
        mask_roll_overall = ~finals_mask
    else:
        mask_roll_overall = pd.Series(True, index=g.index)

    mask_home = (g["is_home"] == 1)
    mask_away = (g["is_away"] == 1)

    newcols = {}  # collect everything here

    for name, col in METRICS.items():
        s_raw = pd.to_numeric(g[col], errors="coerce").fillna(0.0)
        x = _series_for_mode(s_raw)

        # season-to-date
        newcols[f"season_avg_{name}"] = x.expanding(min_periods=1).mean()
        newcols[f"season_min_{name}"] = x.expanding(min_periods=1).min()
        newcols[f"season_max_{name}"] = x.expanding(min_periods=1).max()

        newcols[f"season_avg_home_{name}"] = _season_to_date_subset(s_raw, mask_home, "mean")
        newcols[f"season_min_home_{name}"] = _season_to_date_subset(s_raw, mask_home, "min")
        newcols[f"season_max_home_{name}"] = _season_to_date_subset(s_raw, mask_home, "max")

        newcols[f"season_avg_away_{name}"] = _season_to_date_subset(s_raw, mask_away, "mean")
        newcols[f"season_min_away_{name}"] = _season_to_date_subset(s_raw, mask_away, "min")
        newcols[f"season_max_away_{name}"] = _season_to_date_subset(s_raw, mask_away, "max")

        # rolling windows
        for w in WIN_SIZES:
            tmp = _series_for_mode(s_raw).where(mask_roll_overall, np.nan)
            roll = tmp.rolling(window=w, min_periods=1)
            newcols[f"{name}_avg_last_{w}"] = roll.mean()
            newcols[f"{name}_min_last_{w}"] = roll.min()
            newcols[f"{name}_max_last_{w}"] = roll.max()

            newcols[f"{name}_avg_home_last_{w}"] = _rolling_last_n_subset(s_raw, mask_home, w, "mean")
            newcols[f"{name}_min_home_last_{w}"] = _rolling_last_n_subset(s_raw, mask_home, w, "min")
            newcols[f"{name}_max_home_last_{w}"] = _rolling_last_n_subset(s_raw, mask_home, w, "max")

            newcols[f"{name}_avg_away_last_{w}"] = _rolling_last_n_subset(s_raw, mask_away, w, "mean")
            newcols[f"{name}_min_away_last_{w}"] = _rolling_last_n_subset(s_raw, mask_away, w, "min")
            newcols[f"{name}_max_away_last_{w}"] = _rolling_last_n_subset(s_raw, mask_away, w, "max")

    # one join = no fragmentation
    g = pd.concat([g, pd.DataFrame(newcols, index=g.index)], axis=1)
    return g

tg_out = (
    tg.groupby(["season","Team"], group_keys=False)
      .apply(compute_features, include_groups=False)   # <- key change
      .reset_index(drop=True)
)

# ------------ Save (staging -> drop & swap) ------------
dest = args.dest
staging = f"{dest}__staging"

tg_out.to_sql(staging, engine, if_exists="replace", index=False,
              method="multi", chunksize=args.chunksize)

cols = list(tg_out.columns)
cols_quoted = ", ".join([f'"{c}"' for c in cols])

idx_team_season_date = f'idx_{dest.lower()}_team_season_date'
idx_team_date        = f'idx_{dest.lower()}_team_date'

ddl_indexes = f'''
CREATE INDEX IF NOT EXISTS {idx_team_season_date}
  ON "{dest}" ("Team","season","Date" DESC);
CREATE INDEX IF NOT EXISTS {idx_team_date}
  ON "{dest}" ("Team","Date" DESC);
'''

# explicit list for view stability
col_list = ", ".join([f'"{c}"' for c in cols])

ddl_latest_overall_drop = f'DROP VIEW IF EXISTS {dest}_latest CASCADE;'
ddl_latest_current_drop = f'DROP VIEW IF EXISTS {dest}_latest_current CASCADE;'

ddl_latest_overall = f'''
CREATE VIEW {dest}_latest AS
WITH latest AS (
    SELECT DISTINCT ON (tp."Team")
           {col_list}
    FROM "{dest}" tp
    ORDER BY tp."Team", tp."Date" DESC
)
SELECT
    l.*,
    t."elo"               AS elo_rating,
    t."glicko"            AS glicko_rating,
    t."glicko_rd"         AS glicko_rd,
    t."glicko_vol"        AS glicko_vol,
    t."season_wins"       AS season_wins,
    t."season_losses"     AS season_losses,
    t."season_draws"      AS season_draws,
    t."season_points_for"      AS season_points_for,
    t."season_points_against"  AS season_points_against,
    t."season_percentage" AS season_percentage,
    t."ladder_points"     AS ladder_points,
    t."ladder_position"   AS ladder_position,
    t."season_surprise"   AS season_surprise
FROM latest l
LEFT JOIN teams t
  ON t."Team" = l."Team" AND t."season" = l."season";
'''

ddl_latest_current = f'''
CREATE VIEW {dest}_latest_current AS
SELECT DISTINCT ON ("Team") {col_list}
FROM "{dest}"
WHERE season = (SELECT MAX(season) FROM "{dest}")
ORDER BY "Team", "Date" DESC;
'''

with engine.begin() as con:
    # drop dependent views + table, then swap staging in
    con.execute(text(ddl_latest_overall_drop))
    con.execute(text(ddl_latest_current_drop))
    con.execute(text(f'DROP TABLE IF EXISTS "{dest}" CASCADE;'))
    con.execute(text(f'ALTER TABLE "{staging}" RENAME TO "{dest}"'))
    con.execute(text(ddl_indexes))
    con.execute(text(ddl_latest_overall))
    con.execute(text(ddl_latest_current))

print(f"✅ Wrote {len(tg_out):,} rows to '{dest}' and added concede_* metrics "
      f"(windows={WIN_SIZES}, stats_mode={args.stats_mode}).")
print(f"🏟️ Added home-ground columns: is_primary_home, is_secondary_home, is_home, is_away.")
print(f"🔎 Created indexes: {idx_team_season_date}, {idx_team_date}")
print(f"🔎 Created views: {dest}_latest (overall), {dest}_latest_current (current season)")
