# 502_build_team_precompute.py
# One row per TEAM–GAME with rolling + season-to-date features.
# - Uses your HOME_GROUNDS mapping EXACTLY (no venue aliasing)
# - is_home / is_away flags from raw Venue membership
# - Rolling windows overall + home-only + away-only
# - Finals (EF/QF/SF/PF/GF) can be excluded from *rolling* windows
# - --stats-mode inclusive|pre  (inclusive = includes current game; pre = excludes current game)

import argparse
import numpy as np
import pandas as pd
from sqlalchemy import text
from db_connect import get_engine

# ---------- CLI ----------
ap = argparse.ArgumentParser()
ap.add_argument("--source", default="player_stats")
ap.add_argument("--dest",   default="team_precompute")
ap.add_argument("--windows", default="3,5,10", help="Comma-separated rolling windows, e.g. 3,5,10")
ap.add_argument("--exclude-finals-from-rolling", action="store_true",
                help="Exclude EF/QF/SF/PF/GF from rolling windows (rows still kept).")
ap.add_argument("--stats-mode", choices=["inclusive","pre"], default="inclusive",
                help="inclusive = include current game; pre = exclude current game (pre-game stats).")
ap.add_argument("--only-after", default=None, help="YYYY-MM-DD to restrict input")
ap.add_argument("--chunksize", type=int, default=20000)
args = ap.parse_args()

engine = get_engine()

# ---------- Your home-ground mapping (used verbatim) ----------
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

FINALS = {"EF","QF","SF","PF","GF"}
def is_finals_round(x):
    if x is None: return False
    return str(x).strip().upper() in FINALS

def compute_is_home(team: str, venue: str) -> int:
    if pd.isna(team) or pd.isna(venue): 
        return 0
    team_key = str(team).strip()
    venue_key = str(venue).strip()
    homes = HOME_GROUNDS.get(team_key, [])
    return int(venue_key in homes)

# ---------- Columns ----------
PLAYER_SUM_COLS = {
    "Disposals": "team_disposals",
    "Goals": "team_goals",
    "Clearances": "team_clearances",
    "Tackles": "team_tackles",
}
TEAM_FIRST_COLS = {
    "Team Score": "team_score",
    "Team Inside 50": "team_inside50",
    "Team Turnovers": "team_turnovers",
    "Team Free Kicks": "team_free_kicks",
}
ID_COLS = ["Date","Team","Round","Venue","Timeslot"]

# ---------- Read minimal ----------
all_cols = ID_COLS + list(PLAYER_SUM_COLS.keys()) + list(TEAM_FIRST_COLS.keys())
sel = ", ".join([f'"{c}"' for c in all_cols])
sql = f"SELECT {sel} FROM {args.source}"
params = {}
if args.only_after:
    sql += ' WHERE "Date" > :after'
    params["after"] = args.only_after
df = pd.read_sql(text(sql), engine, params=params)

# ---------- Clean ----------
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df = df.dropna(subset=["Date"]).copy()

for c in ["Team","Round","Venue","Timeslot"]:
    df[c] = df[c].astype(str).str.strip()

# No aliasing of team names or venues
for c in PLAYER_SUM_COLS:
    df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
for c in TEAM_FIRST_COLS:
    df[c] = pd.to_numeric(df[c], errors="coerce")

# ---------- Collapse to ONE ROW PER (Date, Team) ----------
agg_dict = {**{k:"sum" for k in PLAYER_SUM_COLS},
            **{k:"first" for k in TEAM_FIRST_COLS},
            "Round":"first", "Venue":"first", "Timeslot":"first"}

tg = (df.sort_values(["Date","Team"])
        .groupby(["Date","Team"], as_index=False)
        .agg(agg_dict))

tg = tg.rename(columns={**PLAYER_SUM_COLS, **TEAM_FIRST_COLS})
tg["season"]  = tg["Date"].dt.year.astype(int)
tg["is_home"] = tg.apply(lambda r: compute_is_home(r["Team"], r["Venue"]), axis=1).astype(int)
tg["is_away"] = 1 - tg["is_home"]

# Convenience score metric
tg["points_for"] = tg["team_score"].astype(float)

# Finals flag (for rolling exclusion only)
tg["is_finals"] = tg["Round"].apply(is_finals_round)

# ---------- Metrics & windows ----------
METRICS = {
    "score": "points_for",
    "disposals": "team_disposals",
    "goals": "team_goals",
    "clearances": "team_clearances",
    "tackles": "team_tackles",
    "inside50": "team_inside50",
    "turnovers": "team_turnovers",
    "free_kicks": "team_free_kicks",
}
WIN_SIZES = [int(w.strip()) for w in args.windows.split(",") if w.strip()]

tg = tg.sort_values(["season","Team","Date"]).reset_index(drop=True)

# ---------- Helpers with inclusive/pre switch ----------
def _series_for_mode(s: pd.Series) -> pd.Series:
    """Return s (inclusive) or s.shift(1) (pre-game)."""
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
    # overall rolling mask (optionally exclude finals from rolling windows)
    if args.exclude_finals_from_rolling:
        mask_roll_overall = ~g["is_finals"]
    else:
        mask_roll_overall = pd.Series(True, index=g.index)

    mask_home = g["is_home"] == 1
    mask_away = g["is_away"] == 1

    for name, col in METRICS.items():
        s_raw = pd.to_numeric(g[col], errors="coerce").fillna(0.0)
        x = _series_for_mode(s_raw)

        # ---- season-to-date (overall)
        g[f"season_avg_{name}"] = x.expanding(min_periods=1).mean()
        g[f"season_min_{name}"] = x.expanding(min_periods=1).min()
        g[f"season_max_{name}"] = x.expanding(min_periods=1).max()

        # ---- season-to-date (home / away)
        g[f"season_avg_home_{name}"] = _season_to_date_subset(s_raw, mask_home, "mean")
        g[f"season_min_home_{name}"] = _season_to_date_subset(s_raw, mask_home, "min")
        g[f"season_max_home_{name}"] = _season_to_date_subset(s_raw, mask_home, "max")

        g[f"season_avg_away_{name}"] = _season_to_date_subset(s_raw, mask_away, "mean")
        g[f"season_min_away_{name}"] = _season_to_date_subset(s_raw, mask_away, "min")
        g[f"season_max_away_{name}"] = _season_to_date_subset(s_raw, mask_away, "max")

        # ---- rolling (overall/home/away)
        for w in WIN_SIZES:
            xx = _series_for_mode(s_raw)

            # overall rolling (respect finals exclusion for window)
            tmp = xx.where(mask_roll_overall, np.nan)
            roll = tmp.rolling(window=w, min_periods=1)
            g[f"{name}_avg_last_{w}"] = roll.mean()
            g[f"{name}_min_last_{w}"] = roll.min()
            g[f"{name}_max_last_{w}"] = roll.max()

            # home-only / away-only last N
            g[f"{name}_avg_home_last_{w}"] = _rolling_last_n_subset(s_raw, mask_home, w, "mean")
            g[f"{name}_min_home_last_{w}"] = _rolling_last_n_subset(s_raw, mask_home, w, "min")
            g[f"{name}_max_home_last_{w}"] = _rolling_last_n_subset(s_raw, mask_home, w, "max")

            g[f"{name}_avg_away_last_{w}"] = _rolling_last_n_subset(s_raw, mask_away, w, "mean")
            g[f"{name}_min_away_last_{w}"] = _rolling_last_n_subset(s_raw, mask_away, w, "min")
            g[f"{name}_max_away_last_{w}"] = _rolling_last_n_subset(s_raw, mask_away, w, "max")

    return g

tg = (tg.groupby(["season","Team"], group_keys=False)
        .apply(compute_features, include_groups=True))


# Keep identifiers + flags + base per-game metrics
METRIC_COLS = list(METRICS.values())
keep_ids = ["Date","season","Team","Round","Venue","Timeslot","is_home","is_away"]
tg_out = tg[keep_ids + METRIC_COLS + [c for c in tg.columns if c not in keep_ids + METRIC_COLS + ["is_finals"]]]

# ---------- Save safely (staging -> live without dropping the base table) ----------
dest = args.dest
staging = f"{dest}__staging"

# 1) Write to a staging table that has no dependents
tg_out.to_sql(staging, engine, if_exists="replace", index=False,
              method="multi", chunksize=args.chunksize)

cols = list(tg_out.columns)
cols_quoted = ", ".join([f'"{c}"' for c in cols])

with engine.begin() as con:
    # Does the live table already exist?
    dest_exists = con.execute(text("""
        SELECT to_regclass(:tname) IS NOT NULL
    """), {"tname": dest}).scalar()

    if dest_exists:
        # Keep the same table (so views remain valid), just replace rows
        con.execute(text(f'TRUNCATE "{dest}"'))
        con.execute(text(f'INSERT INTO "{dest}" ({cols_quoted}) SELECT {cols_quoted} FROM "{staging}"'))
        con.execute(text(f'DROP TABLE "{staging}"'))
    else:
        # First run: just rename staging -> live
        con.execute(text(f'ALTER TABLE "{staging}" RENAME TO "{dest}"'))

# ---------- Indexes to make the views turbo-fast ----------
idx_team_season_date = f'idx_{dest.lower()}_team_season_date'
idx_team_date        = f'idx_{dest.lower()}_team_date'
ddl_indexes = f'''
CREATE INDEX IF NOT EXISTS {idx_team_season_date}
  ON "{dest}" ("Team","season","Date" DESC);
CREATE INDEX IF NOT EXISTS {idx_team_date}
  ON "{dest}" ("Team","Date" DESC);
'''

# Explicit column list so the view’s shape is stable this run
col_list = ", ".join([f'"{c}"' for c in cols])  # 'cols' is from above (tg_out.columns)

# Overall latest per team (≈18 rows)
ddl_latest_overall_drop = f'DROP VIEW IF EXISTS {dest}_latest CASCADE;'
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
        t."elo"                 AS elo_rating,
        t."glicko"              AS glicko_rating,
        t."glicko_rd"           AS glicko_rd,
        t."glicko_vol"          AS glicko_vol,
        t."season_wins"         AS season_wins,
        t."season_losses"       AS season_losses,
        t."season_draws"        AS season_draws,
        t."season_percentage"   AS season_percentage,
        t."ladder_points"       AS ladder_points,
        t."ladder_position"     AS ladder_position,
        t."season_surprise"     AS surprise_results
    FROM latest l
    LEFT JOIN teams t
        ON t."Team" = l."Team" AND t."season" = l."season";
'''

# Latest per team in CURRENT season
ddl_latest_current_drop = f'DROP VIEW IF EXISTS {dest}_latest_current CASCADE;'
ddl_latest_current = f'''
CREATE VIEW {dest}_latest_current AS
SELECT DISTINCT ON ("Team") {col_list}
FROM "{dest}"
WHERE season = (SELECT MAX(season) FROM "{dest}")
ORDER BY "Team", "Date" DESC;
'''

with engine.begin() as con:
    con.execute(text(ddl_indexes))
    # drop then create (avoid "cannot drop columns from view")
    con.execute(text(ddl_latest_overall_drop))
    con.execute(text(ddl_latest_current_drop))
    con.execute(text(ddl_latest_overall))
    con.execute(text(ddl_latest_current))

print(f"🔎 Created indexes: {idx_team_season_date}, {idx_team_date}")
print(f"🔎 Created views: {dest}_latest (overall), {dest}_latest_current (current season)")

print(
    f"✅ Wrote {len(tg_out):,} rows to '{dest}' "
    f"(windows={WIN_SIZES}, stats_mode={args.stats_mode}, rolling finals "
    f"{'EXCLUDED' if args.exclude_finals_from_rolling else 'included'})."
)