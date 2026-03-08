from pathlib import Path
import sys
import os
import math
import json
import argparse

import pandas as pd
import numpy as np
import joblib
from sqlalchemy import text as sqtext

# ---------- Project root / imports ----------
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from db_connect import get_engine
from util import HOME_GROUNDS, SECONDARY_HOME_GROUNDS, TEAM_ALIASES, TEAM_STATE, VENUE_STATE

engine = get_engine()

MODELS_DIR = (ROOT / "machine_learning" / "models").resolve()
MODEL_PATH = MODELS_DIR / "team_winner_logreg.pkl"
META_PATH = MODELS_DIR / "team_winner_meta.json"

# ---------- Args ----------
ap = argparse.ArgumentParser()
ap.add_argument("--debug-home", default=None, help="Home team name to debug")
ap.add_argument("--debug-away", default=None, help="Away team name to debug")
ap.add_argument("--debug-date", default=None, help="Fixture date to debug: YYYY-MM-DD")
args = ap.parse_args()

# ---------- Normalization knobs ----------
DRAW_BASE_RATE = 10 / 846  # historical draw rate
SCALE_DRAW_BY_CLOSENESS = True

# ---------- Helpers ----------
def normalize_team_key(team: str) -> str:
    t = (team or "").strip()
    return t if t in HOME_GROUNDS else TEAM_ALIASES.get(t, t)

def is_home(team: str, venue: str) -> int:
    if team is None or venue is None:
        return 0
    return int(str(venue).strip() in HOME_GROUNDS.get(str(team).strip(), []))

def timeslot_dummies(ts: str):
    ts = (ts or "").strip().lower()
    return {
        "ts_day": 1 if ts == "day" else 0,
        "ts_night": 1 if ts == "night" else 0,
        "ts_twilight": 1 if ts == "twilight" else 0,
        "ts_unknown": 1 if ts not in ("day", "night", "twilight") else 0,
    }

def team_state(team: str) -> str:
    return TEAM_STATE.get((team or "").strip(), "")

def venue_state(venue: str) -> str:
    return VENUE_STATE.get((venue or "").strip(), "")

def is_interstate(team: str, venue: str) -> int:
    ts, vs = team_state(team), venue_state(venue)
    return int(bool(ts) and bool(vs) and ts != vs)

def is_secondary_home(team: str, venue: str) -> int:
    if team is None or venue is None:
        return 0
    t = str(team).strip()
    v = str(venue).strip()
    return int(v in SECONDARY_HOME_GROUNDS.get(t, []))

def safe_num(x, default=0.0):
    try:
        v = float(x)
        if math.isnan(v):
            return default
        return v
    except Exception:
        return default

def normalize_for_lookup(df: pd.DataFrame, col: str) -> pd.DataFrame:
    df = df.copy()
    df[col] = df[col].astype(str).str.strip().map(normalize_team_key)
    return df

# ---------- Load model + metadata ----------
clf = joblib.load(MODEL_PATH)

meta = {}
if META_PATH.exists():
    try:
        meta = json.loads(META_PATH.read_text())
    except Exception:
        try:
            meta = joblib.load(META_PATH)
        except Exception:
            meta = {}

feature_cols = meta.get("feature_cols")
if not feature_cols:
    raise RuntimeError("team_winner_meta.json missing feature_cols. Re-train the model.")

# ---------- Load upcoming fixtures ----------
up = pd.read_sql('SELECT * FROM upcoming_games', engine)

required_cols = ["Home Team", "Away Team", "Venue", "Timeslot", "Date"]
for c in required_cols:
    if c not in up.columns:
        raise RuntimeError(f"upcoming_games missing column: {c}")

up["Date"] = pd.to_datetime(up["Date"], errors="coerce")
up = up.dropna(subset=["Date"]).copy()

for c in ["Home Team", "Away Team", "Venue", "Timeslot"]:
    up[c] = up[c].astype(str).str.strip()

up["home_key"] = up["Home Team"].map(normalize_team_key)
up["away_key"] = up["Away Team"].map(normalize_team_key)

today = pd.Timestamp.today().normalize()
up_future = up[up["Date"] >= today].sort_values(["Date", "Home Team", "Away Team"]).copy()

if up_future.empty:
    print("⚠️ No future fixtures found in upcoming_games.")
    sys.exit(0)

# ---------- Build mutual team rows ----------
home_rows = (
    up_future[["Date", "Venue", "Timeslot", "home_key", "away_key", "Home Team", "Away Team"]]
    .rename(columns={"home_key": "Team", "away_key": "Opponent"})
)

away_rows = (
    up_future[["Date", "Venue", "Timeslot", "away_key", "home_key", "Home Team", "Away Team"]]
    .rename(columns={"away_key": "Team", "home_key": "Opponent"})
)

ng_df = pd.concat([home_rows, away_rows], ignore_index=True).copy()
ng_df = ng_df.sort_values(["Team", "Date", "Opponent"]).drop_duplicates(
    subset=["Team", "Opponent", "Date", "Venue"], keep="first"
).reset_index(drop=True)

ng_df["pair_key"] = (
    ng_df["Date"].dt.strftime("%Y-%m-%d")
    + "|"
    + ng_df[["Team", "Opponent"]].apply(lambda x: "|".join(sorted(x)), axis=1)
)

pair_counts = ng_df["pair_key"].value_counts()
ng_df = ng_df[ng_df["pair_key"].map(pair_counts) == 2].copy().reset_index(drop=True)

if ng_df.empty:
    print("⚠️ No mutual next-fixture pairs found. Win probabilities will not be written.")
    sys.exit(0)

# ---------- Load latest ratings by team ----------
teams_tbl = pd.read_sql('SELECT * FROM teams', engine)
if teams_tbl.empty:
    raise RuntimeError("Table 'teams' is empty. Run ratings first.")

teams_tbl["Team"] = teams_tbl["Team"].astype(str).str.strip().map(normalize_team_key)
teams_tbl["Date"] = pd.to_datetime(teams_tbl.get("Date"), errors="coerce")

sort_cols = ["Team"]
if "Date" in teams_tbl.columns:
    sort_cols.append("Date")
if "season" in teams_tbl.columns:
    sort_cols.append("season")

teams_latest = (
    teams_tbl.sort_values(sort_cols)
    .groupby("Team", as_index=False)
    .tail(1)
)

rating_cols = ["Team", "elo", "glicko", "glicko_rd", "glicko_vol"]
rating_cols = [c for c in rating_cols if c in teams_latest.columns]
rat_map = teams_latest[rating_cols].set_index("Team").to_dict(orient="index")

# ---------- Rest days ----------
tg_dates = pd.read_sql('SELECT "Team","Date" FROM team_games', engine)
tg_dates["Date"] = pd.to_datetime(tg_dates["Date"], errors="coerce")
tg_dates = tg_dates.dropna(subset=["Date"]).copy()
tg_dates["Team"] = tg_dates["Team"].astype(str).str.strip().map(normalize_team_key)

last_played = (
    tg_dates.sort_values(["Team", "Date"])
    .groupby("Team", as_index=False)
    .tail(1)
    .set_index("Team")["Date"]
    .to_dict()
)

# ---------- Full team_precompute history ----------
tp_all = pd.read_sql('SELECT * FROM team_precompute', engine)
if tp_all.empty:
    raise RuntimeError("team_precompute is empty. Run the precompute build first.")

tp_all["Date"] = pd.to_datetime(tp_all["Date"], errors="coerce")
tp_all = tp_all.dropna(subset=["Date"]).copy()
tp_all["Team"] = tp_all["Team"].astype(str).str.strip().map(normalize_team_key)
tp_all = tp_all.sort_values(["Team", "Date"]).reset_index(drop=True)

tp_needed = [
    "Team",
    "Date",
    "score_avg_last_5",
    "inside50_avg_last_5",
    "turnovers_avg_last_5",
    "free_kicks_avg_last_5",
    "disposals_avg_last_5",
    "clearances_avg_last_5",
    "tackles_avg_last_5",
    "marks_avg_last_5",
    "concede_disposals_avg_last_5",
    "concede_tackles_avg_last_5",
    "concede_marks_avg_last_5",
    "concede_clearances_avg_last_5",
    "is_interstate",
    "is_secondary_home",
]
tp_keep = [c for c in tp_needed if c in tp_all.columns]
tp_all = tp_all[tp_keep].copy()

tp_grouped = {team: grp.reset_index(drop=True) for team, grp in tp_all.groupby("Team", sort=False)}

def get_tp_snapshot(team: str, fixture_date: pd.Timestamp) -> dict:
    grp = tp_grouped.get(team)
    if grp is None or grp.empty:
        return {}
    rows = grp[grp["Date"] <= fixture_date]
    if rows.empty:
        return {}
    return rows.iloc[-1].to_dict()

# ---------- Build prediction rows ----------
rows = []

for r in ng_df.itertuples(index=False):
    A = r.Team
    B = r.Opponent
    date = r.Date
    venue = r.Venue
    ts = r.Timeslot

    RA = rat_map.get(A, {"elo": 1500.0, "glicko": 1500.0, "glicko_rd": 350.0, "glicko_vol": 0.12})
    RB = rat_map.get(B, {"elo": 1500.0, "glicko": 1500.0, "glicko_rd": 350.0, "glicko_vol": 0.12})

    eloA, eloB = safe_num(RA.get("elo"), 1500.0), safe_num(RB.get("elo"), 1500.0)
    gA, gB = safe_num(RA.get("glicko"), 1500.0), safe_num(RB.get("glicko"), 1500.0)
    rdA, rdB = safe_num(RA.get("glicko_rd"), 350.0), safe_num(RB.get("glicko_rd"), 350.0)
    vA, vB = safe_num(RA.get("glicko_vol"), 0.12), safe_num(RB.get("glicko_vol"), 0.12)

    rec = {
        "Team": A,
        "Opponent": B,
        "Date": date,
        "Venue": venue,
        "Timeslot": ts,
        "Home Team": getattr(r, "Home Team", None) if hasattr(r, "Home Team") else None,
        "Away Team": getattr(r, "Away Team", None) if hasattr(r, "Away Team") else None,
    }

    rec["elo_diff"] = eloA - eloB
    rec["glicko_diff"] = gA - gB
    rec["rd_diff"] = rdA - rdB
    rec["vol_diff"] = vA - vB

    hA, hB = is_home(A, venue), is_home(B, venue)
    rec["home_edge"] = hA - hB

    rec.update(timeslot_dummies(ts))
    rec["elo_exp"] = 1.0 / (1.0 + 10.0 ** (-(eloA - eloB) / 400.0))

    A_inter, B_inter = is_interstate(A, venue), is_interstate(B, venue)
    rec["interstate_edge"] = A_inter - B_inter

    A_sec, B_sec = is_secondary_home(A, venue), is_secondary_home(B, venue)
    rec["secondary_home_edge"] = A_sec - B_sec

    A_tp = get_tp_snapshot(A, date)
    B_tp = get_tp_snapshot(B, date)

    rec["A_snapshot_date"] = A_tp.get("Date")
    rec["B_snapshot_date"] = B_tp.get("Date")

    def getv(d, k):
        return safe_num(d.get(k, np.nan), np.nan)

    for name in [
        "score_avg_last_5",
        "inside50_avg_last_5",
        "turnovers_avg_last_5",
        "free_kicks_avg_last_5",
        "disposals_avg_last_5",
        "clearances_avg_last_5",
        "tackles_avg_last_5",
        "marks_avg_last_5",
    ]:
        if (name in A_tp) and (name in B_tp):
            rec[f"{name}_diff5"] = getv(A_tp, name) - getv(B_tp, name)

    if "disposals_avg_last_5" in A_tp and "concede_disposals_avg_last_5" in B_tp:
        rec["disp_matchup5"] = getv(A_tp, "disposals_avg_last_5") - getv(B_tp, "concede_disposals_avg_last_5")

    if "tackles_avg_last_5" in A_tp and "concede_tackles_avg_last_5" in B_tp:
        rec["tack_matchup5"] = getv(A_tp, "tackles_avg_last_5") - getv(B_tp, "concede_tackles_avg_last_5")

    if "marks_avg_last_5" in A_tp and "concede_marks_avg_last_5" in B_tp:
        rec["marks_matchup5"] = getv(A_tp, "marks_avg_last_5") - getv(B_tp, "concede_marks_avg_last_5")

    if "clearances_avg_last_5" in A_tp and "concede_clearances_avg_last_5" in B_tp:
        rec["clear_matchup5"] = getv(A_tp, "clearances_avg_last_5") - getv(B_tp, "concede_clearances_avg_last_5")

    ldA = last_played.get(A, pd.NaT)
    ldB = last_played.get(B, pd.NaT)
    dA = (date - ldA).days if pd.notna(ldA) else np.nan
    dB = (date - ldB).days if pd.notna(ldB) else np.nan
    rec["rest_diff"] = (dA if pd.notna(dA) else 0.0) - (dB if pd.notna(dB) else 0.0)

    rows.append(rec)

F = pd.DataFrame(rows)

if F.empty:
    print("⚠️ No prediction rows assembled.")
    sys.exit(0)

# ---------- Ensure all expected features exist ----------
for c in feature_cols:
    if c not in F.columns:
        F[c] = 0.0

for c in feature_cols:
    F[c] = pd.to_numeric(F[c], errors="coerce").fillna(0.0)

# ---------- Raw probabilities ----------
P_raw = clf.predict_proba(F[feature_cols].astype(float))[:, 1]
F["Win_Probability_raw"] = np.clip(P_raw, 0.0, 1.0)

# ---------- Pair-normalise + draw bucket ----------
F["pair_key"] = (
    pd.to_datetime(F["Date"]).dt.strftime("%Y-%m-%d")
    + "|"
    + F[["Team", "Opponent"]].apply(lambda x: "|".join(sorted(x)), axis=1)
)

def adjust_pair(g: pd.DataFrame) -> pd.DataFrame:
    g = g.copy()

    if len(g) != 2:
        g["Win_Probability"] = 0.0
        g["Draw_Probability"] = 0.0
        return g

    pa = float(g.iloc[0]["Win_Probability_raw"])
    pb = float(g.iloc[1]["Win_Probability_raw"])

    s = max(pa + pb, 1e-12)
    pa_norm = pa / s
    pb_norm = pb / s

    p_draw = float(DRAW_BASE_RATE)
    if SCALE_DRAW_BY_CLOSENESS:
        closeness = 2.0 * min(pa_norm, pb_norm)
        p_draw = float(DRAW_BASE_RATE) * float(closeness)

    g.loc[g.index[0], "Win_Probability"] = (1.0 - p_draw) * pa_norm * 100.0
    g.loc[g.index[1], "Win_Probability"] = (1.0 - p_draw) * pb_norm * 100.0
    g["Draw_Probability"] = p_draw * 100.0
    return g

F = F.groupby("pair_key", group_keys=False).apply(adjust_pair)

F["Win_Probability"] = pd.to_numeric(F["Win_Probability"], errors="coerce").fillna(0.0)
F["Draw_Probability"] = pd.to_numeric(F["Draw_Probability"], errors="coerce").fillna(0.0)

# ---------- Optional debug ----------
if args.debug_home and args.debug_away:
    dbg_home = normalize_team_key(args.debug_home)
    dbg_away = normalize_team_key(args.debug_away)

    dbg = F[
        (
            ((F["Team"] == dbg_home) & (F["Opponent"] == dbg_away))
            | ((F["Team"] == dbg_away) & (F["Opponent"] == dbg_home))
        )
    ].copy()

    if args.debug_date:
        dbg_date = pd.to_datetime(args.debug_date, errors="coerce")
        if pd.notna(dbg_date):
            dbg = dbg[dbg["Date"] == dbg_date]

    if not dbg.empty:
        debug_cols = [
            "Date",
            "Team",
            "Opponent",
            "Venue",
            "Timeslot",
            "A_snapshot_date",
            "B_snapshot_date",
            "Win_Probability_raw",
            "Win_Probability",
            "Draw_Probability",
        ] + feature_cols

        debug_cols = [c for c in debug_cols if c in dbg.columns]
        print("\n🔎 Debug fixture rows:")
        print(dbg[debug_cols].to_string(index=False))
    else:
        print("\n⚠️ Debug fixture not found in assembled prediction rows.")

# ---------- Persist to team_precompute latest historical row per team ----------
with engine.begin() as con:
    con.exec_driver_sql('ALTER TABLE team_precompute ADD COLUMN IF NOT EXISTS "Next_Opponent" TEXT')
    con.exec_driver_sql('ALTER TABLE team_precompute ADD COLUMN IF NOT EXISTS "Next_Venue" TEXT')
    con.exec_driver_sql('ALTER TABLE team_precompute ADD COLUMN IF NOT EXISTS "Next_Timeslot" TEXT')
    con.exec_driver_sql('ALTER TABLE team_precompute ADD COLUMN IF NOT EXISTS "Next_Date" DATE')
    con.exec_driver_sql('ALTER TABLE team_precompute ADD COLUMN IF NOT EXISTS "Win_Probability" DOUBLE PRECISION')
    con.exec_driver_sql('ALTER TABLE team_precompute ADD COLUMN IF NOT EXISTS "Draw_Probability" DOUBLE PRECISION')

tp_dates = pd.read_sql(
    'SELECT "Team", MAX("Date") AS max_date FROM team_precompute GROUP BY "Team"',
    engine
)
tp_dates["Team"] = tp_dates["Team"].astype(str).str.strip().map(normalize_team_key)
tp_dates["max_date"] = pd.to_datetime(tp_dates["max_date"], errors="coerce")
latest_map = dict(zip(tp_dates["Team"], tp_dates["max_date"]))

updated = 0
with engine.begin() as con:
    for r in F.itertuples(index=False):
        team = r.Team
        opp = r.Opponent
        venue = r.Venue
        timeslot = r.Timeslot
        next_date = pd.to_datetime(r.Date).date() if pd.notna(r.Date) else None
        winp = float(r.Win_Probability)
        drawp = float(r.Draw_Probability)

        latest_date = latest_map.get(team)
        if pd.isna(latest_date):
            continue

        params = {
            "team": team,
            "date": latest_date,
            "opp": opp,
            "venue": venue,
            "timeslot": timeslot,
            "next_date": next_date,
            "p": winp,
            "pd": drawp,
        }

        sql = (
            'UPDATE team_precompute '
            'SET "Next_Opponent" = :opp, '
            '"Next_Venue" = :venue, '
            '"Next_Timeslot" = :timeslot, '
            '"Next_Date" = :next_date, '
            '"Win_Probability" = :p, '
            '"Draw_Probability" = :pd '
            'WHERE "Team" = :team AND "Date" = :date'
        )
        res = con.execute(sqtext(sql), params)
        updated += res.rowcount if hasattr(res, "rowcount") else 1

print(f'✅ Wrote next-game fields and probabilities for {updated} team rows in team_precompute.')

# ---------- Refresh latest views ----------
def refresh_team_precompute_views(dest="team_precompute"):
    drop_overall = f'DROP VIEW IF EXISTS {dest}_latest CASCADE;'
    drop_current = f'DROP VIEW IF EXISTS {dest}_latest_current CASCADE;'

    with engine.begin() as con:
        cols = [
            r[0]
            for r in con.execute(
                sqtext("""
                    SELECT column_name
                    FROM information_schema.columns
                    WHERE table_name = :t
                    ORDER BY ordinal_position
                """),
                {"t": dest},
            ).fetchall()
        ]

        col_list_tp = ", ".join(f'tp."{c}"' for c in cols)

        ddl_overall = f'''
        CREATE VIEW {dest}_latest AS
        WITH latest AS (
            SELECT DISTINCT ON (tp."Team")
                   {col_list_tp}
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
            t."season_percentage" AS season_percentage,
            t."ladder_points"     AS ladder_points,
            t."ladder_position"   AS ladder_position,
            t."season_surprise"   AS surprise_results
        FROM latest l
        LEFT JOIN teams t
            ON t."Team" = l."Team" AND t."season" = l."season";
        '''

        ddl_current = f'''
        CREATE VIEW {dest}_latest_current AS
        SELECT DISTINCT ON (tp."Team") {col_list_tp}
        FROM "{dest}" tp
        WHERE tp."season" = (SELECT MAX("season") FROM "{dest}")
        ORDER BY tp."Team", tp."Date" DESC;
        '''

        con.execute(sqtext(drop_overall))
        con.execute(sqtext(drop_current))
        con.execute(sqtext(ddl_overall))
        con.execute(sqtext(ddl_current))

refresh_team_precompute_views("team_precompute")