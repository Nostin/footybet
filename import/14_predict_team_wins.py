from pathlib import Path
import sys, os, math
import pandas as pd
import numpy as np
import joblib
from sqlalchemy import text as sqtext
import json

# ---------- Project root / imports ----------
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from db_connect import get_engine
from util import HOME_GROUNDS, SECONDARY_HOME_GROUNDS, TEAM_ALIASES, TEAM_STATE, VENUE_STATE

engine = get_engine()

MODELS_DIR = (ROOT / "machine_learning" / "models").resolve()
MODEL_PATH = MODELS_DIR / "team_winner_logreg.pkl"
META_PATH  = MODELS_DIR / "team_winner_meta.json"

# ---------- Normalization knobs ----------
DRAW_BASE_RATE = 10 / 846  # ~0.01182 (your data)
SCALE_DRAW_BY_CLOSENESS = True  # draw ↑ when matchup ≈ 50/50

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
        "ts_day":       1 if ts == "day" else 0,
        "ts_night":     1 if ts == "night" else 0,
        "ts_twilight":  1 if ts == "twilight" else 0,
        "ts_unknown":   1 if ts not in ("day", "night", "twilight") else 0,
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

# ---------- Load winner model + meta ----------
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
    raise RuntimeError("team_winner_meta.json missing feature_cols. Re-train to write meta.")

# ---------- Build per-team next fixture ----------
up = pd.read_sql('SELECT * FROM upcoming_games', engine)
for c in ["Home Team","Away Team","Venue","Timeslot","Date"]:
    if c not in up.columns:
        raise RuntimeError(f"upcoming_games missing column: {c}")

up["Date"] = pd.to_datetime(up["Date"], errors="coerce")
up = up.dropna(subset=["Date"]).copy()
for c in ["Home Team","Away Team","Venue","Timeslot"]:
    up[c] = up[c].astype(str).str.strip()

up["home_key"] = up["Home Team"].map(normalize_team_key)
up["away_key"] = up["Away Team"].map(normalize_team_key)

today = pd.Timestamp.today().normalize()

# ---------- Build per-team next fixture (FIXED) ----------
up_future = up[up["Date"] >= today].sort_values("Date")

home_rows = (
    up_future[["Date","Venue","Timeslot","home_key","away_key"]]
    .rename(columns={"home_key":"Team","away_key":"Opponent"})
)

away_rows = (
    up_future[["Date","Venue","Timeslot","away_key","home_key"]]
    .rename(columns={"away_key":"Team","home_key":"Opponent"})
)

ng_df = (
    pd.concat([home_rows, away_rows], ignore_index=True)
      .sort_values(["Team", "Date"])          # critical
      .drop_duplicates(subset=["Team"], keep="first")
      .reset_index(drop=True)
)

# --- Keep only mutual next-fixture pairs ---
# A->B exists AND B->A exists on same date (safer) and same venue optional
ng_key = ng_df.copy()
ng_key["k"] = ng_key["Team"] + "|" + ng_key["Opponent"] + "|" + ng_key["Date"].dt.strftime("%Y-%m-%d")
ng_key["k_rev"] = ng_key["Opponent"] + "|" + ng_key["Team"] + "|" + ng_key["Date"].dt.strftime("%Y-%m-%d")

mutual = ng_key["k_rev"].isin(set(ng_key["k"]))
ng_df = ng_df[mutual].copy().reset_index(drop=True)

if ng_df.empty:
    print("⚠️ No mutual next-fixture pairs found (by team+opponent+date). Win probabilities will not be written.")

# ---------- Latest ratings ----------
teams_tbl = pd.read_sql('SELECT * FROM teams', engine)
if teams_tbl.empty:
    raise RuntimeError("Table 'teams' is empty. Run ratings first.")
latest_season = int(teams_tbl["season"].max())
rat = (teams_tbl[teams_tbl["season"] == latest_season]
       [["Team","elo","glicko","glicko_rd","glicko_vol"]]
       .drop_duplicates("Team"))
rat_map = rat.set_index("Team").to_dict(orient="index")

# ---------- Rest days ----------
tg_dates = pd.read_sql('SELECT "Team","Date" FROM team_games', engine)
tg_dates["Date"] = pd.to_datetime(tg_dates["Date"], errors="coerce")
last_played = (tg_dates.dropna(subset=["Date"])
                      .sort_values(["Team","Date"])
                      .groupby("Team", as_index=False)
                      .tail(1)
                      .set_index("Team")["Date"]
                      .to_dict())

# ---------- Team precompute latest snapshots ----------
try:
    tp_latest = pd.read_sql('SELECT * FROM team_precompute_latest_current', engine)
except Exception:
    base = pd.read_sql('SELECT * FROM team_precompute', engine)
    base["Date"] = pd.to_datetime(base["Date"], errors="coerce")
    base = base.dropna(subset=["Date"]).sort_values(["Team","Date"])
    tp_latest = base.groupby("Team", as_index=False).tail(1)

want_cols = [
    "Team",
    "score_avg_last_5","inside50_avg_last_5","turnovers_avg_last_5","free_kicks_avg_last_5",
    "disposals_avg_last_5","clearances_avg_last_5","tackles_avg_last_5","marks_avg_last_5",
    "concede_disposals_avg_last_5","concede_tackles_avg_last_5","concede_marks_avg_last_5","concede_clearances_avg_last_5",
]
have_cols = ["Team"] + [c for c in want_cols if c in tp_latest.columns]
tp = tp_latest[have_cols].copy()
tp_map = tp.set_index("Team").to_dict(orient="index")

# ---------- Assemble features (A = Team, B = Opponent) ----------
rows = []
for r in ng_df.itertuples(index=False):
    A, B = r.Team, r.Opponent
    date, venue, ts = r.Date, r.Venue, r.Timeslot

    RA = rat_map.get(A, {"elo":1500.0,"glicko":1500.0,"glicko_rd":350.0,"glicko_vol":0.12})
    RB = rat_map.get(B, {"elo":1500.0,"glicko":1500.0,"glicko_rd":350.0,"glicko_vol":0.12})

    eloA, eloB = safe_num(RA["elo"], 1500.0), safe_num(RB["elo"], 1500.0)
    gA, gB     = safe_num(RA["glicko"], 1500.0), safe_num(RB["glicko"], 1500.0)
    rdA, rdB   = safe_num(RA["glicko_rd"], 350.0), safe_num(RB["glicko_rd"], 350.0)
    vA, vB     = safe_num(RA["glicko_vol"], 0.12), safe_num(RB["glicko_vol"], 0.12)

    rec = {"Team": A, "Opponent": B, "Date": date, "Venue": venue, "Timeslot": ts}
    rec["elo_diff"]    = eloA - eloB
    rec["glicko_diff"] = gA - gB
    rec["rd_diff"]     = rdA - rdB
    rec["vol_diff"]    = vA - vB

    hA, hB = is_home(A, venue), is_home(B, venue)
    rec["home_edge"] = hA - hB

    rec.update(timeslot_dummies(ts))
    rec["elo_exp"] = 1.0 / (1.0 + 10.0 ** (-(eloA - eloB) / 400.0))

    A_inter, B_inter = is_interstate(A, venue), is_interstate(B, venue)
    rec["interstate_edge"] = A_inter - B_inter
    A_sec, B_sec = is_secondary_home(A, venue), is_secondary_home(B, venue)
    rec["secondary_home_edge"] = A_sec - B_sec

    A_tp = tp_map.get(A, {})
    B_tp = tp_map.get(B, {})
    def getv(d, k): return safe_num(d.get(k, np.nan), np.nan)

    for name in [
        "score_avg_last_5","inside50_avg_last_5","turnovers_avg_last_5","free_kicks_avg_last_5",
        "disposals_avg_last_5","clearances_avg_last_5","tackles_avg_last_5","marks_avg_last_5"
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

# Ensure every expected feature exists
for c in feature_cols:
    if c not in F.columns:
        F[c] = 0.0
for c in feature_cols:
    F[c] = pd.to_numeric(F[c], errors="coerce").fillna(0.0)

# Predict raw probs and normalize by fixture + add draw bucket
P_raw = clf.predict_proba(F[feature_cols].astype(float))[:, 1]
F["Win_Probability_raw"] = np.clip(P_raw, 0.0, 1.0)

F["pair_key"] = (
    pd.to_datetime(F["Date"]).dt.strftime("%Y-%m-%d") + "|" +
    F[["Team","Opponent"]].apply(lambda x: "|".join(sorted(x)), axis=1)
)

def adjust_pair(g: pd.DataFrame) -> pd.DataFrame:
    g = g.copy()
    if len(g) != 2:
        g["Win_Probability"] = 0.0
        g["Draw_Probability"] = 0.0
        return g

    p_draw = float(DRAW_BASE_RATE)

    pa, pb = float(g.iloc[0]["Win_Probability_raw"]), float(g.iloc[1]["Win_Probability_raw"])
    s = max(pa + pb, 1e-12)
    pa_norm, pb_norm = pa / s, pb / s

    if SCALE_DRAW_BY_CLOSENESS:
        closeness = 2.0 * min(pa_norm, pb_norm)  # 0..1
        p_draw = float(DRAW_BASE_RATE) * float(closeness)

    g.loc[g.index[0], "Win_Probability"] = (1.0 - p_draw) * pa_norm * 100.0
    g.loc[g.index[1], "Win_Probability"] = (1.0 - p_draw) * pb_norm * 100.0
    g["Draw_Probability"] = p_draw * 100.0
    return g

F = F.groupby("pair_key", group_keys=False).apply(adjust_pair)

F["Win_Probability"] = pd.to_numeric(F["Win_Probability"], errors="coerce").fillna(0.0)
F["Draw_Probability"] = pd.to_numeric(F["Draw_Probability"], errors="coerce").fillna(0.0)


# ---------- Persist to team_precompute (latest row per team) ----------
with engine.begin() as con:
    con.exec_driver_sql('ALTER TABLE team_precompute ADD COLUMN IF NOT EXISTS "Next_Opponent" TEXT')
    con.exec_driver_sql('ALTER TABLE team_precompute ADD COLUMN IF NOT EXISTS "Next_Venue" TEXT')
    con.exec_driver_sql('ALTER TABLE team_precompute ADD COLUMN IF NOT EXISTS "Next_Timeslot" TEXT')
    con.exec_driver_sql('ALTER TABLE team_precompute ADD COLUMN IF NOT EXISTS "Next_Date" DATE')
    con.exec_driver_sql('ALTER TABLE team_precompute ADD COLUMN IF NOT EXISTS "Win_Probability" DOUBLE PRECISION')
    con.exec_driver_sql('ALTER TABLE team_precompute ADD COLUMN IF NOT EXISTS "Draw_Probability" DOUBLE PRECISION')

tp_dates = pd.read_sql('SELECT "Team", MAX("Date") AS max_date FROM team_precompute GROUP BY "Team"', engine)
latest_map = dict(zip(tp_dates["Team"], pd.to_datetime(tp_dates["max_date"], errors="coerce")))

updated = 0
with engine.begin() as con:
    for r in F.itertuples(index=False):
        team   = r.Team
        opp    = r.Opponent
        venue  = r.Venue
        winp   = float(r.Win_Probability)
        drawp  = float(r.Draw_Probability)
        timeslot = r.Timeslot
        next_date = pd.to_datetime(r.Date).date() if pd.notna(r.Date) else None


        latest_date = latest_map.get(team)
        if pd.isna(latest_date):
            continue

        params = {"team": team, "date": latest_date, "opp": opp, "venue": venue, "p": winp, "pd": drawp, "timeslot": timeslot, "next_date": next_date}
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

print(f"✅ Wrote next-game Next_Venue + Win_Probability + Draw_Probability for {updated} team rows in team_precompute.")

# ---------- Refresh latest views (auto-includes new cols) ----------
def refresh_team_precompute_views(dest="team_precompute"):
    drop_overall = f'DROP VIEW IF EXISTS {dest}_latest CASCADE;'
    drop_current = f'DROP VIEW IF EXISTS {dest}_latest_current CASCADE;'

    with engine.begin() as con:
        cols = [r[0] for r in con.execute(sqtext("""
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name = :t
            ORDER BY ordinal_position
        """), {"t": dest}).fetchall()]
        col_list_tp = ", ".join(f'tp."{c}"' for c in cols)
        col_list_plain = ", ".join(f'"{c}"' for c in cols)

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
