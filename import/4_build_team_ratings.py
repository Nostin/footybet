# 4_build_team_ratings.py
# Builds team ratings and season summary from player-level table.
# - One row per match -> Elo + Glicko2 updates (robust bisection; bounded iters)
# - Home-ground advantage via HOME_GROUNDS mapping (no aliasing)
# - Elo start-of-season regression, Glicko RD inflation + optional between-game RD drift
# - Logs Glicko expected winprob and volatility (sigma), enabling a "season_surprise" metric
# - Finals (EF/QF/SF/PF/GF) are EXCLUDED from ladder points, percentage, and ladder position
# - Outputs:
#     * team_games   (per-game audit with expectations & post-ratings)
#     * --dest table (season summary: one row per team)

from pathlib import Path
import sys
import math, time, argparse
import pandas as pd
import numpy as np
from sqlalchemy import text

ROOT = Path(__file__).resolve().parents[1]   # /.../root
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from db_connect import get_engine
from util import HOME_GROUNDS, SECONDARY_HOME_GROUNDS

# ---------- CLI ----------
ap = argparse.ArgumentParser()
ap.add_argument("--source", default="player_stats", help="Input table with player-level rows")
ap.add_argument("--dest", default="teams", help="Output table for season summary (one row per team)")

# Elo knobs
ap.add_argument("--k-elo", type=float, default=24.0, help="Elo K-factor")
ap.add_argument("--home-adv", type=float, default=0.0, help="Home-ground advantage in Elo points")
ap.add_argument("--elo-regress", type=float, default=0.0, help="Start-of-season Elo regression weight towards 1500 (0..1)")

# Glicko knobs
ap.add_argument("--skip-glicko", action="store_true", help="Skip Glicko-2 (Elo only)")
ap.add_argument("--glicko-tau", type=float, default=0.9, help="Glicko-2 volatility constraint tau (higher = more responsive sigma)")
ap.add_argument("--glicko-tol", type=float, default=1e-5, help="Glicko-2 solve tolerance (bisection)")
ap.add_argument("--g2-init-rd", type=float, default=350.0, help="Initial Glicko RD")
ap.add_argument("--g2-init-vol", type=float, default=0.12, help="Initial Glicko volatility (sigma)")
ap.add_argument("--g2-rd-inflate", type=float, default=0.0, help="Start-of-season RD inflation, e.g. 0.30 => RD *= 1.30 (clamped [30,350])")
ap.add_argument("--g2-rd-decay-per-day", type=float, default=0.6,
                help="Between-game RD drift (per day) added in quadrature; e.g. 0.6 ~ 4 RD per idle week")

# IO / filters
ap.add_argument("--only-after", default=None, help="YYYY-MM-DD filter to only process games after this date")
ap.add_argument("--chunksize", type=int, default=20000, help="DB write chunk size")

# number of games to compute surprise over
ap.add_argument("--surprise-window", type=int, default=22,
                help="Number of most recent REGULAR-SEASON games to compute surprise over (default 22)")
ap.add_argument("--surprise-min-games", type=int, default=8,
                help="Minimum games required to report surprise; otherwise NaN (default 8)")
args = ap.parse_args()

# Progress bar
try:
    from tqdm import tqdm
    def pbar(it, **kw): return tqdm(it, **kw)
except Exception:
    def pbar(it, **kw): return it

engine = get_engine()

# ---------- Ratings constants ----------
INIT_ELO = 1500.0
G2_INIT_R  = 1500.0
G2_INIT_RD = float(args.g2_init_rd)
G2_INIT_VOL= float(args.g2_init_vol)
G2_Q = math.log(10)/400.0
G2_SCALE = 173.7178

def is_home(team: str, venue: str) -> bool:
    if team is None or venue is None:
        return False
    team_key = str(team).strip()
    venue_key = str(venue).strip()
    return venue_key in HOME_GROUNDS.get(team_key, [])


# ---------- Helpers ----------
def clean_ground_list(xs):
    if xs is None:
        return []
    out = []
    for v in xs:
        if v is None:
            continue
        s = str(v).strip()
        if s:
            out.append(s)
    return out

SECONDARY_HOME_GROUNDS = {k: clean_ground_list(v) for k, v in SECONDARY_HOME_GROUNDS.items()}
HOME_GROUNDS = {k: clean_ground_list(v) for k, v in HOME_GROUNDS.items()}

def g_glicko(phi):
    return 1.0 / math.sqrt(1.0 + 3.0*(G2_Q**2)*(phi**2)/(math.pi**2))

def E_glicko(mu, mu_j, phi_j):
    return 1.0 / (1.0 + math.exp(-g_glicko(phi_j) * (mu - mu_j)))

def glicko2_one_match(r, RD, vol, s, r_op, RD_op, tau=0.5, tol=1e-5, max_iter=60):
    """Robust single-opponent Glicko-2 update (pure bisection, bounded iterations). Returns (r', RD', vol', expected_prob)."""
    mu,  phi  = (r - 1500.0) / G2_SCALE, RD / G2_SCALE
    muj, phij = (r_op - 1500.0) / G2_SCALE, RD_op / G2_SCALE

    phij = max(phij, 1e-12)
    phi  = max(phi,  1e-12)

    e  = E_glicko(mu, muj, phij)
    gj = g_glicko(phij)
    v  = 1.0 / ((gj**2) * e * (1.0 - e))
    delta = v * gj * (s - e)

    if not np.isfinite(v) or not np.isfinite(delta):
        return r, RD, vol, e

    a = math.log(vol**2)

    def f(x):
        ex = math.exp(x)
        num = ex * (delta**2 - phi**2 - v - ex)
        den = 2.0 * (phi**2 + v + ex)**2
        return (num / den) - ((x - a) / (tau**2))

    # bracket
    A = a
    if delta**2 > (phi**2 + v):
        B = math.log(delta**2 - phi**2 - v)
    else:
        k, B = 1, a - 1.0*abs(tau)
        while f(B) > 0 and k < 50:
            k += 1
            B = a - k*abs(tau)

    fa, fb = f(A), f(B)
    if not np.isfinite(fa) or not np.isfinite(fb) or fa * fb > 0:
        # fallback: keep sigma, minimal movement
        phi_star = math.sqrt(phi**2 + vol**2)
        r_new  = 1500.0 + G2_SCALE * mu
        RD_new = G2_SCALE * phi_star
        return r_new, RD_new, vol, e

    # bisection
    for _ in range(max_iter):
        C = 0.5 * (A + B)
        fc = f(C)
        if not np.isfinite(fc):
            break
        if abs(B - A) <= tol:
            A = C
            break
        if fa * fc <= 0:
            B, fb = C, fc
        else:
            A, fa = C, fc

    A_star = 0.5 * (A + B)
    sigma_prime = math.exp(A_star / 2.0)

    phi_star = math.sqrt(phi**2 + sigma_prime**2)
    phi_new  = 1.0 / math.sqrt((1.0 / (phi_star**2)) + (1.0 / v))
    mu_new   = mu + (phi_new**2) * gj * (s - e)

    r_new  = 1500.0 + G2_SCALE * mu_new
    RD_new = G2_SCALE * phi_new
    return r_new, RD_new, sigma_prime, e

def inflate_rd_for_gap(RD, days_gap, per_day):
    """Glicko-1 style drift: RD' = sqrt(RD^2 + (per_day * days_gap)^2), clamped to [30,350]."""
    if per_day <= 0 or days_gap <= 0:
        return RD
    RDp = math.sqrt(RD*RD + (per_day * days_gap)**2)
    return min(350.0, max(30.0, RDp))

def coalesce_dates(df) -> pd.Series:
    """Return a single datetime series from any of ['Date','date','match_date']."""
    candidates = [c for c in df.columns if c.lower() in ("date", "match_date")]
    if "Date" not in df.columns and candidates:
        s = None
        for c in candidates:
            tmp = pd.to_datetime(df[c], errors="coerce")
            s = tmp if s is None else s.fillna(tmp)
        return s
    return pd.to_datetime(df["Date"], errors="coerce")

# ---------- Load & collapse to one row per team–game ----------
t0 = time.perf_counter()
print("▶︎ Stage 1/5: reading ONE ROW PER TEAM–GAME from DB...")

team_game_sql_pg = f"""
SELECT DISTINCT ON ("Date","Team","Opponent")
  "Date","Team","Opponent","Round","Venue","Timeslot",
  "Team Score" AS team_score
FROM {args.source}
ORDER BY "Date","Team","Opponent","Player"
"""
team_game_sql_win = f"""
SELECT
  "Date","Team","Opponent","Round","Venue","Timeslot","Team Score" AS team_score
FROM (
  SELECT *,
    ROW_NUMBER() OVER (PARTITION BY "Date","Team","Opponent" ORDER BY "Player") AS rn
  FROM {args.source}
) t
WHERE rn = 1
"""
params = {}
if args.only_after:
    team_game_sql_pg  = team_game_sql_pg.replace(
        "ORDER BY", 'WHERE "Date" > :after\nORDER BY'
    )
    team_game_sql_win = f"""
    SELECT
      "Date","Team","Opponent","Round","Venue","Timeslot","Team Score" AS team_score
    FROM (
      SELECT *,
        ROW_NUMBER() OVER (PARTITION BY "Date","Team","Opponent" ORDER BY "Player") AS rn
      FROM {args.source}
      WHERE "Date" > :after
    ) t
    WHERE rn = 1
    """
    params = {"after": args.only_after}

# try PG path, else window path, else portable pandas
try:
    tg = pd.read_sql(text(team_game_sql_pg), engine, params=params)
except Exception:
    try:
        tg = pd.read_sql(text(team_game_sql_win), engine, params=params)
    except Exception:
        cols = '"Date","Team","Opponent","Round","Venue","Timeslot","Team Score" AS team_score'
        where_clause = 'WHERE "Date" > :after' if args.only_after else ''
        q = text(f'SELECT {cols} FROM {args.source} {where_clause}')
        raw = pd.read_sql(q, engine, params=params)
        tg = (raw.sort_values(["Date","Team","Opponent","Player"])
                .drop_duplicates(subset=["Date","Team","Opponent"])
                .drop(columns=["Player"], errors="ignore"))

# date clean
tg["Date"] = coalesce_dates(tg)
nat_count = tg["Date"].isna().sum()
if nat_count:
    print(f"   ⚠️ dropping {nat_count} rows with invalid dates")
tg = tg.dropna(subset=["Date"]).copy()

# strip only (NO aliasing)
for col in ("Team","Opponent","Venue"):
    tg[col] = tg[col].astype(str).str.strip()

tg["team_score"] = pd.to_numeric(tg["team_score"], errors="coerce")
tg = tg.dropna(subset=["team_score"])

print(f"   ✓ team-game rows: {len(tg):,} (took {time.perf_counter()-t0:.2f}s)")

# ---------- Pair to one row per match ----------
t1 = time.perf_counter()
print("▶︎ Stage 2/5: pairing team-games into matches...")

tg["_pair_key"] = (
    tg["Date"].dt.strftime("%Y-%m-%d") + "|" +
    tg[["Team","Opponent"]].apply(lambda x: "|".join(sorted(x)), axis=1)
)

counts = tg["_pair_key"].value_counts()
bad_keys = counts[counts != 2]
if len(bad_keys):
    print(f"   ⚠️ {len(bad_keys)} malformed matches (not 2 rows) — they will be skipped.")
    dbg = tg[tg["_pair_key"].isin(bad_keys.index)].sort_values(["Date","Team","Opponent"])
    print(dbg[["Date","Team","Opponent","Round","Venue","Timeslot","team_score"]].head(12).to_string(index=False))

pairs = []
for _, grp in pbar(tg.groupby("_pair_key"), total=counts.size, desc="Pairing"):
    if len(grp) != 2:
        continue
    g = grp.sort_values(by=["Team","Opponent"]).reset_index(drop=True)
    a, b = g.iloc[0], g.iloc[1]
    pairs.append({
        "Date": a["Date"], "Round": a["Round"], "Venue": a["Venue"], "Timeslot": a["Timeslot"],
        "team_a": a["Team"], "team_b": b["Team"],
        "score_a": float(a["team_score"]), "score_b": float(b["team_score"])
    })

matches = pd.DataFrame(pairs).sort_values(["Date","team_a","team_b"]).reset_index(drop=True)
matches["season"] = matches["Date"].dt.year.astype(int)
print(f"   ✓ matches: {len(matches):,} (took {time.perf_counter()-t1:.2f}s)")

# ---------- Build per-team rows + tallies ----------
t2 = time.perf_counter()
print("▶︎ Stage 3/5: computing percentage & season tallies...")

a = matches.rename(columns={"team_a":"Team","team_b":"Opponent","score_a":"points_for","score_b":"points_against"})
b = matches.rename(columns={"team_b":"Team","team_a":"Opponent","score_b":"points_for","score_a":"points_against"})
teams_rows = pd.concat([a,b], ignore_index=True)

teams_rows["percentage"] = np.where(teams_rows["points_against"]>0,
                                    100.0*teams_rows["points_for"]/teams_rows["points_against"], np.nan)
teams_rows["result"] = np.where(teams_rows["points_for"]>teams_rows["points_against"], "Win",
                                np.where(teams_rows["points_for"]<teams_rows["points_against"], "Loss", "Draw"))
res_map = {"Win":1.0, "Loss":0.0, "Draw":0.5}
teams_rows["score_S"] = teams_rows["result"].map(res_map).astype(float)

teams_rows = teams_rows.sort_values(["season","Team","Date"]).reset_index(drop=True)
grp = teams_rows.groupby(["season","Team"], group_keys=False)
for flag, lab in [("is_win","Win"), ("is_loss","Loss"), ("is_draw","Draw")]:
    teams_rows[flag] = (teams_rows["result"]==lab).astype(int)
teams_rows["wins_cum_before"] = grp["is_win" ].transform(lambda s: s.shift(1, fill_value=0).cumsum())
teams_rows["loss_cum_before"] = grp["is_loss"].transform(lambda s: s.shift(1, fill_value=0).cumsum())
teams_rows["draw_cum_before"] = grp["is_draw"].transform(lambda s: s.shift(1, fill_value=0).cumsum())
teams_rows["wins_cum_after"]  = teams_rows["wins_cum_before"] + teams_rows["is_win"]
teams_rows["loss_cum_after"]  = teams_rows["loss_cum_before"] + teams_rows["is_loss"]
teams_rows["draw_cum_after"]  = teams_rows["draw_cum_before"] + teams_rows["is_draw"]

lookup = teams_rows.set_index(["Date","Team","Opponent"])[[
    "percentage","wins_cum_before","loss_cum_before","draw_cum_before",
    "wins_cum_after","loss_cum_after","draw_cum_after"
]]
print(f"   ✓ tallies built in {time.perf_counter()-t2:.2f}s")

# ---------- Iterate matches -> Elo + Glicko2 ----------
print("▶︎ Stage 4/5: updating Elo + Glicko2...")

elo = {}              # team -> rating
g2  = {}              # team -> (r, RD, vol)
last_season_seen = {} # team -> season
last_played = {}      # team -> last match date (for RD drift)

rows = []

for row in pbar(matches.itertuples(index=False), total=len(matches), desc="Matches"):
    date   = row.Date
    season = int(row.season)
    teamA, teamB = row.team_a, row.team_b
    scA, scB     = float(row.score_a), float(row.score_b)

    SA = 1.0 if scA>scB else 0.0 if scA<scB else 0.5
    SB = 1.0 - SA if SA in (0.0,1.0) else 0.5

    RA = elo.get(teamA, INIT_ELO); RB = elo.get(teamB, INIT_ELO)
    rA, rdA, volA = g2.get(teamA, (G2_INIT_R, G2_INIT_RD, G2_INIT_VOL))
    rB, rdB, volB = g2.get(teamB, (G2_INIT_R, G2_INIT_RD, G2_INIT_VOL))

    # start-of-season adjustments (Elo regression + RD inflation)
    if last_season_seen.get(teamA) not in (None, season):
        if args.elo_regress > 0:
            RA = (1.0 - args.elo_regress)*RA + args.elo_regress*INIT_ELO
        if args.g2_rd_inflate > 0:
            rdA = min(350.0, max(30.0, rdA * (1.0 + args.g2_rd_inflate)))
    if last_season_seen.get(teamB) not in (None, season):
        if args.elo_regress > 0:
            RB = (1.0 - args.elo_regress)*RB + args.elo_regress*INIT_ELO
        if args.g2_rd_inflate > 0:
            rdB = min(350.0, max(30.0, rdB * (1.0 + args.g2_rd_inflate)))

    # between-game RD drift (days since last match)
    gapA = (date - last_played.get(teamA, date)).days if teamA in last_played else 0
    gapB = (date - last_played.get(teamB, date)).days if teamB in last_played else 0
    rdA = inflate_rd_for_gap(rdA, gapA, args.g2_rd_decay_per_day)
    rdB = inflate_rd_for_gap(rdB, gapB, args.g2_rd_decay_per_day)

    # home advantage via HOME_GROUNDS mapping
    H_A = args.home_adv if is_home(teamA, row.Venue) else 0.0
    H_B = args.home_adv if is_home(teamB, row.Venue) else 0.0

    # Elo expectation and update
    EA = 1.0 / (1.0 + 10.0 ** (-((RA + H_A) - (RB + H_B)) / 400.0))
    EB = 1.0 - EA
    RA_post = RA + args.k_elo * (SA - EA)
    RB_post = RB + args.k_elo * (SB - EB)

    # Glicko-2 (also capture expectations EA_g/EB_g)
    if not args.skip_glicko:
        rA_post, rdA_post, volA_post, EA_g = glicko2_one_match(
            rA, rdA, volA, SA, rB, rdB, tau=args.glicko_tau, tol=args.glicko_tol
        )
        rB_post, rdB_post, volB_post, EB_g = glicko2_one_match(
            rB, rdB, volB, SB, rA, rdA, tau=args.glicko_tau, tol=args.glicko_tol
        )
    else:
        rA_post, rdA_post, volA_post, EA_g = rA, rdA, volA, EA
        rB_post, rdB_post, volB_post, EB_g = rB, rdB, volB, EB

    # tallies for this match
    A_info = lookup.loc[(date, teamA, teamB)]
    B_info = lookup.loc[(date, teamB, teamA)]

    def pack(team, opp, pf, pa, info, pre, post, exp, r_pre, rd_pre, vol_pre, r_po, rd_po, vol_po, g2exp):
        return {
            "Date": date, "season": season,
            "Round": getattr(row, "Round", None),
            "Team": team, "Opponent": opp, "Venue": row.Venue, "Timeslot": row.Timeslot,
            "points_for": pf, "points_against": pa, "percentage": float(info["percentage"]),
            "result": "Win" if pf>pa else ("Loss" if pf<pa else "Draw"),
            "wins_cum_before": int(info["wins_cum_before"]),
            "loss_cum_before": int(info["loss_cum_before"]),
            "draw_cum_before": int(info["draw_cum_before"]),
            "wins_cum_after": int(info["wins_cum_after"]),
            "loss_cum_after": int(info["loss_cum_after"]),
            "draw_cum_after": int(info["draw_cum_after"]),
            "elo_pre": pre, "elo_exp_winprob": exp, "elo_post": post,
            "g2_rating_pre": r_pre, "g2_rd_pre": rd_pre, "g2_vol_pre": vol_pre,
            "g2_rating_post": r_po, "g2_rd_post": rd_po, "g2_vol_post": vol_po,
            "g2_exp_winprob": g2exp,
        }

    rows.append(pack(teamA, teamB, scA, scB, A_info, RA, RA_post, EA, rA, rdA, volA, rA_post, rdA_post, volA_post, EA_g))
    rows.append(pack(teamB, teamA, scB, scA, B_info, RB, RB_post, EB, rB, rdB, volB, rB_post, rdB_post, volB_post, EB_g))

    # commit states
    elo[teamA], elo[teamB] = RA_post, RB_post
    g2[teamA] = (rA_post, rdA_post, volA_post)
    g2[teamB] = (rB_post, rdB_post, volB_post)
    last_season_seen[teamA] = season
    last_season_seen[teamB] = season
    last_played[teamA] = date
    last_played[teamB] = date

team_table = pd.DataFrame(rows).sort_values(["Date","Team"]).reset_index(drop=True)
print(f"   ✓ ratings computed for {len(matches):,} matches")

# ---------- Write per-game audit + season summary (finals excluded for ladder) ----------
t3 = time.perf_counter()
print("▶︎ Stage 5/5: writing tables...")

# keep per-game ratings for auditing
team_table.to_sql("team_games", engine, if_exists="replace", index=False,
                  method="multi", chunksize=args.chunksize)

# Finals filter for ladder stats
FINALS = {"EF", "QF", "SF", "PF", "GF"}
def is_finals_round(x):
    if x is None:
        return False
    s = str(x).strip().upper()
    return s in FINALS

team_table["is_finals"] = team_table["Round"].apply(is_finals_round)

# Regular-season slice for ladder metrics & surprise
reg = team_table[~team_table["is_finals"]].copy()
res_map = {"Win":1.0, "Loss":0.0, "Draw":0.5}
reg["S"] = reg["result"].map(res_map).astype(float)
# Surprise = |actual - expected by Glicko|
reg["g2_surprise"] = (reg["S"] - reg["g2_exp_winprob"]).abs()
reg["is_home"] = reg.apply(lambda r: is_home(r["Team"], r["Venue"]), axis=1)
reg["home_win"]  = ((reg["result"]=="Win")  & (reg["is_home"]==True)).astype(int)
reg["home_loss"] = ((reg["result"]=="Loss") & (reg["is_home"]==True)).astype(int)
reg["away_win"]  = ((reg["result"]=="Win")  & (reg["is_home"]==False)).astype(int)
reg["away_loss"] = ((reg["result"]=="Loss") & (reg["is_home"]==False)).astype(int)
# season totals (regular season)
# season totals (regular season) — still season-scoped
season_totals = (
    reg.groupby(["season","Team"], as_index=False)
       .agg(
            season_wins   = ("result", lambda s: (s=="Win").sum()),
            season_losses = ("result", lambda s: (s=="Loss").sum()),
            season_draws  = ("result", lambda s: (s=="Draw").sum()),
            pf            = ("points_for", "sum"),
            pa            = ("points_against", "sum"),
            season_home_wins=("home_win","sum"),
            season_home_losses=("home_loss","sum"),
            season_away_wins=("away_win","sum"),
            season_away_losses=("away_loss","sum"),
       )
)
season_totals["season_percentage"] = 100.0 * season_totals["pf"] / season_totals["pa"].replace(0, np.nan)
season_totals = season_totals.rename(columns={"pf":"season_points_for", "pa":"season_points_against"})

# --- NEW: rolling "surprise" over last N regular-season games in that season ---
N = int(args.surprise_window)
MIN_GAMES = int(args.surprise_min_games)

reg_sorted = reg.sort_values(["season", "Team", "Date"]).copy()

# take last N games per (season,Team)
lastN = (
    reg_sorted.groupby(["season", "Team"], as_index=False, group_keys=False)
              .tail(N)
)

surprise_agg = (
    lastN.groupby(["season","Team"], as_index=False)
         .agg(
             season_surprise=("g2_surprise", "mean"),
             season_surprise_games=("g2_surprise", "size"),
         )
)

# If fewer than MIN_GAMES, blank it out (early-season noise control)
surprise_agg.loc[surprise_agg["season_surprise_games"] < MIN_GAMES, "season_surprise"] = np.nan

# last post-ratings (may include finals; that's okay for strength going forward)
last_posts = (team_table
              .sort_values(["season","Team","Date"])
              .groupby(["season","Team"], as_index=False)
              .tail(1)
              [["season","Team","elo_post","g2_rating_post","g2_rd_post","g2_vol_post"]]
              .rename(columns={
                  "elo_post":"elo",
                  "g2_rating_post":"glicko",
                  "g2_rd_post":"glicko_rd",
                  "g2_vol_post":"glicko_vol",
              }))

summary = season_totals.merge(last_posts, on=["season","Team"], how="left")
summary = summary.merge(surprise_agg, on=["season","Team"], how="left")

# ladder points/position (AFL: 4 for win, 2 for draw) — regular season only
summary["ladder_points"] = 4*summary["season_wins"] + 2*summary["season_draws"]
summary = summary.sort_values(
    ["season", "ladder_points", "season_percentage", "season_points_for"],
    ascending=[True, False, False, False]
)
summary["ladder_position"] = summary.groupby("season").cumcount() + 1

# --- Elo/Glicko ranks within each season (1 = best) ---
# Use dense ranking so ties share the same rank without gaps: 1,2,2,3...
summary["elo_rank"] = (
    summary.groupby("season")["elo"]
           .rank(method="dense", ascending=False)
           .astype(int)
)

summary["glicko_rank"] = (
    summary.groupby("season")["glicko"]
           .rank(method="dense", ascending=False)
           .astype(int)
)

summary["percentage_rank"] = (
    summary.groupby("season")["season_percentage"]
           .rank(method="dense", ascending=False)
           .astype(int)
)

# --- NEW: store home grounds ---
def join_grounds(d, team):
    xs = d.get(team, [])
    xs = [str(x).strip() for x in xs if x is not None and str(x).strip()]
    return ", ".join(xs)

summary["primary_home_grounds"] = summary["Team"].apply(lambda t: join_grounds(HOME_GROUNDS, t))
summary["secondary_home_grounds"] = summary["Team"].apply(lambda t: join_grounds(SECONDARY_HOME_GROUNDS, t))


summary = summary[[
    "season", "Team",
    "primary_home_grounds", "secondary_home_grounds",
    "elo", "elo_rank", "glicko", "glicko_rank", "glicko_rd", "glicko_vol",
    "season_wins", "season_losses", "season_draws",
    "season_points_for","season_points_against",
    "season_home_wins", "season_home_losses", "season_away_wins", "season_away_losses",
    "season_percentage", "percentage_rank", "ladder_points", "ladder_position",
    "season_surprise", "season_surprise_games"
]]

ddl_latest_overall_drop = f'DROP VIEW IF EXISTS team_precompute_latest CASCADE;'
with engine.begin() as con:
    con.execute(text(ddl_latest_overall_drop))

summary.to_sql(args.dest, engine, if_exists="replace", index=False,
               method="multi", chunksize=args.chunksize)

print(f"✅ Wrote per-game table 'team_games' ({len(team_table):,} rows) "
      f"and season summary '{args.dest}' ({len(summary):,} teams) in {time.perf_counter()-t3:.2f}s.")
print("✅ Done.")
