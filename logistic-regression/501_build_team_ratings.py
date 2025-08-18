# 501_build_team_ratings_fastest.py
# Builds team ratings and season summary from player-level table.
# - One row per match -> Elo + Glicko2 updates
# - Home-ground advantage, Elo start-of-season regression, Glicko RD inflation
# - Robust Glicko2 (pure bisection; bounded iterations)
# - Date coalescing: handles Date/date/match_date columns gracefully
# - Outputs:
#     * team_games   (per-game audit)
#     * --dest table (season summary: one row per team)

import math, time, argparse
import numpy as np
import pandas as pd
from sqlalchemy import text
from db_connect import get_engine

# ---------- CLI ----------
ap = argparse.ArgumentParser()
ap.add_argument("--source", default="player_stats", help="Input table with player-level rows")
ap.add_argument("--dest", default="teams", help="Output table for season summary (one row per team)")
ap.add_argument("--k-elo", type=float, default=24.0, help="Elo K-factor")
ap.add_argument("--home-adv", type=float, default=0.0, help="Home-ground advantage in Elo points (applied when venue/team mapping says home)")
ap.add_argument("--elo-regress", type=float, default=0.0, help="Start-of-season Elo regression weight to mean 1500.0 (0..1)")
ap.add_argument("--skip-glicko", action="store_true", help="Skip Glicko-2 (Elo only)")
ap.add_argument("--glicko-tau", type=float, default=0.5, help="Glicko-2 volatility constraint tau")
ap.add_argument("--glicko-tol", type=float, default=1e-5, help="Glicko-2 root-solve tolerance (bisection)")
ap.add_argument("--g2-rd-inflate", type=float, default=0.0, help="Start-of-season RD inflation factor, e.g. 0.30 -> RD *= 1.30 (clamped to [30,350])")
ap.add_argument("--only-after", default=None, help="YYYY-MM-DD filter to only process games after this date")
ap.add_argument("--chunksize", type=int, default=20000, help="DB write chunk size")
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
G2_INIT_R, G2_INIT_RD, G2_INIT_VOL = 1500.0, 350.0, 0.06
G2_Q = math.log(10)/400.0
G2_SCALE = 173.7178

# ---------- Optional alias normalization ----------
ALIASES = {
    # Nicknames -> canonical
    "Dees": "Melbourne",
    "GWS": "GWS Giants",
    "Bulldogs": "Western Bulldogs",
    "Eagles": "West Coast",
    "Dockers": "Fremantle",
    "Saints": "St Kilda",
    "Suns": "Gold Coast",
    "Port": "Port Adelaide",
    "North": "North Melbourne",
    "Crows": "Adelaide",
    "Swans": "Sydney",
    "Lions": "Brisbane",
}

# ---------- Optional venue -> home team map (fill in what you know) ----------
# If a venue has multiple home tenants (MCG, Marvel, Adelaide), we leave it blank and no HGA is applied.
VENUE_HOME = {
    "SCG": "Sydney",
    "Gabba": "Brisbane",
    "Brisbane": "Brisbane",
    "Geelong": "Geelong",     # GMHBA
    "Gold Coast": "Gold Coast",
    "Perth": None,            # Optus: West Coast & Fremantle -> ambiguous
    "Adelaide": None,         # Adelaide Oval: Adelaide & Port -> ambiguous
    "MCG": None,              # shared
    "Marvel": None,           # shared
    "Engie": "GWS Giants",    # Sydney Showground (name may vary in your data)
    # add others as needed
}

def is_home(team: str, venue: str) -> bool:
    if not venue:
        return False
    home_team = VENUE_HOME.get(str(venue), None)
    return (home_team is not None) and (team == home_team)

# ---------- Helpers ----------
def g_glicko(phi):
    return 1.0 / math.sqrt(1.0 + 3.0*(G2_Q**2)*(phi**2)/(math.pi**2))

def E_glicko(mu, mu_j, phi_j):
    return 1.0 / (1.0 + math.exp(-g_glicko(phi_j) * (mu - mu_j)))

def glicko2_one_match(r, RD, vol, s, r_op, RD_op, tau=0.5, tol=1e-5, max_iter=60):
    """Robust single-opponent Glicko-2 update (pure bisection, bounded iterations)."""
    mu,  phi  = (r - 1500.0) / G2_SCALE, RD / G2_SCALE
    muj, phij = (r_op - 1500.0) / G2_SCALE, RD_op / G2_SCALE

    # guard-rails
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

    # bracket a root
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
        # fallback: keep volatility; still update rating deviation & rating lightly
        phi_star = math.sqrt(phi**2 + vol**2)
        phi_new  = phi_star
        mu_new   = mu + (phi_new**2) * gj * (s - e) * 0.0
        r_new  = 1500.0 + G2_SCALE * mu_new
        RD_new = G2_SCALE * phi_new
        return r_new, RD_new, vol, e

    # pure bisection
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

    # update RD and rating
    phi_star = math.sqrt(phi**2 + sigma_prime**2)
    phi_new  = 1.0 / math.sqrt((1.0 / (phi_star**2)) + (1.0 / v))
    mu_new   = mu + (phi_new**2) * gj * (s - e)

    r_new  = 1500.0 + G2_SCALE * mu_new
    RD_new = G2_SCALE * phi_new
    return r_new, RD_new, sigma_prime, e

def coalesce_dates(df) -> pd.Series:
    """Return a single datetime series from any of ['Date','date','match_date']."""
    candidates = [c for c in df.columns if c.lower() in ("date", "match_date")]
    if "Date" not in df.columns and candidates:
        # merge multiple candidates left-to-right
        s = None
        for c in candidates:
            tmp = pd.to_datetime(df[c], errors="coerce")
            s = tmp if s is None else s.fillna(tmp)
        return s
    # default: use 'Date'
    return pd.to_datetime(df["Date"], errors="coerce")

t0 = time.perf_counter()
print("▶︎ Stage 1/5: reading ONE ROW PER TEAM–GAME from DB...")

# ---------- 1) ONE ROW PER TEAM–GAME ----------
# Approach: pick the first row per (Date, Team, Opponent) from player table.
# We try a PG DISTINCT ON variant; fall back to a portable window; fall back to pandas.
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
    # portable variant: filter inside subquery
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

# try PG path, else window path, else last-resort Pandas
try:
    tg = pd.read_sql(text(team_game_sql_pg), engine, params=params)
except Exception:
    try:
        tg = pd.read_sql(text(team_game_sql_win), engine, params=params)
    except Exception:
        # portable fallback: read minimal columns then dedupe in pandas
        cols = '"Date","Team","Opponent","Round","Venue","Timeslot","Team Score" AS team_score'
        where_clause = 'WHERE "Date" > :after' if args.only_after else ''
        q = text(f'SELECT {cols} FROM {args.source} {where_clause}')
        raw = pd.read_sql(q, engine, params=params)
        tg = (raw.sort_values(["Date","Team","Opponent","Player"])
                .drop_duplicates(subset=["Date","Team","Opponent"])
                .drop(columns=["Player"], errors="ignore"))

# Date coalescing & cleaning
tg["Date"] = coalesce_dates(tg)
nat_count = tg["Date"].isna().sum()
if nat_count:
    print(f"   ⚠️ dropping {nat_count} rows with invalid dates")
tg = tg.dropna(subset=["Date"]).copy()

# normalize names
for col in ("Team","Opponent","Venue"):
    tg[col] = tg[col].astype(str).str.strip()
tg["Team"] = tg["Team"].replace(ALIASES)
tg["Opponent"] = tg["Opponent"].replace(ALIASES)

tg["team_score"] = pd.to_numeric(tg["team_score"], errors="coerce")
tg = tg.dropna(subset=["team_score"])

print(f"   ✓ team-game rows: {len(tg):,} (took {time.perf_counter()-t0:.2f}s)")

# ---------- 2) ONE ROW PER MATCH ----------
t1 = time.perf_counter()
print("▶︎ Stage 2/5: pairing team-games into matches...")

# key by (date, unordered teams)
tg["_pair_key"] = (
    tg["Date"].dt.strftime("%Y-%m-%d") + "|" +
    tg[["Team","Opponent"]].apply(lambda x: "|".join(sorted(x)), axis=1)
)

# sanity on malformed pairs
counts = tg["_pair_key"].value_counts()
bad_keys = counts[counts != 2]
if len(bad_keys):
    print(f"   ⚠️ {len(bad_keys)} malformed matches (not 2 rows) — they will be skipped.")
    dbg = tg[tg["_pair_key"].isin(bad_keys.index)].sort_values(["Date","Team","Opponent"])
    # print a few to help fix data
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

# ---------- 3) Per-team frame + tallies (vectorised) ----------
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

# ---------- 4) Iterate matches -> Elo + Glicko2 (bounded, fast) ----------
print("▶︎ Stage 4/5: updating Elo + Glicko2...")

elo = {}                                 # team -> rating
g2  = {}                                 # team -> (r, RD, vol)
last_season_seen = {}                    # team -> season (for start-of-season logic)

rows = []

for row in pbar(matches.itertuples(index=False), total=len(matches), desc="Matches"):
    date   = row.Date
    season = int(row.season)
    teamA, teamB = row.team_a, row.team_b
    scA, scB     = float(row.score_a), float(row.score_b)

    # outcomes
    SA = 1.0 if scA>scB else 0.0 if scA<scB else 0.5
    SB = 1.0 - SA if SA in (0.0,1.0) else 0.5

    # pre states
    RA = elo.get(teamA, INIT_ELO); RB = elo.get(teamB, INIT_ELO)
    rA, rdA, volA = g2.get(teamA, (G2_INIT_R, G2_INIT_RD, G2_INIT_VOL))
    rB, rdB, volB = g2.get(teamB, (G2_INIT_R, G2_INIT_RD, G2_INIT_VOL))

    # start-of-season adjustments
    if last_season_seen.get(teamA) not in (None, season):
        # new season for A
        if args.elo_regress > 0:
            RA = (1.0 - args.elo_regress)*RA + args.elo_regress*INIT_ELO
        if args.g2_rd_inflate > 0:
            rdA = min(350.0, max(30.0, rdA * (1.0 + args.g2_rd_inflate)))
    if last_season_seen.get(teamB) not in (None, season):
        if args.elo_regress > 0:
            RB = (1.0 - args.elo_regress)*RB + args.elo_regress*INIT_ELO
        if args.g2_rd_inflate > 0:
            rdB = min(350.0, max(30.0, rdB * (1.0 + args.g2_rd_inflate)))

    # determine home advantage (per venue mapping)
    H_A = args.home_adv if is_home(teamA, row.Venue) else 0.0
    H_B = args.home_adv if is_home(teamB, row.Venue) else 0.0

    # Elo expectation (allow both sides to have HGA if venue ambiguous = 0.0 for both)
    EA = 1.0 / (1.0 + 10.0 ** (-((RA + H_A) - (RB + H_B)) / 400.0))
    EB = 1.0 - EA

    RA_post = RA + args.k_elo * (SA - EA)
    RB_post = RB + args.k_elo * (SB - EB)

    # Glicko-2
    if not args.skip_glicko:
        rA_post, rdA_post, volA_post, _ = glicko2_one_match(rA, rdA, volA, SA, rB, rdB,
                                                            tau=args.glicko_tau, tol=args.glicko_tol)
        rB_post, rdB_post, volB_post, _ = glicko2_one_match(rB, rdB, volB, SB, rA, rdA,
                                                            tau=args.glicko_tau, tol=args.glicko_tol)
    else:
        rA_post, rdA_post, volA_post = rA, rdA, volA
        rB_post, rdB_post, volB_post = rB, rdB, volB

    # tallies for this match
    A_info = lookup.loc[(date, teamA, teamB)]
    B_info = lookup.loc[(date, teamB, teamA)]

    def pack(team, opp, pf, pa, info, pre, post, exp, r_pre, rd_pre, vol_pre, r_po, rd_po, vol_po):
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
        }

    rows.append(pack(teamA, teamB, scA, scB, A_info, RA, RA_post, EA, rA, rdA, volA, rA_post, rdA_post, volA_post))
    rows.append(pack(teamB, teamA, scB, scA, B_info, RB, RB_post, EB, rB, rdB, volB, rB_post, rdB_post, volB_post))

    # commit states
    elo[teamA], elo[teamB] = RA_post, RB_post
    g2[teamA] = (rA_post, rdA_post, volA_post)
    g2[teamB] = (rB_post, rdB_post, volB_post)
    last_season_seen[teamA] = season
    last_season_seen[teamB] = season

team_table = pd.DataFrame(rows).sort_values(["Date","Team"]).reset_index(drop=True)
print(f"   ✓ ratings computed for {len(matches):,} matches")

# ---------- 5) Write per-game audit + season summary (finals excluded) ----------
t3 = time.perf_counter()
print("▶︎ Stage 5/5: writing tables...")

# keep per-game ratings for auditing
team_table.to_sql("team_games", engine, if_exists="replace", index=False,
                  method="multi", chunksize=args.chunksize)

# finals filter
FINALS = {"EF", "QF", "SF", "PF", "GF"}
def is_finals_round(x):
    if x is None: 
        return False
    s = str(x).strip().upper()
    return s in FINALS

team_table["is_finals"] = team_table["Round"].apply(is_finals_round)

# ---- Regular-season only for ladder stats ----
reg = team_table[~team_table["is_finals"]].copy()

# season totals (regular season)
season_totals = (
    reg.groupby(["season","Team"], as_index=False)
       .agg(
           season_wins   = ("result", lambda s: (s=="Win").sum()),
           season_losses = ("result", lambda s: (s=="Loss").sum()),
           season_draws  = ("result", lambda s: (s=="Draw").sum()),
           pf            = ("points_for", "sum"),
           pa            = ("points_against", "sum"),
       )
)

# percentage from regular-season PF/PA
season_totals["season_percentage"] = 100.0 * season_totals["pf"] / season_totals["pa"].replace(0, np.nan)

# final post-ratings after last game (can be finals as well)
last_posts = (team_table
              .sort_values(["season","Team","Date"])
              .groupby(["season","Team"], as_index=False)
              .tail(1)
              [["season","Team","elo_post","g2_rating_post","g2_rd_post"]]
              .rename(columns={
                  "elo_post":"elo",
                  "g2_rating_post":"glicko",
                  "g2_rd_post":"glicko_rd"
              }))

summary = season_totals.merge(last_posts, on=["season","Team"], how="left")

# ladder points/position (AFL: 4 for win, 2 for draw) — regular season only
summary["ladder_points"] = 4*summary["season_wins"] + 2*summary["season_draws"]

summary = summary.sort_values(["season","ladder_points","season_percentage","pf"],
                              ascending=[True, False, False, False])
summary["ladder_position"] = summary.groupby("season").cumcount() + 1

summary = summary[[
    "season", "Team",
    "elo", "glicko", "glicko_rd",
    "season_wins", "season_losses", "season_draws",
    "season_percentage", "ladder_points", "ladder_position"
]]

summary.to_sql(args.dest, engine, if_exists="replace", index=False,
               method="multi", chunksize=args.chunksize)

print(f"✅ Wrote per-game table 'team_games' ({len(team_table):,} rows) "
      f"and season summary '{args.dest}' ({len(summary):,} teams) in {time.perf_counter()-t3:.2f}s.")
print("✅ Done.")
