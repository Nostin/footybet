# main.py - uvicorn main:app --reload
from typing import Any, Dict, Optional, List
from fastapi import FastAPI, HTTPException, Query, Depends
from sqlalchemy import select, text as sqtext, func, desc, cast, Integer, extract
from sqlalchemy.ext.asyncio import AsyncSession
from models import PlayerPrecompute, PlayerNickname, PlayerStats, Teams, TeamPrecompute, UpcomingGame, TeamGame, Tips, TipRanks
from fastapi.middleware.cors import CORSMiddleware
from db import get_session
from datetime import date as DateType, datetime as DateTime

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add/keep your stat map as-is
STAT_COLUMN_MAP: Dict[str, Any] = {
    "disposals":  PlayerStats.Disposals,
    "goals":      PlayerStats.Goals,
    "marks":      PlayerStats.Marks,
    "tackles":    PlayerStats.Tackles,
    "clearances": PlayerStats.Clearances,
    "kicks":      PlayerStats.Kicks,
    "handballs":  PlayerStats.Handballs,
}

def _to_date(value) -> Optional[DateType]:
    if value is None:
        return None

    if isinstance(value, DateType) and not isinstance(value, DateTime):
        return value

    if isinstance(value, DateTime):
        return value.date()

    if isinstance(value, str):
        s = value.strip()
        # Accept ISO: YYYY-MM-DD
        try:
            return DateType.fromisoformat(s)
        except ValueError:
            pass

        # Accept D/M/YYYY (your old CSV style)
        try:
            d, m, y = s.split("/")
            return DateType(int(y), int(m), int(d))
        except Exception:
            return None

    return None

async def _get_opponent_team_stats(session: AsyncSession, opponent_team: str, date_value) -> Optional[dict]:
    dt = _to_date(date_value)
    if not dt:
        return None

    stmt = (
        select(
            PlayerStats.TeamScore.label("opp_score"),
            PlayerStats.TeamInside50.label("opp_inside50"),
            PlayerStats.TeamTurnovers.label("opp_turnovers"),
            PlayerStats.TeamFreeKicks.label("opp_free_kicks"),
        )
        .where(
            PlayerStats.Team == opponent_team,
            PlayerStats.Date == dt,
        )
        .limit(1)
    )

    res = await session.execute(stmt)
    return res.mappings().first()

async def _get_player_stat_game_by_col(
    col,
    player_name: str,
    value: int,
    session: AsyncSession,
    latest: bool = True,  # False = earliest match
):
    stmt = (
        select(PlayerStats)
        .where(PlayerStats.Player == player_name, col == value)
        .order_by(PlayerStats.Date.desc() if latest else PlayerStats.Date.asc())
        .limit(1)
    )
    result = await session.execute(stmt)
    row = result.scalars().first()
    if row is None:
        raise HTTPException(status_code=404, detail="Game not found")

    data = row.to_dict()

    # attach opponent team metrics
    opp = row.Opponent
    dt  = row.Date
    opp_stats = await _get_opponent_team_stats(session, opp, dt)
    if opp_stats:
        # add with camelCase-style keys you asked for
        if opp_stats.get("opp_inside50") is not None:
            data["OpponentInside50"]  = int(opp_stats["opp_inside50"])
        if opp_stats.get("opp_free_kicks") is not None:
            data["OpponentFreeKicks"] = int(opp_stats["opp_free_kicks"])
        if opp_stats.get("opp_turnovers") is not None:
            data["OpponentTurnovers"] = int(opp_stats["opp_turnovers"])
        if opp_stats.get("opp_score") is not None:
            data["OpponentScore"]     = int(opp_stats["opp_score"])

        # margin if both sides' scores are present
        try:
            team_score = int(data.get("Team Score")) if data.get("Team Score") is not None else None
        except Exception:
            team_score = None
        opp_score = opp_stats.get("opp_score")
        if team_score is not None and opp_score is not None:
            data["Margin"] = team_score - int(opp_score)

    return data

async def _get_team_record_and_form(
    session: AsyncSession,
    team_name: str,
    last_n: int = 5,
) -> dict:
    """
    Returns current-season W/L/D record and last N results for a team,
    sourced from team_games (proper timestamp column).
    """

    current_year = DateType.today().year

    # ---------- 1) CURRENT-SEASON RECORD (team_games) ----------
    record_stmt = (
        select(
            TeamGame.result,
            func.count().label("games"),
        )
        .where(
            TeamGame.Team == team_name,
            TeamGame.season == current_year,
        )
        .group_by(TeamGame.result)
    )

    record_res = await session.execute(record_stmt)

    wins = losses = draws = 0
    for game_result, games in record_res.all():
        if not game_result:
            continue
        gr = str(game_result).lower()
        if gr.startswith("win"):
            wins += games
        elif gr.startswith("loss"):
            losses += games
        elif gr.startswith("draw"):
            draws += games

    # ---------- 2) LAST N RESULTS (team_games, all time) ----------
    last_stmt = (
        select(
            TeamGame.Date,
            TeamGame.result,
            TeamGame.Opponent,
            TeamGame.Round,
            TeamGame.Venue,
            TeamGame.points_for,
            TeamGame.points_against,
        )
        .where(TeamGame.Team == team_name)
        .order_by(TeamGame.Date.desc())
        .limit(last_n)
    )

    last_res = await session.execute(last_stmt)
    last_rows = last_res.all()

    last_results = [
        {
            "date": row[0].date() if row[0] else None,  # return date instead of datetime (optional)
            "result": row[1],
            "opponent": row[2],
            "round": row[3],
            "venue": row[4],
            "margin": (row[5] - row[6]) if (row[5] is not None and row[6] is not None) else None,
            "points_for": row[5],
            "points_against": row[6],
        }
        for row in last_rows
    ]

    return {
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "last_results": last_results,
    }

def register_stat_route(stat_name: str, col):
    path = f"/{stat_name}/" + "{player_name}/{value}"

    @app.get(path)
    async def _route(
        player_name: str,
        value: int,
        session: AsyncSession = Depends(get_session),
        _col=col,  # bind default to avoid late-binding issues
    ):
        return await _get_player_stat_game_by_col(_col, player_name, value, session)

async def _get_team_stat_game_from_team_precompute(
    stat_col: str,                 # e.g. "team_score"
    team_name: str,
    value: int,
    session: AsyncSession,
    latest: bool = True,
):
    order = "DESC" if latest else "ASC"
    q = sqtext(f'''
        SELECT *
        FROM team_precompute
        WHERE "Team" = :team AND {stat_col} = :val
        ORDER BY "Date" {order}
        LIMIT 1
    ''')
    res = await session.execute(q, {"team": team_name, "val": value})
    row = res.mappings().first()
    if not row:
        raise HTTPException(status_code=404, detail="Game not found")

    data = dict(row)

    # --- Optional: enrich with opponent team metrics & margin (same as before) ---
    opp = data.get("Opponent")
    dt  = data.get("Date")
    if opp and dt:
        opp_stats = await _get_opponent_team_stats(session, opp, dt)
        if opp_stats:
            if opp_stats.get("opp_inside50") is not None:
                data["OpponentInside50"]  = int(opp_stats["opp_inside50"])
            if opp_stats.get("opp_free_kicks") is not None:
                data["OpponentFreeKicks"] = int(opp_stats["opp_free_kicks"])
            if opp_stats.get("opp_turnovers") is not None:
                data["OpponentTurnovers"] = int(opp_stats["opp_turnovers"])
            if opp_stats.get("opp_score") is not None:
                data["OpponentScore"]     = int(opp_stats["opp_score"])
            try:
                if data.get("Team Score") is not None and opp_stats.get("opp_score") is not None:
                    data["Margin"] = int(data["Team Score"]) - int(opp_stats["opp_score"])
            except Exception:
                pass

    return data

def register_team_precompute_stat_route(stat_name: str, stat_col: str):
    # e.g. /team-score/Hawthorn/150?latest=true
    path = f"/team-{stat_name}/" + "{team_name}/{value}"

    @app.get(path)
    async def _route(
        team_name: str,
        value: int,
        latest: bool = True,
        session: AsyncSession = Depends(get_session),
        _stat_col=stat_col,
    ):
        return await _get_team_stat_game_from_team_precompute(_stat_col, team_name, value, session, latest)

for name, col in STAT_COLUMN_MAP.items():
    register_stat_route(name, col)

register_team_precompute_stat_route("score", "team_score")
register_team_precompute_stat_route("disposals", "team_disposals")

@app.get("/player/{player_name}")
async def get_player(player_name: str, session: AsyncSession = Depends(get_session)):
    stmt = select(PlayerPrecompute).where(PlayerPrecompute.Player == player_name)
    result = await session.execute(stmt)
    player = result.scalar_one_or_none()

    if not player:
        raise HTTPException(status_code=404, detail="Player not found")

    return player.to_dict()

@app.get("/player-disposals/{player_name}")
async def get_player_disposals(player_name: str, session: AsyncSession = Depends(get_session)):
    q = sqtext("""
        SELECT *
        FROM player_precomputes_disposals
        WHERE "Player" = :player
        LIMIT 1
    """)
    res = await session.execute(q, {"player": player_name})
    row = res.mappings().first()  # returns dict-like mapping
    if not row:
        raise HTTPException(status_code=404, detail="Player not found")
    return dict(row)  # already the disposals-only view columns
    
@app.get("/players")
async def get_all_player_names(session: AsyncSession = Depends(get_session)):
    # Get all real player names
    stmt_real = select(PlayerPrecompute.Player)
    result_real = await session.execute(stmt_real)
    real_names = result_real.scalars().all()

    # Get all nicknames and map to real names
    stmt_nick = select(PlayerNickname.Nickname, PlayerNickname.Player)
    result_nick = await session.execute(stmt_nick)
    nicknames = result_nick.all()

    # Combine into a list of (searchable_label, actual_player_name)
    merged = set()
    for name in real_names:
        merged.add((name, name))  # self-reference for real names

    for nickname, player in nicknames:
        merged.add((nickname, player))  # nickname points to real player

    return [{"label": label, "value": player} for label, player in merged]

@app.get("/top-disposals")
async def get_top_disposal_players(session: AsyncSession = Depends(get_session)):
    stmt = (
        select(PlayerPrecompute)
        .where(PlayerPrecompute.Disposal_Season_Avg != None)
        .order_by(PlayerPrecompute.Disposal_Season_Avg.desc())
        .limit(20)
    )
    result = await session.execute(stmt)
    players = result.scalars().all()
    return [player.to_dict() for player in players]


@app.get("/players/by-names")
async def get_players_by_names(names: str = Query(..., description="Comma-separated player names"), session: AsyncSession = Depends(get_session)):
    player_names = [name.strip() for name in names.split(",")]

    stmt = select(PlayerPrecompute).where(PlayerPrecompute.Player.in_(player_names))
    result = await session.execute(stmt)
    players = result.scalars().all()

    # Map by name for reordering
    player_map = {p.Player: p for p in players}

    # Only include players that were found, in original order
    ordered_players = [
        player_map[name].to_dict()
        for name in player_names
        if name in player_map
    ]

    return ordered_players
    
# get all players by team name eg: /player-by-team/Carlton
@app.get("/player-by-team/{team_name}")
async def get_players_by_team_path(team_name: str, session: AsyncSession = Depends(get_session)):
    stmt = select(PlayerPrecompute).where(PlayerPrecompute.Team == team_name)
    result = await session.execute(stmt)
    players = result.scalars().all()

    if not players:
        raise HTTPException(status_code=404, detail="Team not found")

    return [player.to_dict() for player in players]

# get single team by team name eg: /team/Carlton
@app.get("/team/{team_name}")
async def get_team(team_name: str, session: AsyncSession = Depends(get_session)):
    # latest precompute row
    pre_stmt = (
        select(TeamPrecompute)
        .where(TeamPrecompute.Team == team_name)
        .order_by(desc(TeamPrecompute.Date))
        .limit(1)
    )
    pre_res = await session.execute(pre_stmt)
    pre = pre_res.scalar_one_or_none()

    if not pre:
        raise HTTPException(status_code=404, detail="Team not found")

    season = getattr(pre, "season", None) or pre.Date.year  # fallback

    # season summary from teams table
    teams_stmt = (
        select(Teams)
        .where(Teams.Team == team_name, Teams.season == season)
        .limit(1)
    )
    teams_res = await session.execute(teams_stmt)
    season_row = teams_res.scalar_one_or_none()

    # record + last N results (reused helper; uses team_games)
    form = await _get_team_record_and_form(session, team_name, last_n=5)

    payload = pre.to_dict()
    payload["season_summary"] = season_row.to_dict() if season_row else None

    # attach last results + record
    payload["record"] = {
        "wins": form.get("wins"),
        "losses": form.get("losses"),
        "draws": form.get("draws"),
    }
    payload["last_results"] = form.get("last_results")  # or call it last_5_results if you prefer

    return payload

    
# get all teams
@app.get("/teams")
async def get_teams(session: AsyncSession = Depends(get_session)):
    stmt = (
        select(TeamPrecompute)
    )
    result = await session.execute(stmt)
    teams = result.scalars().all()
    return [team.to_dict() for team in teams]

@app.get("/teams/by-names")
async def get_teams_by_names(names: str = Query(..., description="Comma-separated team names"), session: AsyncSession = Depends(get_session)):
    team_names = [name.strip() for name in names.split(",")]

    stmt = select(TeamPrecompute).where(TeamPrecompute.Team.in_(team_names))
    result = await session.execute(stmt)
    teams = result.scalars().all()

    # Map by name for reordering
    team_map = {t.Team: t for t in teams}

    # Only include players that were found, in original order
    ordered_players = [
        team_map[name].to_dict()
        for name in team_names
        if name in team_map
    ]

    return ordered_players

from models import UpcomingGameWithTips

@app.get("/upcoming_games")
async def get_upcoming_games(session: AsyncSession = Depends(get_session)):
    stmt_games = (
        select(UpcomingGameWithTips)
        .where(UpcomingGameWithTips.Date >= func.current_date())
        .order_by(UpcomingGameWithTips.Date, UpcomingGameWithTips.Timeslot)
    )

    res_games = await session.execute(stmt_games)
    games = res_games.scalars().all()

    if not games:
        return []

    team_names: set[str] = set()
    for g in games:
        if g.HomeTeam:
            team_names.add(g.HomeTeam)
        if g.AwayTeam:
            team_names.add(g.AwayTeam)

    stmt_teams = select(TeamPrecompute).where(TeamPrecompute.Team.in_(team_names))
    res_teams = await session.execute(stmt_teams)
    team_rows = res_teams.scalars().all()
    team_map = {t.Team: t for t in team_rows}

    team_form_cache: dict[str, dict] = {}
    for name in team_names:
        team_form_cache[name] = await _get_team_record_and_form(session, name, last_n=5)

    output = []
    for g in games:
        home_name = g.HomeTeam
        away_name = g.AwayTeam

        home_team = team_map.get(home_name)
        away_team = team_map.get(away_name)

        home_form = team_form_cache.get(home_name, {})
        away_form = team_form_cache.get(away_name, {})

        def ladder_for_team(team_obj, form: Optional[dict]):
            if not team_obj:
                return None
            base_ladder = getattr(team_obj, "ladder_position", None)
            if base_ladder is None:
                return None
            w = (form or {}).get("wins") or 0
            l = (form or {}).get("losses") or 0
            d = (form or {}).get("draws") or 0
            return None if (w + l + d == 0) else base_ladder

        home_ladder = ladder_for_team(home_team, home_form)
        away_ladder = ladder_for_team(away_team, away_form)

        game_payload = {
            "date": g.Date,
            "venue": g.Venue,
            "timeslot": g.Timeslot,
            "round": g.Round,
            "tip": {
                "team": g.Tip,
                "confidence": g.TipConfidence,
                "margin": g.TipMargin,
                "correct": g.Correct,
                "actual_margin": g.ActualMargin,
            } if g.Tip is not None else None,
            "home_team": {
                "name": home_name,
                "ladder_position": home_ladder,
                "elo_rating": getattr(home_team, "elo_rating", None) if home_team else None,
                "glicko_rating": getattr(home_team, "glicko_rating", None) if home_team else None,
                "ml_win_probability": getattr(home_team, "Win_Probability", None) if home_team else None,
                "record": {
                    "wins": home_form.get("wins"),
                    "losses": home_form.get("losses"),
                    "draws": home_form.get("draws"),
                },
                "last_5_results": home_form.get("last_results"),
            },
            "away_team": {
                "name": away_name,
                "ladder_position": away_ladder,
                "elo_rating": getattr(away_team, "elo_rating", None) if away_team else None,
                "glicko_rating": getattr(away_team, "glicko_rating", None) if away_team else None,
                "ml_win_probability": getattr(away_team, "Win_Probability", None) if away_team else None,
                "record": {
                    "wins": away_form.get("wins"),
                    "losses": away_form.get("losses"),
                    "draws": away_form.get("draws"),
                },
                "last_5_results": away_form.get("last_results"),
            },
        }

        output.append(game_payload)

    return output

@app.get("/tips")
async def get_tips(
    season: Optional[int] = Query(None, description="Season year, e.g. 2026"),
    session: AsyncSession = Depends(get_session),
):
    tips_stmt = select(Tips).order_by(Tips.Date, Tips.Round, Tips.Venue, Tips.HomeTeam)

    # If season is provided, filter tips in SQL (fast + less data)
    if season is not None:
        tips_stmt = tips_stmt.where(extract("year", Tips.Date) == season)

    tips_rows = (await session.execute(tips_stmt)).scalars().all()
    if not tips_rows:
        return {"seasons": []}

    ranks_stmt = select(TipRanks)
    # Optional: filter ranks too (nice + consistent)
    if season is not None:
        ranks_stmt = ranks_stmt.where(TipRanks.Season == season)

    ranks_rows = (await session.execute(ranks_stmt)).scalars().all()

    FINAL_ORDER = {"QF": 100, "EF": 101, "SF": 102, "PF": 103, "GF": 104}

    def normalize_round_key(raw) -> str:
        if raw is None:
            return ""
        k = str(raw).strip().upper()
        if k in {"OPENING ROUND", "OPEN ROUND"}:
            return "OR"
        if k.startswith("ROUND "):
            k = k.replace("ROUND ", "").strip()
        if k.startswith("R") and k[1:].isdigit():
            return k[1:]
        long_map = {
            "QUALIFYING FINAL": "QF",
            "ELIMINATION FINAL": "EF",
            "SEMI FINAL": "SF",
            "PRELIMINARY FINAL": "PF",
            "GRAND FINAL": "GF",
        }
        return long_map.get(k, k)

    def round_sort_key(raw):
        k = normalize_round_key(raw)
        if k == "OR":
            return (0, 0)
        if k.isdigit():
            return (1, int(k))
        if k in FINAL_ORDER:
            return (2, FINAL_ORDER[k])
        return (3, k)

    rank_map: Dict[tuple[int, str], Dict[str, Any]] = {}
    for r in ranks_rows:
        if r.Season is None or r.Round is None:
            continue
        rank_map[(int(r.Season), normalize_round_key(r.Round))] = {
            "overall_rank": r.Rank,
            "total_players": r.Total,
        }

    seasons: Dict[int, Dict[str, Any]] = {}

    def ensure_season(season: int) -> Dict[str, Any]:
        if season not in seasons:
            seasons[season] = {"season": season, "rounds": {}, "summary": {}}
        return seasons[season]

    def ensure_round(season_obj: Dict[str, Any], round_key: str) -> Dict[str, Any]:
        rounds = season_obj["rounds"]
        if round_key not in rounds:
            rounds[round_key] = {"round": round_key, "matches": [], "summary": {}}
        return rounds[round_key]

    for t in tips_rows:
        if not t.Date:
            continue
        season = t.Date.year
        round_key = normalize_round_key(t.Round)

        season_obj = ensure_season(season)
        round_obj = ensure_round(season_obj, round_key)

        round_obj["matches"].append({
            "date": t.Date.isoformat(),
            "venue": t.Venue,
            "home_team": t.HomeTeam,
            "away_team": t.AwayTeam,
            "timeslot": t.Timeslot,
            "round": round_key,
            "season": season,
            "tip": t.Tip,
            "confidence": t.TipConfidence,
            "tip_margin": t.TipMargin,
            "correct": t.Correct,
            "actual_margin": t.ActualMargin,
        })

    def summarize_matches(matches: list[dict], season: int, round_key: Optional[str]):
        total_tips = sum(1 for m in matches if m.get("tip") is not None)

        decided = [m for m in matches if m.get("correct") in ("Yes", "No")]
        total_decided = len(decided)
        total_correct = sum(1 for m in decided if m.get("correct") == "Yes")
        accuracy = (total_correct / total_decided) if total_decided else None

        diffs = []
        for m in matches:
            tm = m.get("tip_margin")
            am = m.get("actual_margin")
            if tm is None or am is None:
                continue
            try:
                diffs.append(abs(int(tm) - int(am)))  # cumulative absolute error
            except Exception:
                continue

        margin_error_total = sum(diffs) if diffs else None
        margin_error_avg = (margin_error_total / len(diffs)) if diffs else None

        def conf_correct(level: str):
            mm = [m for m in decided if (m.get("confidence") or "").strip().lower() == level.lower()]
            return {"tips": len(mm), "correct": sum(1 for x in mm if x["correct"] == "Yes")}

        summary: Dict[str, Any] = {
            "total_tips": total_tips,
            "total_correct": total_correct,
            "accuracy": accuracy,
            "margin_error_total": margin_error_total,
            "margin_error_avg": margin_error_avg,
            "confidence_breakdown": {
                "high": conf_correct("High"),
                "moderate": conf_correct("Moderate"),
                "low": conf_correct("Low"),
            },
        }

        if round_key is not None:
            nrk = normalize_round_key(round_key)
            rk = rank_map.get((season, nrk))
            if rk:
                summary.update(rk)
                if rk.get("overall_rank") and rk.get("total_players"):
                    summary["rank_fraction"] = rk["overall_rank"] / rk["total_players"]
                    summary["top_percent"] = 100.0 * summary["rank_fraction"]

        return summary

    for season, season_obj in seasons.items():
        # round summaries
        for rk, round_obj in season_obj["rounds"].items():
            round_obj["summary"] = summarize_matches(round_obj["matches"], season, rk)

        # season summary across all rounds
        all_matches: List[Dict[str, Any]] = []
        for r in season_obj["rounds"].values():
            all_matches.extend(r["matches"])

        ranked_round_keys = []
        for rk in season_obj["rounds"].keys():
            nrk = normalize_round_key(rk)
            if (season, nrk) in rank_map:
                ranked_round_keys.append(nrk)

        latest_rank_round = None
        if ranked_round_keys:
            latest_rank_round = sorted(ranked_round_keys, key=round_sort_key)[-1]

        season_obj["summary"] = summarize_matches(all_matches, season, latest_rank_round)

        # ORDER ROUNDS for frontend: OR, 1.., finals...
        ordered_round_keys = sorted(season_obj["rounds"].keys(), key=round_sort_key)
        season_obj["rounds"] = [season_obj["rounds"][k] for k in ordered_round_keys]

    return {"seasons": [seasons[s] for s in sorted(seasons.keys())]}
