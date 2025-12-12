# main.py - uvicorn main:app --reload
from typing import Any, Dict, Optional, List
from fastapi import FastAPI, HTTPException, Query, Depends
from sqlalchemy import select, text as sqtext, func
from sqlalchemy.ext.asyncio import AsyncSession
from models import PlayerPrecompute, PlayerNickname, PlayerStats, TeamPrecompute, UpcomingGame
from fastapi.middleware.cors import CORSMiddleware
from db import get_session
from datetime import date, datetime

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

def _to_date_str(value) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        return value  # assume already like 'YYYY-MM-DD'
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    # fallback
    return str(value)

async def _get_opponent_team_stats(
    session: AsyncSession,
    opponent_team: str,
    date_value,  # accept any, we'll normalize
) -> Optional[dict]:
    dt_str = _to_date_str(date_value)

    stmt = (
        select(
            PlayerStats.TeamScore.label("opp_score"),
            PlayerStats.TeamInside50.label("opp_inside50"),
            PlayerStats.TeamTurnovers.label("opp_turnovers"),
            PlayerStats.TeamFreeKicks.label("opp_free_kicks"),
        )
        .where(
            PlayerStats.Team == opponent_team,
            PlayerStats.Date == dt_str,   # string-to-string compare
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
    Returns current-year W/L/D record and last N results for a team.
    Assumes PlayerStats.Date is stored as 'YYYY-MM-DD' text.
    """

    current_year = date.today().year
    year_prefix = f"{current_year}-"  # e.g. '2025-'

    # Base filter: this team, this year
    base_filters = (
        PlayerStats.Team == team_name,
        PlayerStats.Date.like(f"{year_prefix}%"),
    )

    # Overall record for current year
    record_stmt = (
        select(
            PlayerStats.GameResult,
            func.count(func.distinct(PlayerStats.Date)).label("games"),
        )
        .where(*base_filters)
        .group_by(PlayerStats.GameResult)
    )
    record_res = await session.execute(record_stmt)

    wins = losses = draws = 0
    for game_result, games in record_res.all():
        if not game_result:
            continue
        gr = game_result.lower()
        if gr.startswith("win"):
            wins += games
        elif gr.startswith("loss"):
            losses += games
        else:
            draws += games

    # Last N results within current year
    last_stmt = (
        select(
            PlayerStats.Date,
            PlayerStats.GameResult,
            PlayerStats.Opponent,
        )
        .where(*base_filters)
        .group_by(PlayerStats.Date, PlayerStats.GameResult, PlayerStats.Opponent)
        .order_by(PlayerStats.Date.desc())
        .limit(last_n)
    )
    last_res = await session.execute(last_stmt)
    last_rows = last_res.all()

    last_results: List[dict] = [
        {
            "date": row[0],
            "result": row[1],
            "opponent": row[2],
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
    stmt = select(TeamPrecompute).where(TeamPrecompute.Team == team_name)
    result = await session.execute(stmt)
    team = result.scalar_one_or_none()

    if not team:
        raise HTTPException(status_code=404, detail="Team not found")

    return team.to_dict()
    
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

@app.get("/upcoming_games")
async def get_upcoming_games(session: AsyncSession = Depends(get_session)):
    # 1. Filter using Postgres date conversion: 'DD/MM/YYYY' → date
    stmt_games = (
        select(UpcomingGame)
        .where(
            func.to_date(UpcomingGame.Date, 'DD/MM/YYYY') > func.current_date()
            # If you want to include today as well, use >= instead
            # func.to_date(UpcomingGame.Date, 'DD/MM/YYYY') >= func.current_date()
        )
        .order_by(
            func.to_date(UpcomingGame.Date, 'DD/MM/YYYY'),
            UpcomingGame.Timeslot,
        )
    )

    res_games = await session.execute(stmt_games)
    games = res_games.scalars().all()

    if not games:
        return []

    # 2. Collect unique teams involved
    team_names: set[str] = set()
    for g in games:
        if g.HomeTeam:
            team_names.add(g.HomeTeam)
        if g.AwayTeam:
            team_names.add(g.AwayTeam)

    # 3. Fetch precomputed team-level data in bulk
    stmt_teams = select(TeamPrecompute).where(TeamPrecompute.Team.in_(team_names))
    res_teams = await session.execute(stmt_teams)
    team_rows = res_teams.scalars().all()
    team_map = {t.Team: t for t in team_rows}

    # 4. For each team, get record + last 5
    team_form_cache: dict[str, dict] = {}
    for name in team_names:
        team_form_cache[name] = await _get_team_record_and_form(session, name, last_n=5)

    # 5. Build response
    output = []
    for g in games:
        home_name = g.HomeTeam
        away_name = g.AwayTeam

        home_team = team_map.get(home_name)
        away_team = team_map.get(away_name)

        home_form = team_form_cache.get(home_name, {})
        away_form = team_form_cache.get(away_name, {})

        game_payload = {
            "date": g.Date,        # still the original "d/m/yyyy" string
            "venue": g.Venue,
            "timeslot": g.Timeslot,
            "home_team": {
                "name": home_name,
                "ladder_position": getattr(home_team, "ladder_position", None) if home_team else None,
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
                "ladder_position": getattr(away_team, "ladder_position", None) if away_team else None,
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
