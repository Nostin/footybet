# main.py - uvicorn main:app --reload
from typing import Any, Dict, Optional
from fastapi import FastAPI, HTTPException, Query, Depends
from sqlalchemy import select, text as sqtext
from sqlalchemy.ext.asyncio import AsyncSession
from models import PlayerPrecompute, PlayerNickname, PlayerStats, TeamPrecompute
from fastapi.middleware.cors import CORSMiddleware
from db import get_session

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

STAT_COLUMN_MAP: Dict[str, Any] = {
    "disposals":  PlayerStats.Disposals,
    "goals":      PlayerStats.Goals,
    "marks":      PlayerStats.Marks,
    "tackles":    PlayerStats.Tackles,
    "clearances": PlayerStats.Clearances,
    "kicks":      PlayerStats.Kicks,
    "handballs":  PlayerStats.Handballs,
}

async def _get_opponent_score(
    session: AsyncSession,
    opponent_team: str,
    date_value: str,
) -> Optional[int]:
    stmt = (
        select(PlayerStats.TeamScore)
        .where(
            PlayerStats.Team == opponent_team,
            PlayerStats.Date == date_value,
        )
        .limit(1)
    )
    res = await session.execute(stmt)
    return res.scalar_one_or_none()

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

    opp = row.Opponent
    dt  = row.Date
    opp_score = await _get_opponent_score(session, opp, dt)
    if opp_score is not None:
        data["Opponent Score"] = int(opp_score)
        try:
            team_score = int(data.get("Team Score")) if data.get("Team Score") is not None else None
        except Exception:
            team_score = None
        if team_score is not None:
            data["Margin"] = team_score - opp_score

    return data

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

for name, col in STAT_COLUMN_MAP.items():
    register_stat_route(name, col)

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