# main.py - uvicorn main:app --reload
from fastapi import FastAPI, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from models import PlayerPrecompute, PlayerNickname, TeamPrecompute
from fastapi.middleware.cors import CORSMiddleware

DB_NAME = "footy"
DB_USER = "seanthompson"
DB_PASS = ""  # Set if required
DB_HOST = "localhost"
DB_PORT = "5432"

DATABASE_URL = f"postgresql+asyncpg://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

engine = create_async_engine(DATABASE_URL, echo=True)
async_session = sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/player/{player_name}")
async def get_player(player_name: str):
    async with async_session() as session:
        stmt = select(PlayerPrecompute).where(PlayerPrecompute.Player == player_name)
        result = await session.execute(stmt)
        player = result.scalar_one_or_none()

        if not player:
            raise HTTPException(status_code=404, detail="Player not found")

        return player.to_dict()
    
@app.get("/players")
async def get_all_player_names():
    async with async_session() as session:
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
async def get_top_disposal_players():
    async with async_session() as session:
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
async def get_players_by_names(names: str = Query(..., description="Comma-separated player names")):
    player_names = [name.strip() for name in names.split(",")]

    async with async_session() as session:
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
async def get_player(team_name: str):
    async with async_session() as session:
        stmt = select(PlayerPrecompute).where(PlayerPrecompute.Team == team_name)
        result = await session.execute(stmt)
        players = result.scalars().all()

        if not players:
            raise HTTPException(status_code=404, detail="Team not found")

        return [player.to_dict() for player in players]

# get single team by team name eg: /team/Carlton
@app.get("/team/{team_name}")
async def get_team(team_name: str):
    async with async_session() as session:
        stmt = select(TeamPrecompute).where(TeamPrecompute.Team == team_name)
        result = await session.execute(stmt)
        team = result.scalar_one_or_none()

        if not team:
            raise HTTPException(status_code=404, detail="Team not found")

        return team.to_dict()
    
# get all teams
@app.get("/teams")
async def get_teams():
    async with async_session() as session:
        stmt = (
            select(TeamPrecompute)
        )
        result = await session.execute(stmt)
        teams = result.scalars().all()
        return [team.to_dict() for team in teams]

@app.get("/teams/by-names")
async def get_teams_by_names(names: str = Query(..., description="Comma-separated team names")):
    team_names = [name.strip() for name in names.split(",")]

    async with async_session() as session:
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