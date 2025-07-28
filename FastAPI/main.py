# main.py
from fastapi import FastAPI, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from models import PlayerPrecompute

DB_NAME = "footy"
DB_USER = "seanthompson"
DB_PASS = ""  # Set if required
DB_HOST = "localhost"
DB_PORT = "5432"

DATABASE_URL = f"postgresql+asyncpg://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

engine = create_async_engine(DATABASE_URL, echo=True)
async_session = sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)

app = FastAPI()

@app.get("/player/{player_name}")
async def get_player(player_name: str):
    async with async_session() as session:
        stmt = select(PlayerPrecompute).where(PlayerPrecompute.Player == player_name)
        result = await session.execute(stmt)
        player = result.scalar_one_or_none()

        if not player:
            raise HTTPException(status_code=404, detail="Player not found")

        return player.to_dict()
