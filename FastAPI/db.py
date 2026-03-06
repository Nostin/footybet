import os
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    DB_NAME = os.getenv("DB_NAME", "footy")
    DB_USER = os.getenv("DB_USER", "seanthompson")
    DB_PASS = os.getenv("DB_PASS", "")
    DB_HOST = os.getenv("DB_HOST", "localhost")
    DB_PORT = os.getenv("DB_PORT", "5432")

    DATABASE_URL = f"postgresql+asyncpg://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

else:
    if DATABASE_URL.startswith("postgresql://"):
        DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://", 1)

    if "sslmode=require" in DATABASE_URL:
        DATABASE_URL = DATABASE_URL.replace("sslmode=require", "ssl=require")

    DATABASE_URL = DATABASE_URL.replace("&channel_binding=require", "")
    DATABASE_URL = DATABASE_URL.replace("?channel_binding=require", "")

engine = create_async_engine(DATABASE_URL, echo=False)

async_session = sessionmaker(
    engine,
    expire_on_commit=False,
    class_=AsyncSession,
)

async def get_session() -> AsyncSession:
    async with async_session() as session:
        yield session