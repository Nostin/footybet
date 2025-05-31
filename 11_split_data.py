import pandas as pd
from sqlalchemy import create_engine, text
from db_connect import get_engine

# Connect to the database
engine = get_engine()

# Step 1: Create training and test tables
with engine.begin() as conn:
    conn.execute(text("""
        DROP TABLE IF EXISTS player_stats_train;
        CREATE TABLE player_stats_train AS
        SELECT * FROM player_stats WHERE "Date" < '2025-05-20';
    """))

    conn.execute(text("""
        DROP TABLE IF EXISTS player_stats_test;
        CREATE TABLE player_stats_test AS
        SELECT * FROM player_stats WHERE "Date" >= '2025-05-20';
    """))

# Step 2: Get eligible players
eligible_players_query = """
    SELECT "Team", "Player"
    FROM player_stats_train
    GROUP BY "Player", "Team"
    HAVING COUNT(*) >= 7 AND AVG("Disposals") >= 18;
"""
eligible_players = pd.read_sql(eligible_players_query, engine)

# Step 3: Filter test set to only include eligible players
test_data = pd.read_sql('SELECT * FROM player_stats_test', engine)
filtered_test_data = test_data.merge(eligible_players, on=["Team", "Player"], how="inner")

# Step 4: Overwrite player_stats_test with filtered data
filtered_test_data.to_sql('player_stats_test', engine, if_exists='replace', index=False)

print("✅ Test table filtered and overwritten with eligible players only.")
