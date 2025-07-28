import pandas as pd
from sqlalchemy import text
from db_connect import get_engine

engine = get_engine()

with engine.begin() as conn:
    # Training data
    conn.execute(text("""
        DROP TABLE IF EXISTS player_stats_train;
        CREATE TABLE player_stats_train AS
        SELECT * FROM player_stats WHERE "Date" < '2025-07-01';
    """))

    # Raw test set
    conn.execute(text("""
        DROP TABLE IF EXISTS player_stats_test_raw;
        CREATE TABLE player_stats_test_raw AS
        SELECT * FROM player_stats WHERE "Date" >= '2025-05-01';
    """))

# Only use players with ≥7 games and ≥18 avg disposals
eligible_players_query = """
    SELECT "Team", "Player"
    FROM player_stats_train
    GROUP BY "Player", "Team"
    HAVING COUNT(*) >= 7 AND AVG("Disposals") >= 18;
"""
eligible_players = pd.read_sql(eligible_players_query, engine)

# Filter test data to eligible players
test_raw = pd.read_sql("SELECT * FROM player_stats_test_raw", engine)
test_filtered = test_raw.merge(eligible_players, on=["Team", "Player"], how="inner")

# Save targets separately for evaluation
target_cols = ['Player', 'Date', 'Disposals', 'Game Result']
test_filtered[target_cols].to_sql('player_stats_test_targets', engine, if_exists='replace', index=False)

# Save full raw test input (with Disposals intact)
test_filtered.to_sql('player_stats_test', engine, if_exists='replace', index=False)

print("✅ Train and test tables created (including raw test targets)")
