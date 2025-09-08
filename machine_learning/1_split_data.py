from pathlib import Path
import sys
import pandas as pd
from sqlalchemy import text

ROOT = Path(__file__).resolve().parents[1]   # /.../root
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from db_connect import get_engine

engine = get_engine()

CUTOFF_DATE = '2025-07-29'

with engine.begin() as conn:
    # Training data
    conn.execute(text(f"""
        DROP TABLE IF EXISTS player_stats_train;
        CREATE TABLE player_stats_train AS
        SELECT * FROM player_stats WHERE "Date" < '{CUTOFF_DATE}';
    """))

    # Raw test set
    conn.execute(text(f"""
        DROP TABLE IF EXISTS player_stats_test;
        CREATE TABLE player_stats_test AS
        SELECT * FROM player_stats WHERE "Date" >= '{CUTOFF_DATE}';
    """))

# Only use players with ≥7 games and ≥18 avg disposals
eligible_players_query = """
    SELECT "Team", "Player"
    FROM player_stats_train
    GROUP BY "Player", "Team"
    HAVING COUNT(*) >= 7;
"""
eligible_players = pd.read_sql(eligible_players_query, engine)

# Filter test data to eligible players
test_raw = pd.read_sql("SELECT * FROM player_stats_test", engine)
test_filtered = test_raw.merge(eligible_players, on=["Team", "Player"], how="inner")

# Save targets separately for evaluation
target_cols = ['Player', 'Date', 'Disposals', 'Game Result']
test_filtered[target_cols].to_sql('player_stats_test_targets', engine, if_exists='replace', index=False)

# Save full raw test input (with Disposals intact)
test_filtered.to_sql('player_stats_test', engine, if_exists='replace', index=False)

print("✅ Train and test tables created (including raw test targets)")
