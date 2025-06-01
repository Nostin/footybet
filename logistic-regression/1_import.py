import pandas as pd
from sqlalchemy import text
from db_connect import get_engine

def create_player_stats_table():
    # Get database engine
    engine = get_engine()
    
    # Drop table if exists
    with engine.connect() as conn:
        conn.execute(text("DROP TABLE IF EXISTS player_stats"))
        conn.commit()
    
    # Read CSV file
    df = pd.read_csv('2025_footy_player_stats.csv')
    
    # Create table and import data
    df.to_sql('player_stats', engine, if_exists='replace', index=False)
    
    print("Table 'player_stats' created and data imported successfully!")

if __name__ == "__main__":
    create_player_stats_table()