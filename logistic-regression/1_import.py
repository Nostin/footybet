import pandas as pd
from sqlalchemy import text
from db_connect import get_engine

def import_csv_to_table(csv_filename, table_name):
    engine = get_engine()

    with engine.connect() as conn:
        conn.execute(text(f"DROP TABLE IF EXISTS {table_name}"))
        conn.commit()

    df = pd.read_csv(csv_filename)
    df.to_sql(table_name, engine, if_exists='replace', index=False)
    print(f"✅ Table '{table_name}' created and data from '{csv_filename}' imported successfully!")

def main():
    import_csv_to_table('2025_footy_player_stats.csv', 'player_stats')
    import_csv_to_table('afl_upcoming_games.csv', 'upcoming_games')

if __name__ == "__main__":
    main()
