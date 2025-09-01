from pathlib import Path
import sys
import pandas as pd
from sqlalchemy import text

ROOT = Path(__file__).resolve().parents[1]   # /.../root
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from db_connect import get_engine

CSV_DIR = ROOT / "csv"

def import_csv_to_table(csv_filename: str, table_name: str):
    engine = get_engine()
    csv_path = CSV_DIR / csv_filename
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    df.to_sql(table_name, engine, if_exists="replace", index=False)
    print(f"✅ '{table_name}' created from '{csv_path.relative_to(ROOT)}'")

def main():
    import_csv_to_table("2025_footy_player_stats.csv", "player_stats")
    import_csv_to_table("afl_upcoming_games.csv", "upcoming_games")

if __name__ == "__main__":
    main()
