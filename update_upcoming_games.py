from pathlib import Path
import sys
import pandas as pd
from sqlalchemy import text as sqtext

ROOT = Path(__file__).resolve().parents[0]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from db_connect import get_engine

CSV_DIR = ROOT / "csv"
engine = get_engine()

def import_csv_to_table(csv_filename: str, table_name: str):
    csv_path = CSV_DIR / csv_filename
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)

    # Normalise date columns up-front
    if table_name == "player_stats" and "Date" in df.columns:
        # These are already YYYY-MM-DD, but this guarantees dtype=date
        df["Date"] = pd.to_datetime(df["Date"], errors="raise").dt.date

    if table_name == "upcoming_games" and "Date" in df.columns:
        # Be tolerant of both DD/MM/YYYY and YYYY-MM-DD
        # dayfirst=True handles e.g. 31/07/2025; ISO still parses fine
        df["Date"] = pd.to_datetime(df["Date"], errors="raise", dayfirst=True).dt.date

    df.to_sql(table_name, engine, if_exists="replace", index=False)
    print(f"✅ '{table_name}' created from '{csv_path.relative_to(ROOT)}'")

def main():
    import_csv_to_table("afl_upcoming_games.csv", "upcoming_games")

if __name__ == "__main__":
    main()
