from pathlib import Path
import sys
import pandas as pd
from sqlalchemy import text as sqtext

ROOT = Path(__file__).resolve().parents[1]
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

def drop_player_views():
    targets = {
        "player_precomputes_disposals",
        "player_precomputes_goals",
        "player_precomputes_marks",
        "player_precomputes_tackles",
        "player_precomputes_clearances",
        "player_precomputes_kicks",
        "player_precomputes_handballs",
    }

    with engine.begin() as con:
        # What schema are we in?
        schema_row = con.execute(sqtext("SELECT current_schema()")).fetchone()
        schema = schema_row[0] if schema_row else "public"

        # Discover existing normal views & matviews
        existing_views = {
            r[0] for r in con.execute(
                sqtext("""
                    SELECT viewname
                    FROM pg_views
                    WHERE schemaname = :schema
                """),
                {"schema": schema},
            ).fetchall()
        }
        existing_matviews = {
            r[0] for r in con.execute(
                sqtext("""
                    SELECT matviewname
                    FROM pg_matviews
                    WHERE schemaname = :schema
                """),
                {"schema": schema},
            ).fetchall()
        }

        # If you prefer to nuke *all* player_precomputes_* views, uncomment:
        # existing_starts = {
        #     *[v for v in existing_views if v.startswith("player_precomputes_")],
        #     *[v for v in existing_matviews if v.startswith("player_precomputes_")],
        # }
        # targets |= existing_starts

        print(f"🔎 Current schema: {schema}")
        for name in sorted(targets):
            fq = f'{schema}."{name}"'  # quoting is safe for any casing
            if name in existing_views:
                con.execute(sqtext(f"DROP VIEW IF EXISTS {fq} CASCADE;"))
                print(f"🗑️  Dropped VIEW {fq}")
            elif name in existing_matviews:
                con.execute(sqtext(f"DROP MATERIALIZED VIEW IF EXISTS {fq} CASCADE;"))
                print(f"🗑️  Dropped MATERIALIZED VIEW {fq}")
            else:
                print(f"ℹ️  Not found: {fq}")

def main():
    drop_player_views()
    import_csv_to_table("2025_footy_player_stats.csv", "player_stats")
    import_csv_to_table("afl_upcoming_games.csv", "upcoming_games")

if __name__ == "__main__":
    main()
