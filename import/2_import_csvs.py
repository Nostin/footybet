from pathlib import Path
import sys
import pandas as pd
import numpy as np
from sqlalchemy import text as sqtext, Date as SQLDate

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from db_connect import get_engine

CSV_DIR = ROOT / "csv"
engine = get_engine()

def normalize_blanks_to_null(df: pd.DataFrame) -> pd.DataFrame:
    # Convert empty strings / whitespace-only strings to NaN (-> NULL in SQL)
    obj_cols = df.select_dtypes(include=["object"]).columns
    for c in obj_cols:
        df[c] = df[c].astype(str).str.strip()
        df.loc[df[c] == "", c] = np.nan
        # If you had literal "nan" from earlier conversions, clean that too
        df.loc[df[c].str.lower() == "nan", c] = np.nan
    return df

def normalize_nullable_bool(series: pd.Series) -> pd.Series:
    # Accept a bunch of user-friendly inputs; keep nulls as nulls
    if series is None:
        return series
    s = series.copy()
    # Leave NaNs alone
    mask = s.notna()
    vals = s[mask].astype(str).str.strip().str.lower()

    true_set  = {"1", "true", "t", "yes", "y"}
    false_set = {"0", "false", "f", "no", "n"}

    out = pd.Series([pd.NA] * len(s), index=s.index, dtype="boolean")
    out.loc[mask & vals.isin(true_set)] = True
    out.loc[mask & vals.isin(false_set)] = False
    # Anything else stays NULL (forces you to be explicit rather than guessing)
    return out

def import_csv_to_table(csv_filename: str, table_name: str):
    csv_path = CSV_DIR / csv_filename
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path, keep_default_na=True, na_values=["", " "])
    df = normalize_blanks_to_null(df)

    # ---- Date normalization (strict YYYY-MM-DD) ----
    dtype_map = {}
    if "Date" in df.columns:
        raw = df["Date"].astype("string").str.strip()

        bad = ~raw.str.match(r"^\d{4}-\d{2}-\d{2}$", na=False)
        if bad.any():
            print(f"Bad Date values in {table_name} (first 30):")
            print(raw[bad].head(30).tolist())
            raise ValueError(f"{table_name}: Date must be YYYY-MM-DD")

        df["Date"] = pd.to_datetime(raw, format="%Y-%m-%d", errors="raise").dt.date
        dtype_map["Date"] = SQLDate()

    # ---- Table-specific cleanup ----
    if table_name == "tips":
        for col in ["Tip Margin", "Actual Margin"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        if "Correct" in df.columns:
            c = df["Correct"].astype("string").str.strip().str.lower()
            df["Correct"] = pd.Series([pd.NA] * len(df), index=df.index, dtype="string")
            df.loc[c.isin(["yes", "y", "true", "t", "1"]), "Correct"] = "Yes"
            df.loc[c.isin(["no", "n", "false", "f", "0"]), "Correct"] = "No"

        for col in ["Venue", "Home Team", "Away Team", "Tip", "Tip Confidence", "Timeslot", "Round"]:
            if col in df.columns:
                df[col] = df[col].astype("string").str.strip().replace({"": pd.NA})

    # ---- Write once (important) ----
    df.to_sql(
        table_name,
        engine,
        if_exists="replace",
        index=False,
        dtype=dtype_map if dtype_map else None,
    )

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
        schema_row = con.execute(sqtext("SELECT current_schema()")).fetchone()
        schema = schema_row[0] if schema_row else "public"

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

        print(f"🔎 Current schema: {schema}")
        for name in sorted(targets):
            fq = f'{schema}."{name}"'
            if name in existing_views:
                con.execute(sqtext(f"DROP VIEW IF EXISTS {fq} CASCADE;"))
                print(f"🗑️  Dropped VIEW {fq}")
            elif name in existing_matviews:
                con.execute(sqtext(f"DROP MATERIALIZED VIEW IF EXISTS {fq} CASCADE;"))
                print(f"🗑️  Dropped MATERIALIZED VIEW {fq}")
            else:
                print(f"ℹ️  Not found: {fq}")


def refresh_upcoming_games_with_tips_view():
    ddl = """
    CREATE VIEW upcoming_games_with_tips AS
    SELECT
      ug.*,
      t."Tip"            AS "Tip",
      t."Tip Confidence" AS "Tip Confidence",
      t."Tip Margin"     AS "Tip Margin",
      t."Correct"        AS "Correct",
      t."Actual Margin"  AS "Actual Margin"
    FROM upcoming_games ug
    LEFT JOIN tips t
      ON ug."Date" = t."Date"
     AND ug."Venue" = t."Venue"
     AND ug."Home Team" = t."Home Team"
     AND ug."Away Team" = t."Away Team";
    """
    with engine.begin() as con:
        con.execute(sqtext(ddl))
    print("✅ View 'upcoming_games_with_tips' refreshed.")

def drop_dependent_views():
    with engine.begin() as con:
        con.execute(sqtext('DROP VIEW IF EXISTS upcoming_games_with_tips CASCADE;'))


def main():
    drop_player_views()
    drop_dependent_views()
    import_csv_to_table("2025_footy_player_stats.csv", "player_stats")
    import_csv_to_table("afl_upcoming_games.csv", "upcoming_games")
    import_csv_to_table("afl_tips.csv", "tips")
    refresh_upcoming_games_with_tips_view()

if __name__ == "__main__":
    main()
