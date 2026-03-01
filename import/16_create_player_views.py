from __future__ import annotations
from pathlib import Path
import sys
from sqlalchemy import text as sqtext

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from db_connect import get_engine
engine = get_engine()

BASE_CANDIDATES = [
    "Player","Team","Next_Opponent","Next_Venue","Next_Timeslot","Days_since_last_game"
]

def _existing_columns(table: str) -> list[str]:
    with engine.begin() as con:
        rows = con.execute(sqtext("""
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name = :t
            ORDER BY ordinal_position
        """), {"t": table}).fetchall()
    return [r[0] for r in rows]

def ensure_metric_view(
    metric_prefix: str,
    *,
    view_name: str,
    prob_suffix: str | None = None,
    extra_prefixes: list[str] | None = None,
):
    """
    Create/replace a thin view over player_precomputes with only:
      - base columns (if present)
      - columns starting with `metric_prefix + '_'` (regex-anchored)
      - optional probability columns: ^Prob_.*_{prob_suffix}$
      - optional extra prefixes (e.g. 'Goals_per100ToG')
    """
    table = "player_precomputes"
    existing = set(_existing_columns(table))
    base = [c for c in BASE_CANDIDATES if c in existing]

    prefixes = [metric_prefix] + (extra_prefixes or [])
    # Build regex conditions, e.g. column_name ~* '^Disposal_' OR column_name ~* '^Goals_per100ToG_'
    prefix_regex_clauses = [f"column_name ~* '^{p}_'" for p in prefixes]

    prob_clause = []
    if prob_suffix:
        # e.g. Prob_20_Disposals, Prob_25_Disposals, …
        prob_clause.append(f"column_name ~* '^Prob_.*_{prob_suffix}$'")

    where_sql = " OR ".join(prefix_regex_clauses + prob_clause)

    with engine.begin() as con:
        cols = [r[0] for r in con.execute(sqtext(f"""
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name = '{table}'
              AND ({where_sql})
            ORDER BY ordinal_position
        """)).fetchall()]

        if not cols:
            print(f"⚠️  No columns found for {view_name} (prefix={metric_prefix}). Skipping.")
            return

        sel = base + cols
        col_list = ", ".join(f'"{c}"' for c in sel)

        con.execute(sqtext(f'DROP VIEW IF EXISTS {view_name} CASCADE;'))
        con.execute(sqtext(f'CREATE VIEW {view_name} AS SELECT {col_list} FROM "{table}";'))
        print(f"✅ Created view {view_name} with {len(sel)} columns")

def main():
    ensure_metric_view(
        "Disposal",
        view_name="player_precomputes_disposals",
        prob_suffix="Disposals",
    )
    ensure_metric_view(
        "Goal",
        view_name="player_precomputes_goals",
        prob_suffix="Goals",
        extra_prefixes=["Goals_per100ToG"],  # pulls Goals_per100ToG_Season/_N
    )
    ensure_metric_view("Mark",      view_name="player_precomputes_marks")
    ensure_metric_view("Tackle",    view_name="player_precomputes_tackles")
    ensure_metric_view("Clearance", view_name="player_precomputes_clearances")
    ensure_metric_view("Kick",      view_name="player_precomputes_kicks",     extra_prefixes=["Kicks_per100ToG"])
    ensure_metric_view("Handball",  view_name="player_precomputes_handballs")  # add extra_prefixes if you create rates

if __name__ == "__main__":
    main()
