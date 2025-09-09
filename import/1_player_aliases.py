from pathlib import Path
import sys
import pandas as pd
from sqlalchemy import text

ROOT = Path(__file__).resolve().parents[1]   # /.../root
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from db_connect import get_engine

engine = get_engine()

NICKNAMES = [
    ("Nick Watson", "the wizard"),
    ("Bailey Smith", "bazlenka"),
    ("Jai Newcombe", "nuke"),
    ("Jacob Koschitzke", "shitski"),
    ("Patrick Cripps", "crippa"),
    ("Henry Hustwaite", "pencil"),
    ("Mitch Lewis", "loo"),
    ("Changkuoth Jiath", "cj"),
    ("James Sicily", "skipperly"),
    ("Josh Weddle", "joe webble"),
    ("Josh Weddle", "john wembley"),
    ("Cam Mackenzie", "mini mitch"),
    ("Mabior Chol", "journeyman"),
    ("Riley Thilthorpe", "beast"),
    ("Luke Davies-Uniacke", "ldu"),
    ("Jake Stringer", "package"),
    ("Jake Stringer", "parcel"),
    ("Mitch McGovern", "brackets"),
    ("Nick Blakey", "lizard"),
]

DROP_SQL = 'DROP TABLE IF EXISTS player_nicknames;'

CREATE_SQL = '''
CREATE TABLE player_nicknames (
    "Player"   TEXT NOT NULL,
    "Nickname" TEXT NOT NULL,
    PRIMARY KEY ("Player", "Nickname")
);
'''

INSERT_SQL = 'INSERT INTO player_nicknames ("Player","Nickname") VALUES (:player, :nickname);'

def main() -> None:
    with engine.begin() as conn:  # single atomic transaction
        # Replace the table
        conn.exec_driver_sql(DROP_SQL)
        conn.exec_driver_sql(CREATE_SQL)

        # Bulk insert
        conn.execute(
            text(INSERT_SQL),
            [{"player": p, "nickname": n} for p, n in NICKNAMES]
        )

    print(f"Inserted {len(NICKNAMES)} nickname rows into player_nicknames")

if __name__ == "__main__":
    main()
