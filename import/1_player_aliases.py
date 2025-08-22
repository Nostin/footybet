# create_or_replace_player_nicknames.py
from sqlalchemy import text
from db_connect import get_engine  # your project helper

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
