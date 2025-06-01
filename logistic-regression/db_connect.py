from sqlalchemy import create_engine

# -------------------------
# DATABASE CONFIGURATION
# -------------------------
DB_NAME = "footy"
DB_USER = "seanthompson"
DB_PASS = ""  # Set if required
DB_HOST = "localhost"
DB_PORT = "5432"

# -------------------------
# DATABASE CONNECTION FUNCTION
# -------------------------
def get_engine():
    """Returns a SQLAlchemy engine for the database connection."""
    return create_engine(f"postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}")
