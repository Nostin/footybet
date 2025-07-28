import pandas as pd
import numpy as np
from db_connect import get_engine

engine = get_engine()

# Load your data
df = pd.read_sql("SELECT * FROM model_feed_train", engine)

# Only keep rows with known targets
df = df.dropna(subset=["target_25"])

# Define bins and labels
bins = [0, 5, 6, 7, 8, 9, 10, 11, 13, 15, np.inf]
labels = [
    "≤5 days", "6 days", "7 days", "8 days", "9 days",
    "10 days", "11 days", "12–13 days", "14–15 days", "16+ days"
]

# Create binned column
df["days_bin"] = pd.cut(df["days_since_last_game"], bins=bins, labels=labels, right=True)

# Group by bin and calculate hit rate (mean of target_25)
summary = df.groupby("days_bin").agg(
    games_played=("target_25", "count"),
    hit_rate=("target_25", "mean")
).reset_index()

# Format hit rate as percentage
summary["hit_rate_pct"] = (summary["hit_rate"] * 100).round(1).astype(str) + "%"

print(summary.to_string(index=False))
