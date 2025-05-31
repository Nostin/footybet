import pandas as pd
import joblib
import numpy as np
from db_connect import get_engine
from model_config import categorical_cols, numerical_cols

# Load the model and trained columns (no encoder needed if using get_dummies)
calibrated_model = joblib.load("calibrated_model.joblib")
trained_columns = joblib.load("trained_columns.joblib")

# Connect to database
engine = get_engine()

# Get the last game BEFORE the Suns match
query = """
    SELECT *
    FROM model_feed
    WHERE "Player" = 'Caleb Serong'
      AND "Date" < '2025-05-31'
    ORDER BY "Date" DESC
    LIMIT 1
"""
latest_row = pd.read_sql(query, engine)

# Inject context for the *next* match (vs Suns)
latest_row["Opponent"] = "Suns"
latest_row["timeslot_category"] = "day"
latest_row["Team Win Odds"] = 2.94
latest_row["is_away_game"] = 1  # Suns game is away

# Drop outcome variables and date
input_data = latest_row.drop(columns=["Date", "target_20", "target_25", "target_30"])

# Split into categorical and numerical
X_num = input_data[numerical_cols].astype(float)
X_cat = pd.get_dummies(input_data[categorical_cols], drop_first=True)

# Combine them
X_combined_df = pd.concat([X_num, X_cat], axis=1)

# Align to training columns
for col in trained_columns:
    if col not in X_combined_df.columns:
        X_combined_df[col] = 0

X_combined_df = X_combined_df[trained_columns]
X_encoded = X_combined_df.astype(float)

# Predict
prob = calibrated_model.predict_proba(X_encoded)[0][1]
print(f"🔮 Probability Caleb Serong gets 25+ disposals: {prob:.2%}")
