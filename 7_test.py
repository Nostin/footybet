import pandas as pd
import joblib
import numpy as np
from db_connect import get_engine
from model_config import categorical_cols, numerical_cols

# Load model artifacts
calibrated_model = joblib.load("calibrated_model.joblib")
encoder = joblib.load("encoder.joblib")
trained_columns = joblib.load("trained_columns.joblib")

# Connect to DB
engine = get_engine()

# === Define test set ===
test_players = [
    {"Player": "Andrew Brayshaw", "Opponent": "Suns", "Venue": "Gold Coast", "Timeslot": "day", "Team Win Odds": 2.94, "Expected": 1.26},
    {"Player": "Noah Anderson", "Opponent": "Fremantle", "Venue": "Gold Coast", "Timeslot": "day", "Team Win Odds": 1.38, "Expected": 1.45},
    {"Player": "Touk Miller", "Opponent": "Fremantle", "Venue": "Gold Coast", "Timeslot": "day", "Team Win Odds": 1.38, "Expected": 2.00},
    {"Player": "John Noble", "Opponent": "Fremantle", "Venue": "Gold Coast", "Timeslot": "day", "Team Win Odds": 1.38, "Expected": 1.91},
    {"Player": "Matt Rowell", "Opponent": "Fremantle", "Venue": "Gold Coast", "Timeslot": "day", "Team Win Odds": 1.38, "Expected": 2.60},
    {"Player": "Daniel Rioli", "Opponent": "Fremantle", "Venue": "Gold Coast", "Timeslot": "day", "Team Win Odds": 1.38, "Expected": 3.60},
    {"Player": "Tom Green", "Opponent": "Richmond", "Venue": "Engie", "Timeslot": "day", "Team Win Odds": 1.06, "Expected": 1.12},
    {"Player": "Lachie Whitfield", "Opponent": "Richmond", "Venue": "Engie", "Timeslot": "day", "Team Win Odds": 1.06, "Expected": 1.16},
    {"Player": "Lachie Ash", "Opponent": "Richmond", "Venue": "Engie", "Timeslot": "day", "Team Win Odds": 1.06, "Expected": 1.28},
    {"Player": "Tim Taranto", "Opponent": "GWS", "Venue": "Engie", "Timeslot": "day", "Team Win Odds": 9.50, "Expected": 1.56},
    {"Player": "Jacob Hopper", "Opponent": "GWS", "Venue": "Engie", "Timeslot": "day", "Team Win Odds": 9.50, "Expected": 1.89},
]

# Prepare results
results = []

for player in test_players:
    name = player["Player"]
    query = f"""
        SELECT * FROM model_feed
        WHERE "Player" = '{name}'
          AND "Date" < '2025-05-31'
        ORDER BY "Date" DESC
        LIMIT 1
    """
    try:
        row = pd.read_sql(query, engine)
        if row.empty:
            results.append({"Player": name, "Error": "No data"})
            continue

        row["Opponent"] = player["Opponent"]
        row["timeslot_category"] = player["Timeslot"]
        row["Team Win Odds"] = player["Team Win Odds"]
        row["is_away_game"] = int(row["Venue"].iloc[0] != player["Venue"])

        # Drop columns
        input_data = row.drop(columns=["Date", "target_20", "target_25", "target_30"])

        # Encode
        X_num = input_data[numerical_cols].astype(float)
        X_cat = pd.get_dummies(input_data[categorical_cols], drop_first=True)
        X_combined_df = pd.concat([X_num, X_cat], axis=1)

        # Align to training cols
        for col in trained_columns:
            if col not in X_combined_df.columns:
                X_combined_df[col] = 0
        X_combined_df = X_combined_df[trained_columns]

        # Predict
        X_encoded = X_combined_df.astype(float)
        model_prob = calibrated_model.predict_proba(X_encoded)[0][1]

        # Implied probability
        decimal_odds = player["Expected"]
        implied_prob = 1 / decimal_odds

        results.append({
            "Player": name,
            "Model Prob": round(model_prob * 100, 2),
            "Bookie Prob": round(implied_prob * 100, 2),
            "Bookie Odds": decimal_odds
        })
    except Exception as e:
        results.append({"Player": name, "Error": str(e)})

# Output
results_df = pd.DataFrame(results)
print("\n🎯 Prediction Comparison (Model vs Bookie Implied %):\n")
print(results_df.to_string(index=False))
