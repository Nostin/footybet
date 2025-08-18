# 450_score_players_using_enrich.py
import os
import numpy as np
import pandas as pd
from sqlalchemy import text
from db_connect import get_engine
import joblib

# === IMPORTANT: import the EXACT enrich() you used for training ===
# Adjust this import to wherever your training enrich() lives.
# It must be byte-for-byte equivalent to the training version.
from enrich_fns import enrich_disposals  # <-- EDIT THIS

engine = get_engine()

TARGETS = [20, 25, 30]
CALIB_FILES = {
    20: dict(lr="models/logreg_calibrated_t20.pkl",
             lgbm="models/lgbm_calibrated_t20.pkl",
             scaler="models/logreg_scaler_t20.pkl",
             alpha="models/blend_alpha_t20.pkl"),
    25: dict(lr="models/logreg_calibrated_t25.pkl",
             lgbm="models/lgbm_calibrated_t25.pkl",
             scaler="models/logreg_scaler_t25.pkl",
             alpha="models/blend_alpha_t25.pkl"),
    30: dict(lr="models/logreg_calibrated_t30.pkl",
             lgbm="models/lgbm_calibrated_t30.pkl",
             scaler="models/logreg_scaler_t30.pkl",
             alpha="models/blend_alpha_t30.pkl"),
}

# --- feature list & binning must match training ---
def bin_days(val):
    if pd.isna(val): return np.nan
    if val <= 5: return '≤5 days'
    if val == 6: return '6 days'
    if val == 7: return '7 days'
    if val == 8: return '8 days'
    if val == 9: return '9 days'
    if val == 10: return '10 days'
    if val == 11: return '11 days'
    if val in [12,13]: return '12-13 days'
    if val in [14,15]: return '14-15 days'
    return '16+ days'

exact_wishlist = [
    'disposals_trend_last_5','disposals_delta_5',
    'disposals_max_last_5','disposals_min_last_5','disposals_std_last_5',
    'is_home_game','avg_team_disposals_last_4','is_wet_game',
    'wet_disposals_last_3','dry_disposals_last_3',
    'form_minus_season_med_last_3','season_to_date_median',
    'opp_concessions_last_5',
]
prefix_wishlist = [
    'disp_cap_avg_last_','disp_cap_med_last_','disp_cap_max_last_','disp_cap_min_last_','disp_cap_var_last_',
    'disp_cap_wet_avg_last_','disp_cap_dry_avg_last_','disp_cap_home_avg_last_','disp_cap_away_avg_last_',
    'disp_cap_day_avg_last_','disp_cap_night_avg_last_','disp_cap_twilight_avg_last_',
    'disp_floor_score_last_',
]

def build_feature_list(df_train):
    bin_features = [c for c in df_train.columns if c.startswith("days_bin_")]
    def existing(cols): return [c for c in cols if c in df_train.columns]
    def with_prefix(prefixes):
        return [c for c in df_train.columns if any(c.startswith(p) for p in prefixes)]
    feats = sorted(set(existing(exact_wishlist) + with_prefix(prefix_wishlist) + bin_features))
    if not feats:
        raise RuntimeError("No available features found. Check training feed.")
    return feats

def load_models_for_target(tgt):
    files = CALIB_FILES[tgt]
    lr = joblib.load(files["lr"])
    lgbm = joblib.load(files["lgbm"])
    scaler = joblib.load(files["scaler"])
    alpha = joblib.load(files["alpha"])["alpha"]
    return lr, lgbm, scaler, float(alpha)

def score_one_player(player_row, train_means, feature_cols, lr, lgbm, scaler, alpha):
    """
    player_row: row from player_precomputes (needs Player, Team, Next_* fields).
    Returns dict of probs for each target (keys '20','25','30').
    """
    player = player_row["Player"]
    team   = player_row.get("Team", None)
    next_venue    = player_row.get("Next_Venue", "")
    next_timeslot = player_row.get("Next_Timeslot", "")
    next_opponent = player_row.get("Next_Opponent", "")

    # Pull this player's full history
    hist = pd.read_sql(
        'SELECT * FROM player_stats WHERE "Player" = %(p)s ORDER BY "Date"',
        engine, params={"p": player}
    )
    if hist.empty:
        return None  # no history -> skip

    # Append a one-row "next game" stub so enrich() computes pre-game features for it
    # Default to Dry unless you have a signal (you can override per player if needed).
    # Use last known Team if not present in precomputes.
    team_for_stub = team if team else hist["Team"].dropna().iloc[-1]
    next_date = None
    # try parse a date from player_precomputes via upcoming_games join you already did
    # If you also store the date there, use it. Otherwise, we can’t infer date -> still OK;
    # enrich() features depend on past games; days_since_last_game will be NaN though.
    # Here we try to infer it from upcoming_games by team:
    try:
        ug = pd.read_sql(
            'SELECT * FROM upcoming_games WHERE "Home Team" = %(t)s OR "Away Team" = %(t)s',
            engine, params={"t": team_for_stub}
        )
        ug["Date"] = pd.to_datetime(ug["Date"], errors="coerce")
        if not ug.empty:
            # take earliest after last game
            last_date = pd.to_datetime(hist["Date"]).max()
            ug = ug[ug["Date"] > last_date].sort_values("Date")
            if not ug.empty:
                next_date = ug.iloc[0]["Date"]
    except Exception:
        pass

    stub = {
        "Date": next_date if next_date is not None else pd.to_datetime(hist["Date"]).max() + pd.Timedelta(days=7),
        "Team": team_for_stub,
        "Player": player,
        "Opponent": next_opponent if isinstance(next_opponent, str) else None,
        "Venue": next_venue if isinstance(next_venue, str) else None,
        "Timeslot": next_timeslot if isinstance(next_timeslot, str) else None,
        "Conditions": "Dry",  # default; change to "Wet" if you have a weather signal
        # Any other columns present in player_stats but missing here will be NaN
    }
    hist_plus = pd.concat([hist, pd.DataFrame([stub])], ignore_index=True)

    # Re-run the SAME enrich() as training
    enriched = enrich_disposals(hist_plus)

    # Create day bin dummies exactly like training
    enriched["days_bin"] = enriched["days_since_last_game"].apply(bin_days)
    enriched = pd.get_dummies(enriched, columns=["days_bin"], drop_first=True)

    # Build feature list from the TRAIN table (ensures exact alignment)
    df_train = pd.read_sql("SELECT * FROM model_feed_train LIMIT 5", engine)
    df_train["days_bin"] = df_train["days_since_last_game"].apply(bin_days)
    df_train = pd.get_dummies(df_train, columns=["days_bin"], drop_first=True)
    feature_cols_train = build_feature_list(df_train)

    # Ensure prediction frame has all feature columns
    for c in feature_cols_train:
        if c not in enriched.columns:
            enriched[c] = 0.0

    # Take the last row (our stub)
    x = enriched.iloc[[-1]][feature_cols_train].astype(float).copy()

    # Impute NaNs with train means (per training convention)
    x = x.fillna(train_means.reindex(feature_cols_train))

    # Scale for LR, predict for both models, blend
    x_lr = scaler.transform(x.values)
    p_lr = lr.predict_proba(x_lr)[:, 1]
    p_lgbm = lgbm.predict_proba(x.values)[:, 1]
    p_blend = alpha * p_lr + (1 - alpha) * p_lgbm

    return float(p_blend[0])

def main():
    # Load train table once to compute train means for imputation (matching training)
    df_train = pd.read_sql("SELECT * FROM model_feed_train", engine)
    df_train["days_bin"] = df_train["days_since_last_game"].apply(bin_days)
    df_train = pd.get_dummies(df_train, columns=["days_bin"], drop_first=True)
    feature_cols = build_feature_list(df_train)
    train_means = df_train[feature_cols].mean(numeric_only=True)

    # Load models per target
    models = {}
    for tgt in TARGETS:
        models[tgt] = load_models_for_target(tgt)

    # Read precomputes and iterate
    pre = pd.read_sql('SELECT * FROM player_precomputes', engine)

    # We’ll update in chunks to avoid locking issues
    updates = []
    for _, row in pre.iterrows():
        player = row["Player"]
        out = {}
        for tgt in TARGETS:
            lr, lgbm, scaler, alpha = models[tgt]
            try:
                prob = score_one_player(row, train_means, feature_cols, lr, lgbm, scaler, alpha)
            except Exception as e:
                # If something goes wrong for this player/target, skip gracefully
                prob = np.nan
            out[tgt] = prob

        updates.append({
            "Player": player,
            "Prob_20_Disposals": out[20],
            "Prob_25_Disposals": out[25],
            "Prob_30_Disposals": out[30],
        })

    upd_df = pd.DataFrame(updates)

    # Merge back to preserve all existing data
    pre2 = pre.merge(upd_df, on="Player", how="left", suffixes=("", "_new"))
    for col in ["Prob_20_Disposals", "Prob_25_Disposals", "Prob_30_Disposals"]:
        if f"{col}_new" in pre2.columns:
            pre2[col] = pre2[f"{col}_new"]
            pre2.drop(columns=[f"{col}_new"], inplace=True)

    pre2.to_sql("player_precomputes", engine, if_exists="replace", index=False)
    with engine.connect() as conn:
        conn.execute(text('CREATE INDEX IF NOT EXISTS idx_player_precomputes_player ON player_precomputes ("Player")'))
        conn.execute(text('CREATE INDEX IF NOT EXISTS idx_player_precomputes_team ON player_precomputes ("Team")'))
    print("✅ player_precomputes updated with Prob_20/25/30_Disposals")

if __name__ == "__main__":
    main()
