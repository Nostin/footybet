import os
import joblib
from pathlib import Path
import sys
import pandas as pd
import numpy as np
from sqlalchemy import text

ROOT = Path(__file__).resolve().parents[1]   # /.../root
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from db_connect import get_engine
from util import HOME_GROUNDS

engine = get_engine()

# ===================== Config =====================
TARGETS = [20, 25, 30]
MODELS_DIR = (ROOT / "machine_learning" / "models").resolve()

# ===================== Helpers =====================
def to_python_scalar(v):
    if v is None:
        return None
    try:
        if pd.isna(v):
            return None
    except Exception:
        pass
    if isinstance(v, (np.floating, np.float32, np.float64)):
        return float(v)
    if isinstance(v, (np.integer, np.int32, np.int64)):
        return int(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    return v

def is_home_game(row):
    team = row.get('Team')
    venue = row.get('Venue')
    return pd.notna(venue) and venue in HOME_GROUNDS.get(team, [])

def bin_days(val):
    if pd.isna(val): return np.nan
    val = int(val)
    if val <= 5: return '≤5 days'
    if val == 6: return '6 days'
    if val == 7: return '7 days'
    if val == 8: return '8 days'
    if val == 9: return '9 days'
    if val == 10: return '10 days'
    if val == 11: return '11 days'
    if val in [12, 13]: return '12–13 days'
    if val in [14, 15]: return '14–15 days'
    return '16+ days'

def enrich(df: pd.DataFrame) -> pd.DataFrame:
    """Training-identical player feature engineering (team/opp windows added later)."""
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.sort_values(by=['Player', 'Date'])

    # --- Helpers ---
    def capped_arr(x): return np.minimum(x, 35)
    def filtered_roll_mean(g, mask_col, w):
        vals = g['disp_capped'].where(g[mask_col])
        return vals.shift(1).rolling(w, min_periods=1).mean()

    # --- Base rolling windows (capped disposals) ---
    df['disp_capped'] = capped_arr(df['Disposals'].values)
    grouped_p = df.groupby('Player', sort=False)
    for w in [3, 6, 10]:
        df[f'disp_cap_avg_last_{w}'] = grouped_p.apply(
            lambda g: g['disp_capped'].shift(1).rolling(w, min_periods=1).mean(),
            include_groups=False
        ).reset_index(level=0, drop=True)
        df[f'disp_cap_med_last_{w}'] = grouped_p.apply(
            lambda g: g['disp_capped'].shift(1).rolling(w, min_periods=1).median(),
            include_groups=False
        ).reset_index(level=0, drop=True)
        df[f'disp_cap_max_last_{w}'] = grouped_p.apply(
            lambda g: g['disp_capped'].shift(1).rolling(w, min_periods=1).max(),
            include_groups=False
        ).reset_index(level=0, drop=True)
        df[f'disp_cap_min_last_{w}'] = grouped_p.apply(
            lambda g: g['disp_capped'].shift(1).rolling(w, min_periods=1).min(),
            include_groups=False
        ).reset_index(level=0, drop=True)
        df[f'disp_cap_var_last_{w}'] = grouped_p.apply(
            lambda g: g['disp_capped'].shift(1).rolling(w, min_periods=2).var(ddof=0),
            include_groups=False
        ).reset_index(level=0, drop=True)

    # --- Trend / spread / rest ---
    df['disposals_trend_last_5'] = grouped_p['Disposals'].transform(
        lambda x: x.shift(1).rolling(5, min_periods=2).apply(
            lambda y: np.polyfit(range(len(y)), y, 1)[0], raw=True
        )
    )
    df['disposals_delta_5'] = grouped_p['Disposals'].transform(lambda x: x.shift(1) - x.shift(6))
    df['disposals_max_last_5'] = grouped_p['Disposals'].transform(lambda x: x.shift(1).rolling(5).max())
    df['disposals_min_last_5'] = grouped_p['Disposals'].transform(lambda x: x.shift(1).rolling(5).min())
    df['disposals_std_last_5'] = grouped_p['Disposals'].transform(lambda x: x.shift(1).rolling(5).std())
    df['days_since_last_game'] = grouped_p['Date'].transform(lambda x: x.diff().dt.days.shift(1))

    # --- Home / Away flags ---
    df['is_home_game'] = df.apply(is_home_game, axis=1)
    df['is_away_game'] = ~df['is_home_game']

    # --- Wet / Dry / Timeslot flags ---
    cond = df['Conditions'].astype(str).str.lower()
    df['is_wet_game'] = (cond == 'wet')
    df['is_dry_game'] = (cond == 'dry')
    df['timeslot_category'] = df['Timeslot'].astype(str).str.lower().map(
        {'day': 'day', 'twilight': 'twilight', 'night': 'night'}
    ).fillna('unknown')

    grouped_p = df.groupby('Player', sort=False)
    for w in [3, 6, 10]:
        for flag, colname in [
            ('is_wet_game', f'disp_cap_wet_avg_last_{w}'),
            ('is_dry_game', f'disp_cap_dry_avg_last_{w}'),
            ('is_home_game', f'disp_cap_home_avg_last_{w}'),
            ('is_away_game', f'disp_cap_away_avg_last_{w}')
        ]:
            df[colname] = grouped_p.apply(
                lambda g: filtered_roll_mean(g, flag, w),
                include_groups=False
            ).reset_index(level=0, drop=True)

        for ts in ['day', 'night', 'twilight']:
            df[f'disp_cap_{ts}_avg_last_{w}'] = grouped_p.apply(
                lambda g: g.assign(ts=(g['timeslot_category'] == ts))
                          .pipe(lambda gg: filtered_roll_mean(gg, 'ts', w)),
                include_groups=False
            ).reset_index(level=0, drop=True)

    # --- Floor score on last N ---
    def floor_score(series):
        if len(series) == 0 or np.all(pd.isna(series)): return np.nan
        med = np.nanmedian(series)
        if not np.isfinite(med) or med <= 0: return np.nan
        drops = med - series
        drops = drops[drops > 0]
        mean_drop = np.nanmean(drops) if len(drops) else 0.0
        return round(1 - (mean_drop / med), 3)

    for w in [3, 6, 10, 22]:
        df[f'disp_floor_score_last_{w}'] = grouped_p['Disposals'].transform(
            lambda x: x.shift(1).rolling(w, min_periods=2).apply(floor_score, raw=False)
        )

    # --- Season-to-date & form ---
    df['season_to_date_median'] = grouped_p['Disposals'].transform(lambda x: x.expanding().median().shift(1))
    df['form_minus_season_med_last_3'] = df['disp_cap_avg_last_3'] - df['season_to_date_median']

    df = df.drop_duplicates(subset=['Player', 'Date'])
    return df

def load_bundle(tgt):
    base = MODELS_DIR
    lr   = joblib.load(base / f"logreg_calibrated_t{tgt}.pkl")
    lgb  = joblib.load(base / f"lgbm_calibrated_t{tgt}.pkl")
    sc   = joblib.load(base / f"logreg_scaler_t{tgt}.pkl")
    meta = joblib.load(base / f"blend_alpha_t{tgt}.pkl")
    alpha = float(meta["alpha"] if isinstance(meta, dict) else meta)
    return lr, lgb, sc, alpha

def compute_train_means(expected_features):
    """Compute means for exactly expected_features using model_feed_train (with days_bin dummies)."""
    df_train = pd.read_sql("SELECT * FROM model_feed_train", engine)
    df_train["days_bin"] = df_train["days_since_last_game"].apply(bin_days)
    df_train = pd.get_dummies(df_train, columns=["days_bin"], drop_first=True)

    missing_in_train = [c for c in expected_features if c not in df_train.columns]
    for c in missing_in_train:
        df_train[c] = 0.0 if c.startswith("days_bin_") else np.nan

    means = df_train[expected_features].mean(numeric_only=True).fillna(0.0)
    return means

def get_team_ctx_latest():
    """
    Fetch latest team context from your team_precompute views/tables:
      - disposals_avg_last_5  -> pace5
      - concede_disposals_avg_last_5 -> concede5
    """
    for view in ("team_precompute_latest_current", "team_precompute_latest"):
        try:
            q = f'''
            SELECT "Team",
                   "disposals_avg_last_5" AS pace5,
                   "concede_disposals_avg_last_5" AS concede5
            FROM {view}
            '''
            df = pd.read_sql(q, engine)
            if len(df):
                return df
        except Exception:
            pass

    try:
        base = pd.read_sql(
            'SELECT "Team","Date","disposals_avg_last_5","concede_disposals_avg_last_5" FROM team_precompute',
            engine
        )
        base["Date"] = pd.to_datetime(base["Date"], errors="coerce")
        base = base.dropna(subset=["Date"]).sort_values(["Team","Date"])
        latest = base.groupby("Team", as_index=False).tail(1).rename(
            columns={"disposals_avg_last_5":"pace5", "concede_disposals_avg_last_5":"concede5"}
        )[["Team","pace5","concede5"]]
        return latest
    except Exception:
        return pd.DataFrame(columns=["Team","pace5","concede5"])

# ===================== Build prediction rows from base table =====================
# 1) History
hist = pd.read_sql('SELECT * FROM player_stats ORDER BY "Date" ASC', engine)
hist['Date'] = pd.to_datetime(hist['Date'], errors='coerce')

# 2) Base (next-game info per player)
base = pd.read_sql(
    'SELECT "Player","Team","Next_Opponent","Next_Venue","Next_Timeslot","Next_Venue_Home","Days_since_last_game" '
    'FROM player_precomputes', engine
)
base["Next_Timeslot"] = base["Next_Timeslot"].astype(str).str.lower().replace({"nan": np.nan})

# Build synthetic future rows using Days_since_last_game
future_rows = []
skipped_no_rest_or_hist = 0

for _, r in base.iterrows():
    player = r["Player"]
    team   = r["Team"]
    if not isinstance(player, str) or not isinstance(team, str) or team.strip() == "":
        continue

    p_hist = hist[hist["Player"] == player].sort_values("Date")
    last_date = p_hist["Date"].max() if not p_hist.empty else pd.NaT
    days = r["Days_since_last_game"]

    if pd.isna(last_date) or pd.isna(days):
        skipped_no_rest_or_hist += 1
        continue

    future_date = last_date + pd.Timedelta(days=int(days))
    future_rows.append({
        "Player": player,
        "Date": future_date,
        "Team": team,
        "Venue": r.get("Next_Venue", np.nan),
        "Timeslot": r.get("Next_Timeslot", np.nan),
        "Conditions": np.nan,
        "Opponent": r.get("Next_Opponent", np.nan),
        "Disposals": np.nan,
        "Goals": np.nan,
        "Behinds": np.nan,
        "Kicks": np.nan,
        "Handballs": np.nan,
        "Game Result": np.nan,
        "Time on Ground %": np.nan,
        "_force_is_home": int(r.get("Next_Venue_Home", 0)) if not pd.isna(r.get("Next_Venue_Home", np.nan)) else 0,
        "_force_days_since": to_python_scalar(days),
    })

future_df = pd.DataFrame(future_rows)
if future_df.empty:
    print("ℹ️ No eligible players with Days_since_last_game & history; nothing to predict. Leaving prior predictions untouched.")
    sys.exit(0)  # or return cleanly if you wrap this in a main()

# 3) Enrich combined (player-level rolling)
combined = pd.concat(
    [hist, future_df.drop(columns=["_force_is_home","_force_days_since"], errors="ignore")],
    ignore_index=True
)
enriched = enrich(combined)

# Match enriched rows for our future predictions
key_cols = ["Player", "Date"]
pred_rows = future_df[key_cols + ["_force_is_home","_force_days_since","Team","Opponent"]].merge(
    enriched,
    on=key_cols,
    how="left",
    suffixes=("", "_enr")  # keep left 'Team'/'Opponent'
)

# In case a previous run already produced _x/_y columns, normalize them:
for col in ["Team", "Opponent"]:
    if f"{col}_x" in pred_rows.columns or f"{col}_y" in pred_rows.columns:
        # prefer left (future_df)
        left = pred_rows.get(f"{col}_x", pred_rows.get(col))
        right = pred_rows.get(f"{col}_y", pred_rows.get(f"{col}_enr"))
        pred_rows[col] = left if left is not None else right
        drop_cols = [c for c in [f"{col}_x", f"{col}_y", f"{col}_enr"] if c in pred_rows.columns]
        if drop_cols:
            pred_rows.drop(columns=drop_cols, inplace=True)

# Override home/away + rest days with base values
pred_rows["is_home_game"] = pred_rows["_force_is_home"].astype(int)
pred_rows["is_away_game"] = (1 - pred_rows["is_home_game"]).astype(int)
pred_rows["days_since_last_game"] = pred_rows["_force_days_since"]

# ===================== Inject team windows from team_precompute =====================
team_ctx = get_team_ctx_latest()
team_to_pace5     = team_ctx.set_index("Team")["pace5"].to_dict()
team_to_concede5  = team_ctx.set_index("Team")["concede5"].to_dict()

pred_rows["avg_team_disposals_last_5"] = pred_rows["Team"].map(team_to_pace5)
pred_rows["opp_concessions_last_5"]    = pred_rows["Opponent"].map(team_to_concede5)

# ===================== BACKFILL PLAYER MEDIANS/FORM (edge cases) =====================
enriched_hist = enrich(hist)

player_last_stm = (
    enriched_hist.sort_values("Date")
    .groupby("Player", as_index=True)["season_to_date_median"]
    .apply(lambda s: s.dropna().iloc[-1] if s.notna().any() else np.nan)
    .to_dict()
)
player_last_form3 = (
    enriched_hist.sort_values("Date")
    .groupby("Player", as_index=True)["form_minus_season_med_last_3"]
    .apply(lambda s: s.dropna().iloc[-1] if s.notna().any() else np.nan)
    .to_dict()
)

pred_rows["season_to_date_median"] = pred_rows["season_to_date_median"].fillna(
    pred_rows["Player"].map(player_last_stm)
)
pred_rows["form_minus_season_med_last_3"] = pred_rows["form_minus_season_med_last_3"].fillna(
    pred_rows["Player"].map(player_last_form3)
)

# Recreate days_bin one-hots like training
pred_rows["days_bin"] = pred_rows["days_since_last_game"].apply(bin_days)
pred_rows = pd.get_dummies(pred_rows, columns=["days_bin"], drop_first=True)

# ===================== Predict per target, aligned to scaler features =====================
pred_out = pred_rows[["Player"]].copy()

for tgt in TARGETS:
    lr, lgb, sc, alpha = load_bundle(tgt)
    if not hasattr(sc, "feature_names_in_"):
        raise RuntimeError(f"Scaler for target {tgt} lacks feature_names_in_. Retrain saving sklearn >=1.0 artifacts.")

    expected_features = list(sc.feature_names_in_)

    # Ensure missing one-hot columns exist
    for c in expected_features:
        if c.startswith("days_bin_") and c not in pred_rows.columns:
            pred_rows[c] = 0.0

    # Build X in exact order; fill NA with train means computed for these exact features
    X_use = pred_rows.reindex(columns=expected_features, fill_value=np.nan).astype(float)
    train_means = compute_train_means(expected_features)
    X_use = X_use.fillna(train_means)

    # Predict
    p_lr   = lr.predict_proba(sc.transform(X_use))[:, 1]
    p_lgbm = lgb.predict_proba(X_use)[:, 1]
    p_bl   = alpha * p_lr + (1 - alpha) * p_lgbm

    # store as 0–100 (%) instead of 0–1
    p_bl_pct = np.clip(p_bl * 100.0, 0.0, 100.0)

    pred_out[f"Prob_{tgt}_Disposals"] = p_bl_pct

# ===================== Persist to player_precomputes =====================
with engine.begin() as conn:
    for c in ["Prob_20_Disposals", "Prob_25_Disposals", "Prob_30_Disposals"]:
        conn.exec_driver_sql(f'ALTER TABLE player_precomputes ADD COLUMN IF NOT EXISTS "{c}" DOUBLE PRECISION')

updated = 0
with engine.begin() as conn:
    for _, r in pred_out.iterrows():
        params = {
            "p20": to_python_scalar(r.get("Prob_20_Disposals")),
            "p25": to_python_scalar(r.get("Prob_25_Disposals")),
            "p30": to_python_scalar(r.get("Prob_30_Disposals")),
            "player": r["Player"],
        }
        sql = (
            'UPDATE player_precomputes '
            'SET "Prob_20_Disposals" = :p20, '
                '"Prob_25_Disposals" = :p25, '
                '"Prob_30_Disposals" = :p30 '
            'WHERE "Player" = :player'
        )
        res = conn.execute(text(sql), params)
        updated += res.rowcount if hasattr(res, "rowcount") else 1

print(f"✅ Wrote predictions for {len(pred_out)} players to player_precomputes "
      f"(updated rows: {updated}, skipped for missing rest/history: {skipped_no_rest_or_hist})")
