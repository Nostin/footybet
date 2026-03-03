import sys
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
from sqlalchemy import text

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from db_connect import get_engine
from util import HOME_GROUNDS

engine = get_engine()

# ===================== Config =====================
TARGETS = [
    ("m2", "Prob_2_Marks"),
    ("m4", "Prob_4_Marks"),
    ("m6", "Prob_6_Marks"),
]


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


def bin_days(val):
    if pd.isna(val):
        return np.nan
    val = int(val)
    if val <= 5:
        return '≤5 days'
    if val == 6:
        return '6 days'
    if val == 7:
        return '7 days'
    if val == 8:
        return '8 days'
    if val == 9:
        return '9 days'
    if val == 10:
        return '10 days'
    if val == 11:
        return '11 days'
    if val in [12, 13]:
        return '12–13 days'
    if val in [14, 15]:
        return '14–15 days'
    return '16+ days'


def resolve_models_dir() -> Path:
    candidates = [
        ROOT / "models" / "marks",
        ROOT / "machine_learning" / "models" / "marks",
    ]
    for p in candidates:
        if p.exists():
            return p.resolve()
    raise FileNotFoundError(
        "Could not find marks model directory. Checked:\n" +
        "\n".join(str(p.resolve()) for p in candidates)
    )


MODELS_DIR = resolve_models_dir()


def load_bundle(key):
    lr = joblib.load(MODELS_DIR / f"logreg_calibrated_{key}.pkl")
    lgb = joblib.load(MODELS_DIR / f"lgbm_calibrated_{key}.pkl")
    sc = joblib.load(MODELS_DIR / f"logreg_scaler_{key}.pkl")
    meta = joblib.load(MODELS_DIR / f"blend_alpha_{key}.pkl")
    alpha = float(meta["alpha"] if isinstance(meta, dict) else meta)
    return lr, lgb, sc, alpha


def compute_train_means(expected_features):
    df_train = pd.read_sql("SELECT * FROM model_feed_marks_train", engine)

    if "days_since_last_game" in df_train.columns:
        df_train["days_bin"] = df_train["days_since_last_game"].apply(bin_days)
    else:
        df_train["days_bin"] = np.nan

    df_train = pd.get_dummies(df_train, columns=["days_bin"], drop_first=True)

    missing = [c for c in expected_features if c not in df_train.columns]
    for c in missing:
        df_train[c] = 0.0 if c.startswith("days_bin_") else np.nan

    X = df_train[expected_features].apply(pd.to_numeric, errors="coerce")

    # kill infs before taking means
    X = X.replace([np.inf, -np.inf], np.nan)

    # logical clipping used in training
    for col in [c for c in expected_features if "floor_score" in c]:
        if col in X.columns:
            X[col] = X[col].clip(0, 1)

    for col in [c for c in expected_features if "ratio" in c]:
        if col in X.columns:
            X[col] = X[col].clip(-10, 10)

    means = X.mean(numeric_only=True)
    means = means.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    return means


# ===================== Exact training enrich_marks =====================
def enrich_marks(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values(["Player", "Date"]).reset_index(drop=True)

    # Clean strings
    for c in ["Team", "Opponent", "Venue", "Timeslot", "Conditions"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()

    # --- Helpers ---
    def capped_marks_arr(x):
        return np.minimum(x, 12)

    def filtered_roll_mean(g, mask_col, w, value_col):
        vals = g[value_col].where(g[mask_col])
        return vals.shift(1).rolling(w, min_periods=1).mean()

    # --- Base features ---
    df["marks_capped"] = capped_marks_arr(df["Marks"].fillna(0).values)

    grouped_p = df.groupby("Player", sort=False)

    # --- Base rolling windows ---
    for w in [3, 6, 10]:
        df[f"marks_cap_avg_last_{w}"] = grouped_p.apply(
            lambda g: g["marks_capped"].shift(1).rolling(w, min_periods=1).mean(),
            include_groups=False
        ).reset_index(level=0, drop=True)

        df[f"marks_cap_med_last_{w}"] = grouped_p.apply(
            lambda g: g["marks_capped"].shift(1).rolling(w, min_periods=1).median(),
            include_groups=False
        ).reset_index(level=0, drop=True)

        df[f"marks_cap_max_last_{w}"] = grouped_p.apply(
            lambda g: g["marks_capped"].shift(1).rolling(w, min_periods=1).max(),
            include_groups=False
        ).reset_index(level=0, drop=True)

        df[f"marks_cap_min_last_{w}"] = grouped_p.apply(
            lambda g: g["marks_capped"].shift(1).rolling(w, min_periods=1).min(),
            include_groups=False
        ).reset_index(level=0, drop=True)

        df[f"marks_cap_var_last_{w}"] = grouped_p.apply(
            lambda g: g["marks_capped"].shift(1).rolling(w, min_periods=2).var(ddof=0),
            include_groups=False
        ).reset_index(level=0, drop=True)

    # --- Trend / spread / rest ---
    df["marks_trend_last_5"] = grouped_p["Marks"].transform(
        lambda x: x.shift(1).rolling(5, min_periods=2).apply(
            lambda y: np.polyfit(range(len(y)), y, 1)[0], raw=True
        )
    )
    df["marks_delta_5"] = grouped_p["Marks"].transform(lambda x: x.shift(1) - x.shift(6))
    df["marks_max_last_5"] = grouped_p["Marks"].transform(lambda x: x.shift(1).rolling(5).max())
    df["marks_min_last_5"] = grouped_p["Marks"].transform(lambda x: x.shift(1).rolling(5).min())
    df["marks_std_last_5"] = grouped_p["Marks"].transform(lambda x: x.shift(1).rolling(5).std())
    df["days_since_last_game"] = grouped_p["Date"].transform(lambda x: x.diff().dt.days.shift(1))

    # --- Home / Away flags ---
    def is_home_game(row):
        return pd.notna(row["Venue"]) and row["Venue"] in HOME_GROUNDS.get(row["Team"], [])

    df["is_home_game"] = df.apply(is_home_game, axis=1)
    df["is_away_game"] = ~df["is_home_game"]

    # --- Wet / Dry / Timeslot flags ---
    cond = df["Conditions"].astype(str).str.lower() if "Conditions" in df.columns else ""
    df["is_wet_game"] = (cond == "wet")
    df["is_dry_game"] = (cond == "dry")
    df["timeslot_category"] = df["Timeslot"].astype(str).str.lower().map(
        {"day": "day", "twilight": "twilight", "night": "night"}
    ).fillna("unknown")

    grouped_p = df.groupby("Player", sort=False)

    # --- Conditional last-N windows ---
    for w in [3, 6, 10]:
        for flag, colname in [
            ("is_wet_game", f"marks_cap_wet_avg_last_{w}"),
            ("is_dry_game", f"marks_cap_dry_avg_last_{w}"),
            ("is_home_game", f"marks_cap_home_avg_last_{w}"),
            ("is_away_game", f"marks_cap_away_avg_last_{w}")
        ]:
            df[colname] = grouped_p.apply(
                lambda g: filtered_roll_mean(g, flag, w, "marks_capped"),
                include_groups=False
            ).reset_index(level=0, drop=True)

        for ts in ["day", "night", "twilight"]:
            df[f"marks_cap_{ts}_avg_last_{w}"] = grouped_p.apply(
                lambda g: g.assign(ts=(g["timeslot_category"] == ts))
                          .pipe(lambda gg: filtered_roll_mean(gg, "ts", w, "marks_capped")),
                include_groups=False
            ).reset_index(level=0, drop=True)

    # --- Floor / consistency metric ---
    def floor_score(series):
        if len(series) == 0 or np.all(pd.isna(series)):
            return np.nan
        med = np.nanmedian(series)
        if not np.isfinite(med) or med <= 0:
            return np.nan
        drops = med - series
        drops = drops[drops > 0]
        mean_drop = np.nanmean(drops) if len(drops) else 0.0
        return round(1 - (mean_drop / med), 3)

    for w in [3, 6, 10, 22]:
        df[f"marks_floor_score_last_{w}"] = grouped_p["Marks"].transform(
            lambda x: x.shift(1).rolling(w, min_periods=2).apply(floor_score, raw=False)
        )

    # --- Season-to-date medians + form deltas ---
    df["season_to_date_marks_median"] = grouped_p["Marks"].transform(
        lambda x: x.expanding().median().shift(1)
    )
    df["form_minus_season_marks_med_last_3"] = (
        df["marks_cap_avg_last_3"] - df["season_to_date_marks_median"]
    )

    # --- Team marks pace & opponent concessions ---
    def merge_team_marks_features(df_in: pd.DataFrame):
        try:
            tp = pd.read_sql(
                'SELECT "Date","Team","marks_avg_last_5","concede_marks_avg_last_5" '
                'FROM team_precompute',
                engine
            )
            tp["Date"] = pd.to_datetime(tp["Date"], errors="coerce")
            tp["Team"] = tp["Team"].astype(str).str.strip()
            tp = tp.dropna(subset=["Date"]).sort_values(["Team", "Date"]).reset_index(drop=True)

            tp["team_marks_last_5_pre"] = tp.groupby("Team")["marks_avg_last_5"].shift(1)
            tp["opp_marks_conc_last_5_pre"] = tp.groupby("Team")["concede_marks_avg_last_5"].shift(1)

            out = df_in.merge(
                tp[["Date", "Team", "team_marks_last_5_pre"]]
                .rename(columns={"team_marks_last_5_pre": "team_marks_avg_last_5"}),
                on=["Date", "Team"], how="left"
            )
            if "Opponent" in out.columns:
                out = out.merge(
                    tp[["Date", "Team", "opp_marks_conc_last_5_pre"]]
                    .rename(columns={
                        "Team": "Opponent",
                        "opp_marks_conc_last_5_pre": "opp_marks_conc_last_5"
                    }),
                    on=["Date", "Opponent"], how="left"
                )
            else:
                out["opp_marks_conc_last_5"] = np.nan
            return out

        except Exception:
            base = df_in[["Date", "Team", "Opponent", "Marks"]].copy()
            base["Marks"] = base["Marks"].fillna(0)

            team_for = (
                base.groupby(["Date", "Team"], as_index=False)["Marks"]
                    .sum()
                    .rename(columns={"Marks": "team_marks"})
                    .sort_values(["Team", "Date"])
                    .reset_index(drop=True)
            )

            team_against = (
                base.groupby(["Date", "Opponent"], as_index=False)["Marks"]
                    .sum()
                    .rename(columns={"Opponent": "Team", "Marks": "opp_marks_conceded"})
                    .sort_values(["Team", "Date"])
                    .reset_index(drop=True)
            )

            t = (
                team_for.merge(team_against, on=["Date", "Team"], how="left")
                        .sort_values(["Team", "Date"])
                        .reset_index(drop=True)
            )

            t["team_marks_avg_last_5"] = (
                t.groupby("Team")["team_marks"]
                 .transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
            )
            t["opp_marks_conc_last_5"] = (
                t.groupby("Team")["opp_marks_conceded"]
                 .transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
            )

            out = df_in.merge(
                t[["Date", "Team", "team_marks_avg_last_5"]],
                on=["Date", "Team"], how="left"
            )

            if "Opponent" in out.columns:
                t_opp = t[["Date", "Team", "opp_marks_conc_last_5"]].rename(columns={"Team": "Opponent"})
                out = out.merge(t_opp, on=["Date", "Opponent"], how="left")
            else:
                out["opp_marks_conc_last_5"] = np.nan

            return out

    df = merge_team_marks_features(df)

    # --- Missed-time proxy ---
    def missed_time_last4(g):
        dates = g["Date"].values
        team = g["Team"].iloc[0]
        team_dates = df[df["Team"] == team][["Date"]].drop_duplicates().sort_values("Date")
        out = []
        for d in dates:
            recent_team = team_dates[team_dates["Date"] < d].tail(4)["Date"].values
            miss = False
            for td in recent_team:
                row = g[g["Date"] == td]
                if row.empty:
                    miss = True
                    break
                tog = row["Time on Ground %"].iloc[0] if "Time on Ground %" in g.columns else np.nan
                if pd.isna(tog) or tog < 50:
                    miss = True
                    break
            out.append(int(miss))
        return pd.Series(out, index=g.index)

    df["missed_time_last4"] = grouped_p.apply(
        missed_time_last4,
        include_groups=False
    ).reset_index(level=0, drop=True)

    # --- Wet/Dry rolling avgs ---
    df["wet_marks_last_3"] = grouped_p.apply(
        lambda g: g["Marks"].where(g["is_wet_game"]).shift(1).rolling(3, min_periods=1).mean(),
        include_groups=False
    ).reset_index(level=0, drop=True)

    df["dry_marks_last_3"] = grouped_p.apply(
        lambda g: g["Marks"].where(g["is_dry_game"]).shift(1).rolling(3, min_periods=1).mean(),
        include_groups=False
    ).reset_index(level=0, drop=True)

    df["wet_dry_marks_ratio_last_3"] = np.where(
        df["dry_marks_last_3"] > 0,
        df["wet_marks_last_3"] / df["dry_marks_last_3"],
        np.nan
    )

    # --- One-hot-ish ints ---
    df["is_home_game"] = df["is_home_game"].astype(int)
    df["is_away_game"] = df["is_away_game"].astype(int)
    df["is_wet_game"]  = df["is_wet_game"].astype(int)
    df["is_dry_game"]  = df["is_dry_game"].astype(int)

    # --- Targets ---
    df["target_m2"] = (df["Marks"] >= 2).astype(int)
    df["target_m4"] = (df["Marks"] >= 4).astype(int)
    df["target_m6"] = (df["Marks"] >= 6).astype(int)

    # --- Clean-up / de-dup ---
    df = df.drop_duplicates(subset=["Player", "Date"]).reset_index(drop=True)
    return df


# ===================== Build future rows =====================
print(f"✅ Using models dir: {MODELS_DIR}")

hist = pd.read_sql('SELECT * FROM player_stats ORDER BY "Date" ASC', engine)
hist["Date"] = pd.to_datetime(hist["Date"], errors="coerce")

base = pd.read_sql(
    'SELECT "Player","Team","Next_Opponent","Next_Venue","Next_Timeslot","Next_Venue_Home","Days_since_last_game" '
    'FROM player_precomputes',
    engine
)

base["Next_Timeslot"] = base["Next_Timeslot"].astype(str).str.lower().replace({"nan": np.nan})

future_rows = []
skipped_no_rest_or_hist = 0

for _, r in base.iterrows():
    player = r["Player"]
    team = r["Team"]

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
        "Marks": np.nan,
        "Game Result": np.nan,
        "Time on Ground %": np.nan,
        "_force_is_home": int(r.get("Next_Venue_Home", 0)) if not pd.isna(r.get("Next_Venue_Home", np.nan)) else 0,
        "_force_days_since": to_python_scalar(days),
    })

future_df = pd.DataFrame(future_rows)

if future_df.empty:
    print("ℹ️ No eligible players with Days_since_last_game & history; nothing to predict.")
    sys.exit(0)

combined = pd.concat(
    [hist, future_df.drop(columns=["_force_is_home", "_force_days_since"], errors="ignore")],
    ignore_index=True
)

enriched = enrich_marks(combined)

pred_rows = future_df[["Player", "Date", "_force_is_home", "_force_days_since", "Team", "Opponent"]].merge(
    enriched,
    on=["Player", "Date"],
    how="left",
    suffixes=("", "_enr")
)

for col in ["Team", "Opponent"]:
    if f"{col}_x" in pred_rows.columns or f"{col}_y" in pred_rows.columns:
        left = pred_rows.get(f"{col}_x", pred_rows.get(col))
        right = pred_rows.get(f"{col}_y", pred_rows.get(f"{col}_enr"))
        pred_rows[col] = left if left is not None else right
        drop_cols = [c for c in [f"{col}_x", f"{col}_y", f"{col}_enr"] if c in pred_rows.columns]
        if drop_cols:
            pred_rows.drop(columns=drop_cols, inplace=True)

# Override with known next-game values
pred_rows["is_home_game"] = pred_rows["_force_is_home"].astype(int)
pred_rows["is_away_game"] = (1 - pred_rows["is_home_game"]).astype(int)
pred_rows["days_since_last_game"] = pred_rows["_force_days_since"]

# Recreate days_bin one-hots like training
pred_rows["days_bin"] = pred_rows["days_since_last_game"].apply(bin_days)
pred_rows = pd.get_dummies(pred_rows, columns=["days_bin"], drop_first=True)

# ===================== Predict =====================
pred_out = pred_rows[["Player"]].copy()

for key, out_col in TARGETS:
    lr, lgb, sc, alpha = load_bundle(key)

    if not hasattr(sc, "feature_names_in_"):
        raise RuntimeError(
            f"Scaler for target {key} lacks feature_names_in_. "
            "Retrain with a sklearn version that persists feature_names_in_."
        )

    expected_features = list(sc.feature_names_in_)

    # Ensure missing day-bin dummies exist
    for c in expected_features:
        if c.startswith("days_bin_") and c not in pred_rows.columns:
            pred_rows[c] = 0.0

    X_use = pred_rows.reindex(columns=expected_features, fill_value=np.nan)
    X_use = X_use.apply(pd.to_numeric, errors="coerce").astype(float)

    # Match training sanitisation
    X_use = X_use.replace([np.inf, -np.inf], np.nan)

    for col in [c for c in expected_features if "floor_score" in c]:
        if col in X_use.columns:
            X_use[col] = X_use[col].clip(0, 1)

    for col in [c for c in expected_features if "ratio" in c]:
        if col in X_use.columns:
            X_use[col] = X_use[col].clip(-10, 10)

    train_means = compute_train_means(expected_features)
    X_use = X_use.fillna(train_means)
    X_use = X_use.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    bad = ~np.isfinite(X_use.values)
    if bad.any():
        bad_cols = list(X_use.columns[np.where(bad.any(axis=0))[0]])
        raise RuntimeError(f"{key}: prediction matrix still has non-finite values in columns: {bad_cols}")

    p_lr = lr.predict_proba(sc.transform(X_use))[:, 1]
    p_lgb = lgb.predict_proba(X_use)[:, 1]
    p_bl = alpha * p_lr + (1 - alpha) * p_lgb

    pred_out[out_col] = np.clip(p_bl * 100.0, 0.0, 100.0)

# ===================== Persist =====================
with engine.begin() as conn:
    for col in ["Prob_2_Marks", "Prob_4_Marks", "Prob_6_Marks"]:
        conn.exec_driver_sql(
            f'ALTER TABLE player_precomputes ADD COLUMN IF NOT EXISTS "{col}" DOUBLE PRECISION'
        )

updated = 0
with engine.begin() as conn:
    for _, r in pred_out.iterrows():
        params = {
            "p2": to_python_scalar(r.get("Prob_2_Marks")),
            "p4": to_python_scalar(r.get("Prob_4_Marks")),
            "p6": to_python_scalar(r.get("Prob_6_Marks")),
            "player": r["Player"],
        }

        sql = (
            'UPDATE player_precomputes '
            'SET "Prob_2_Marks" = :p2, '
                '"Prob_4_Marks" = :p4, '
                '"Prob_6_Marks" = :p6 '
            'WHERE "Player" = :player'
        )
        res = conn.execute(text(sql), params)
        updated += res.rowcount if hasattr(res, "rowcount") else 1

print(
    f"✅ Wrote marks predictions for {len(pred_out)} players to player_precomputes "
    f"(updated rows: {updated}, skipped for missing rest/history: {skipped_no_rest_or_hist})"
)