# machine_learning/6_train_goals_models.py

import os
from pathlib import Path
import sys
import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    roc_auc_score, log_loss, brier_score_loss, precision_recall_curve
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from db_connect import get_engine
engine = get_engine()

# =========================
# CONFIG
# =========================
TARGETS = [
    ("g1", 1),  # Goals >= 1
    ("g2", 2),  # Goals >= 2
    ("g3", 3),  # Goals >= 3
]
BLEND_GRID = np.linspace(0, 1, 21)  # LR share
CALIB_CV = 5
RANDOM_STATE = 42

# Optional goal odds columns (edit to your schema if you have market odds)
ODDS_COLS = [
    "odds_goals_1p", "odds_goals_2p", "odds_goals_3p",
    "odds_anytime_goal", "odds_2plus_goals", "odds_3plus_goals"
]

# -----------------------------
# Load data
# -----------------------------
df_train = pd.read_sql("SELECT * FROM model_feed_goals_train", engine)
df_test  = pd.read_sql("SELECT * FROM model_feed_goals_test",  engine)

# -----------------------------
# Rest-days binning -> one-hot
# -----------------------------
def bin_days(val):
    if pd.isna(val): return np.nan
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

for frame in (df_train, df_test):
    if "days_since_last_game" in frame.columns:
        frame["days_bin"] = frame["days_since_last_game"].apply(bin_days)
    else:
        frame["days_bin"] = np.nan

df_train = pd.get_dummies(df_train, columns=["days_bin"], drop_first=True)
df_test  = pd.get_dummies(df_test,  columns=["days_bin"], drop_first=True)

# -----------------------------
# Feature selection (dynamic)
# -----------------------------
exact_wishlist = [
    # trend / spread
    'goals_trend_last_5', 'goals_delta_5',
    'goals_max_last_5', 'goals_min_last_5', 'goals_std_last_5',
    # context + season/form
    'is_home_game', 'season_to_date_goals_median',
    'form_minus_season_goals_med_last_3',
    'team_goals_avg_last_5', 'opp_goals_conc_last_5',
    # shot volume / acc & wet/dry splits (if present)
    'wet_goals_last_3', 'dry_goals_last_3', 'wet_dry_goals_ratio_last_3',
    'shots_avg_last_3', 'shots_avg_last_6', 'shots_avg_last_10',
    'accuracy_avg_last_3', 'accuracy_avg_last_6', 'accuracy_avg_last_10',
]
prefix_wishlist = [
    # capped goal windows
    'goals_cap_avg_last_', 'goals_cap_med_last_', 'goals_cap_max_last_',
    'goals_cap_min_last_', 'goals_cap_var_last_',
    # contextual windows
    'goals_cap_home_avg_last_', 'goals_cap_away_avg_last_',
    'goals_cap_day_avg_last_', 'goals_cap_night_avg_last_', 'goals_cap_twilight_avg_last_',
    # floor/consistency
    'goals_floor_score_last_', 'accuracy_floor_score_last_',
]

def existing(cols): 
    return [c for c in cols if c in df_train.columns]
def existing_with_prefixes(prefixes):
    return [c for c in df_train.columns if any(c.startswith(p) for p in prefixes)]
bin_features = [c for c in df_train.columns if c.startswith("days_bin_")]

available_features_master = sorted(set(
    existing(exact_wishlist) +
    existing_with_prefixes(prefix_wishlist) +
    bin_features
))
if not available_features_master:
    raise RuntimeError("No available features found for goals. Check feature engineering output names.")

# Ensure test has all train features (add missing as zeros)
for c in [f for f in available_features_master if f not in df_test.columns]:
    df_test[c] = 0.0

# -----------------------------
# Blending helpers
# -----------------------------
def fit_oof_alpha(X_df, y_series):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    oof_lr   = np.zeros(len(X_df))
    oof_lgbm = np.zeros(len(X_df))
    for tr_idx, va_idx in skf.split(X_df, y_series):
        X_tr, X_va = X_df.iloc[tr_idx], X_df.iloc[va_idx]
        y_tr, y_va = y_series.iloc[tr_idx], y_series.iloc[va_idx]

        # LR in-fold scaling + sigmoid calibration
        sc = StandardScaler().fit(X_tr)
        lr_fold = LogisticRegression(max_iter=2000)
        lr_cal_fold = CalibratedClassifierCV(estimator=lr_fold, method='sigmoid', cv=CALIB_CV)
        lr_cal_fold.fit(sc.transform(X_tr), y_tr)
        oof_lr[va_idx] = lr_cal_fold.predict_proba(sc.transform(X_va))[:, 1]

        # LGBM + isotonic calibration
        lgbm_fold = LGBMClassifier(
            n_estimators=1200, learning_rate=0.03, max_depth=-1, num_leaves=63,
            min_child_samples=25, subsample=0.9, colsample_bytree=0.9,
            reg_alpha=0.1, reg_lambda=0.2, is_unbalance=True,
            random_state=RANDOM_STATE, n_jobs=-1, verbose=-1
        )
        lgbm_cal_fold = CalibratedClassifierCV(estimator=lgbm_fold, method='isotonic', cv=CALIB_CV)
        lgbm_cal_fold.fit(X_tr, y_tr)
        oof_lgbm[va_idx] = lgbm_cal_fold.predict_proba(X_va)[:, 1]

    best_alpha, best_ll = None, 1e9
    for a in BLEND_GRID:
        p = a * oof_lr + (1 - a) * oof_lgbm
        ll = log_loss(y_series, p)
        if ll < best_ll:
            best_ll, best_alpha = ll, a
    print(f"🧮 Learned blend α (LR share): {best_alpha:.2f} | OOF LogLoss: {best_ll:.4f}")
    return best_alpha

def fit_final_models(X_df, y_series):
    scaler = StandardScaler().fit(X_df)
    X_lr = scaler.transform(X_df)

    lr = LogisticRegression(max_iter=2000)
    lr_cal = CalibratedClassifierCV(estimator=lr, method='sigmoid', cv=CALIB_CV)
    lr_cal.fit(X_lr, y_series)

    lgbm = LGBMClassifier(
        n_estimators=1200, learning_rate=0.03, max_depth=-1, num_leaves=63,
        min_child_samples=25, subsample=0.9, colsample_bytree=0.9,
        reg_alpha=0.1, reg_lambda=0.2, is_unbalance=True,
        random_state=RANDOM_STATE, n_jobs=-1, verbose=-1
    )
    lgbm_cal = CalibratedClassifierCV(estimator=lgbm, method='isotonic', cv=CALIB_CV)
    lgbm_cal.fit(X_df, y_series)

    return lr_cal, lgbm_cal, scaler

def eval_and_best_threshold(name, probs, y_true):
    auc   = roc_auc_score(y_true, probs)
    ll    = log_loss(y_true, probs)
    brier = brier_score_loss(y_true, probs)
    print(f"✅ {name} | AUC: {auc:.4f} | LogLoss: {ll:.4f} | Brier: {brier:.4f}")
    precisions, recalls, thresholds = precision_recall_curve(y_true, probs)
    f1_scores = (2 * precisions * recalls) / (precisions + recalls + 1e-9)
    best_idx = int(np.nanargmax(f1_scores))
    thr_f1 = thresholds[max(best_idx - 1, 0)] if len(thresholds) else 0.5
    print(f"📈 {name} best F1 threshold: {thr_f1:.3f} | P={precisions[best_idx]:.3f} R={recalls[best_idx]:.3f} F1={f1_scores[best_idx]:.3f}")
    return thr_f1

# -----------------------------
# Per-target training loop
# -----------------------------
all_out = []
for key, thresh in TARGETS:
    print("\n" + "="*70)
    print(f"🎯 Training ensemble for target: Goals {thresh}+")

    target_col = f"target_{key}"  # target_g1 / target_g2 / target_g3
    if target_col not in df_train.columns:
        raise RuntimeError(f"{target_col} not found in training table.")

    # Use a per-target copy of features so we can drop zero-variance without mutating master
    features = list(available_features_master)

    tr = df_train.dropna(subset=[target_col]).copy()
    te = df_test.copy()

    # --- 1) Replace ±inf with NaN early
    for frame in (tr, te):
        frame[features] = frame[features].replace([np.inf, -np.inf], np.nan)

    # --- 2) Coerce ALL features to float (avoids boolean quantile errors)
    for frame in (tr, te):
        frame[features] = frame[features].apply(pd.to_numeric, errors='coerce').astype(float)

    # --- 3) Initial NA fill (train means) before clipping to stabilise quantiles
    train_means0 = tr[features].mean(numeric_only=True)
    tr[features] = tr[features].fillna(train_means0)
    te[features] = te[features].fillna(train_means0)

    # --- 4) Winsorize (clip) based on TRAIN distribution
    q_lo = tr[features].quantile(0.001)
    q_hi = tr[features].quantile(0.999)
    tr[features] = tr[features].clip(q_lo, q_hi, axis=1)
    te[features] = te[features].clip(q_lo, q_hi, axis=1)

    # --- 5) Feature-specific logical bounds
    for col in [c for c in features if c.startswith("accuracy_")]:
        tr[col] = tr[col].clip(0, 1)
        te[col] = te[col].clip(0, 1)
    for col in [c for c in features if ("floor_score" in c)]:
        tr[col] = tr[col].clip(0, 1)
        te[col] = te[col].clip(0, 1)
    for col in [c for c in features if "ratio" in c]:
        tr[col] = tr[col].clip(-10, 10)
        te[col] = te[col].clip(-10, 10)

    # --- 6) Final NA fill after clipping
    train_means = tr[features].mean(numeric_only=True)
    tr[features] = tr[features].fillna(train_means)
    te[features] = te[features].fillna(train_means)

    # --- 7) Drop zero-variance columns (per target)
    zero_var = [c for c in features if tr[c].std(skipna=True) == 0]
    if zero_var:
        print(f"ℹ️ Dropping zero-variance features: {zero_var}")
        features = [c for c in features if c not in zero_var]

    # --- 8) Guardrails: assert finite
    def assert_all_finite(name, df):
        bad = ~np.isfinite(df.values)
        if bad.any():
            bad_cols = list(df.columns[np.where(bad.any(axis=0))[0]])
            raise RuntimeError(f"{name} still has non-finite values in columns: {bad_cols}")
    assert_all_finite("TRAIN", tr[features])
    assert_all_finite("TEST",  te[features])

    # --- 9) Build matrices AFTER sanitising
    X_df = tr[features].astype(float)
    y    = tr[target_col].astype(int)
    X_te = te[features].astype(float)

    print(f"✅ Train shape: {X_df.shape} | Test shape: {X_te.shape}")

    # learn OOF alpha
    alpha = fit_oof_alpha(X_df, y)

    # fit final models
    lr_cal, lgbm_cal, scaler = fit_final_models(X_df, y)

    # predict
    p_lr   = lr_cal.predict_proba(scaler.transform(X_te))[:, 1]
    p_lgbm = lgbm_cal.predict_proba(X_te)[:, 1]
    p_bl   = alpha * p_lr + (1 - alpha) * p_lgbm

    # outputs
    base_cols = [c for c in ['Date', 'Team', 'Player', 'Conditions'] if c in te.columns]
    te_out = te[base_cols].copy()
    te_out['target_label']      = f'{thresh}+'
    te_out['pred_prob_lr']      = p_lr
    te_out['pred_prob_lgbm']    = p_lgbm
    te_out['pred_prob_blend']   = p_bl
    te_out['blend_alpha']       = float(alpha)

    # evaluation if targets exist in test
    if (target_col in te.columns) and (not te[target_col].isna().all()):
        y_test = te[target_col].astype(int).values
        # Optional: print metrics – comment out if you prefer silent test
        _ = eval_and_best_threshold("Logistic (cal)", p_lr,  y_test)
        _ = eval_and_best_threshold("LightGBM (cal)", p_lgbm, y_test)
        thr_bl  = eval_and_best_threshold(f"BLEND (α={alpha:.2f})", p_bl, y_test)
        te_out['pred_class_blend_f1']  = (p_bl >= thr_bl).astype(int)
        te_out['pred_class_blend_0_5'] = (p_bl >= 0.5).astype(int)
    else:
        print("✅ Predictions complete. No test targets to evaluate.")
        te_out['pred_class_blend_f1']  = (p_bl >= 0.5).astype(int)
        te_out['pred_class_blend_0_5'] = (p_bl >= 0.5).astype(int)

    # EV / ROI (optional): choose a relevant odds column if present
    odds_col = next((c for c in ODDS_COLS if c in te.columns), None)
    if odds_col is not None:
        implied = 1.0 / te[odds_col].astype(float)
        buffer = 0.02
        te_out['ev_threshold'] = implied.values + buffer
        te_out['bet_flag']     = (p_bl > te_out['ev_threshold']).astype(int)

    # save per-target artifacts
    os.makedirs("models/goals", exist_ok=True)
    joblib.dump(lr_cal,   f"models/goals/logreg_calibrated_{key}.pkl")
    joblib.dump(lgbm_cal, f"models/goals/lgbm_calibrated_{key}.pkl")
    joblib.dump(scaler,   f"models/goals/logreg_scaler_{key}.pkl")
    joblib.dump({"alpha": float(alpha)}, f"models/goals/blend_alpha_{key}.pkl")

    all_out.append(te_out)

# -----------------------------
# Save predictions (all targets)
# -----------------------------
if all_out:
    out = pd.concat(all_out, ignore_index=True)
    out.to_sql("goals_blend_preds_multi", engine, if_exists="replace", index=False)
    print("✅ Predictions saved to table: goals_blend_preds_multi")
else:
    print("⚠️ No outputs produced.")

print("✅ Done.")
