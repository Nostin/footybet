# 447_multi_target_ensemble.py
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    roc_auc_score, log_loss, classification_report,
    brier_score_loss, precision_recall_curve
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier
from db_connect import get_engine
import joblib

# =========================
# CONFIG
# =========================
TARGETS = [20, 25, 30]                 # thresholds to model
BLEND_GRID = np.linspace(0, 1, 21)     # alpha grid (LR share)
CALIB_CV = 5
RANDOM_STATE = 42
ODDS_COLS = ['odds_25_over', 'odds_over_25', 'odds_25p', 'odds_25', 'odds_over25']  # tweak per book if needed

engine = get_engine()

# -----------------------------
# Load data
# -----------------------------
df_train = pd.read_sql("SELECT * FROM model_feed_train", engine)
df_test  = pd.read_sql("SELECT * FROM model_feed_test", engine)

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
    if val in [12,13]: return '12–13 days'
    if val in [14,15]: return '14–15 days'
    return '16+ days'

for frame in (df_train, df_test):
    frame["days_bin"] = frame["days_since_last_game"].apply(bin_days)

df_train = pd.get_dummies(df_train, columns=["days_bin"], drop_first=True)
df_test  = pd.get_dummies(df_test,  columns=["days_bin"], drop_first=True)

# -----------------------------
# Feature selection (dynamic)
# -----------------------------
exact_wishlist = [
    'disposals_trend_last_5','disposals_delta_5',
    'disposals_max_last_5','disposals_min_last_5','disposals_std_last_5',
    'is_home_game','avg_team_disposals_last_4','is_wet_game',   # <- weather in
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
def existing(cols): return [c for c in cols if c in df_train.columns]
def existing_with_prefixes(prefixes):
    return [c for c in df_train.columns if any(c.startswith(p) for p in prefixes)]
bin_features = [c for c in df_train.columns if c.startswith("days_bin_")]

available_features = sorted(set(existing(exact_wishlist) +
                                existing_with_prefixes(prefix_wishlist) +
                                bin_features))
if not available_features:
    raise RuntimeError("No available features found. Check feature engineering.")

# ensure test has all features (without reindexing whole frame)
for c in [f for f in available_features if f not in df_test.columns]:
    df_test[c] = 0.0

def fit_oof_alpha(X_df, y_series):
    """Learn blend alpha on TRAIN via OOF log-loss."""
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    oof_lr   = np.zeros(len(X_df))
    oof_lgbm = np.zeros(len(X_df))
    for tr_idx, va_idx in skf.split(X_df, y_series):
        X_tr, X_va = X_df.iloc[tr_idx], X_df.iloc[va_idx]
        y_tr, y_va = y_series.iloc[tr_idx], y_series.iloc[va_idx]

        # LR fold (scale inside fold)
        sc = StandardScaler().fit(X_tr)
        lr_fold = LogisticRegression(max_iter=2000)
        lr_cal_fold = CalibratedClassifierCV(estimator=lr_fold, method='sigmoid', cv=CALIB_CV)
        lr_cal_fold.fit(sc.transform(X_tr), y_tr)
        oof_lr[va_idx] = lr_cal_fold.predict_proba(sc.transform(X_va))[:, 1]

        # LGBM fold
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
    """Fit calibrated LR (with scaler) and calibrated LGBM on full TRAIN."""
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
    thr_f1 = thresholds[max(best_idx - 1, 0)]
    print(f"📈 {name} best F1 threshold: {thr_f1:.3f} | P={precisions[best_idx]:.3f} R={recalls[best_idx]:.3f} F1={f1_scores[best_idx]:.3f}")
    return thr_f1

# -----------------------------
# Prepare odds column (optional ROI)
# -----------------------------
odds_col = next((c for c in ODDS_COLS if c in df_test.columns), None)

# -----------------------------
# Train/evaluate per target
# -----------------------------
all_out = []
for tgt in TARGETS:
    print("\n" + "="*70)
    print(f"🎯 Training ensemble for target: {tgt}+ disposals")

    target_col = f"target_{tgt}"
    if target_col not in df_train.columns:
        raise RuntimeError(f"{target_col} not found in training table.")

    # --- per-target NA handling
    tr = df_train.dropna(subset=[target_col]).copy()
    te = df_test.copy()

    # fill NA in features with train means (per target)
    train_means = tr[available_features].mean(numeric_only=True)
    tr[available_features] = tr[available_features].fillna(train_means)
    te[available_features] = te[available_features].fillna(train_means)

    # if test table has target col, keep for eval; else create NaNs
    if target_col in te.columns:
        pass
    else:
        te[target_col] = np.nan

    X_df = tr[available_features].astype(float)
    y    = tr[target_col].astype(int)
    X_te = te[available_features].astype(float)

    print(f"✅ Train shape: {X_df.shape} | Test shape: {X_te.shape}")

    # --- learn OOF alpha
    alpha = fit_oof_alpha(X_df, y)

    # --- fit final models
    lr_cal, lgbm_cal, scaler = fit_final_models(X_df, y)

    # --- predict probs
    p_lr   = lr_cal.predict_proba(scaler.transform(X_te))[:, 1]
    p_lgbm = lgbm_cal.predict_proba(X_te)[:, 1]
    p_bl   = alpha * p_lr + (1 - alpha) * p_lgbm

    te_out = te[['Date','Team','Player','Conditions']].copy()
    te_out['is_wet_flag']       = (te_out['Conditions'].astype(str).str.lower() == 'wet').astype(int)
    te_out['target_label']      = tgt
    te_out['pred_prob_lr']      = p_lr
    te_out['pred_prob_lgbm']    = p_lgbm
    te_out['pred_prob_blend']   = p_bl
    te_out['blend_alpha']       = float(alpha)

    # --- evaluation (if targets present)
    if not te[target_col].isna().all():
        y_test = te[target_col].astype(int).values
        thr_lr  = eval_and_best_threshold("Logistic (cal)", p_lr,  y_test)
        thr_lgb = eval_and_best_threshold("LightGBM (cal)", p_lgbm, y_test)
        thr_bl  = eval_and_best_threshold(f"BLEND (α={alpha:.2f})", p_bl, y_test)

        te_out['pred_class_blend_f1']  = (p_bl >= thr_bl).astype(int)
        te_out['pred_class_blend_0_5'] = (p_bl >= 0.5).astype(int)

        # quick wet vs dry performance cut (sanity check on feature utility)
        wet_mask = te_out['is_wet_flag'] == 1
        if wet_mask.any() and (~wet_mask).any():
            try:
                auc_wet  = roc_auc_score(y_test[wet_mask.values],  p_bl[wet_mask.values])
                auc_dry  = roc_auc_score(y_test[~wet_mask.values], p_bl[~wet_mask.values])
                print(f"🌧️ AUC (wet): {auc_wet:.4f} | ☀️ AUC (dry): {auc_dry:.4f}")
            except Exception:
                pass
    else:
        print("✅ Predictions complete. No test targets to evaluate.")
        te_out['pred_class_blend_f1']  = (p_bl >= 0.5).astype(int)
        te_out['pred_class_blend_0_5'] = (p_bl >= 0.5).astype(int)

    # --- EV / ROI (optional) on blend
    if odds_col is not None:
        implied = 1.0 / te[odds_col].astype(float) if odds_col in te.columns else pd.Series(np.nan, index=te.index)
        buffer = 0.02
        te_out['ev_threshold'] = implied.values + buffer
        te_out['bet_flag']     = (p_bl > te_out['ev_threshold']).astype(int)

    # --- Interpretability: is_wet rank
    # LR: mean coefficients across folds
    try:
        lr_fold_ests = [cc.estimator for cc in lr_cal.calibrated_classifiers_]
        lr_coef = np.vstack([est.coef_[0] for est in lr_fold_ests]).mean(axis=0)
    except Exception:
        lr_coef = LogisticRegression(max_iter=2000).fit(scaler.transform(X_df), y).coef_[0]
    lr_imp = pd.DataFrame({
        "feature": available_features,
        "coef": lr_coef,
        "abs": np.abs(lr_coef)
    }).sort_values("abs", ascending=False)
    if 'is_wet_game' in lr_imp['feature'].values:
        wet_rank_lr = lr_imp.reset_index(drop=True).query("feature=='is_wet_game'").index[0] + 1
        print(f"🧪 LR: 'is_wet_game' importance rank ≈ {wet_rank_lr}/{len(lr_imp)}")

    # LGBM: mean gain across folds
    try:
        lgbm_fold_models = [cc.estimator for cc in lgbm_cal.calibrated_classifiers_]
        gains = np.vstack([fm.booster_.feature_importance(importance_type='gain') for fm in lgbm_fold_models])
        gain_mean = gains.mean(axis=0)
        lgbm_imp = pd.DataFrame({"feature": available_features, "gain": gain_mean}) \
                    .sort_values("gain", ascending=False)
        if 'is_wet_game' in lgbm_imp['feature'].values:
            wet_rank_lgbm = lgbm_imp.reset_index(drop=True).query("feature=='is_wet_game'").index[0] + 1
            print(f"🧪 LGBM: 'is_wet_game' gain rank ≈ {wet_rank_lgbm}/{len(lgbm_imp)}")
    except Exception:
        pass

    # --- save models & scaler per target
    os.makedirs("models", exist_ok=True)
    joblib.dump(lr_cal,   f"models/logreg_calibrated_t{tgt}.pkl")
    joblib.dump(lgbm_cal, f"models/lgbm_calibrated_t{tgt}.pkl")
    joblib.dump(scaler,   f"models/logreg_scaler_t{tgt}.pkl")
    joblib.dump({"alpha": float(alpha)}, f"models/blend_alpha_t{tgt}.pkl")

    all_out.append(te_out)

# -----------------------------
# Save predictions (all targets)
# -----------------------------
if all_out:
    out = pd.concat(all_out, ignore_index=True)
    out.to_sql("blend_preds_multi", engine, if_exists="replace", index=False)
    print("✅ Predictions saved to table: blend_preds_multi")
else:
    print("⚠️ No outputs produced.")

print("✅ Done.")
