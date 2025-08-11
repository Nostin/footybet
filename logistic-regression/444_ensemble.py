# 445_ensemble_oof_blend.py
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

engine = get_engine()

# -----------------------------
# Load data
# -----------------------------
df_train = pd.read_sql("SELECT * FROM model_feed_train", engine)
df_test  = pd.read_sql("SELECT * FROM model_feed_test", engine)
test_had_target = 'target_25' in df_test.columns

# -----------------------------
# Rest-days binning -> one-hot
# -----------------------------
def bin_days(val):
    if pd.isna(val): return np.nan
    if val <= 5:      return '≤5 days'
    if val == 6:      return '6 days'
    if val == 7:      return '7 days'
    if val == 8:      return '8 days'
    if val == 9:      return '9 days'
    if val == 10:     return '10 days'
    if val == 11:     return '11 days'
    if val in [12,13]:return '12–13 days'
    if val in [14,15]:return '14–15 days'
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
bin_features = [c for c in df_train.columns if c.startswith("days_bin_")]

def existing(cols): return [c for c in cols if c in df_train.columns]
def existing_with_prefixes(prefixes):
    return [c for c in df_train.columns if any(c.startswith(p) for p in prefixes)]

available_features = sorted(set(existing(exact_wishlist) + existing_with_prefixes(prefix_wishlist) + bin_features))
if not available_features:
    raise RuntimeError("No available features found. Check feature engineering step.")
print(f"✅ Using {len(available_features)} features")

# -----------------------------
# Targets & NA handling
# -----------------------------
if 'target_25' not in df_train.columns:
    raise RuntimeError("target_25 not found in training table.")
df_train = df_train.dropna(subset=['target_25'])

train_means = df_train[available_features].mean(numeric_only=True)
df_train[available_features] = df_train[available_features].fillna(train_means)

# Ensure test has all feature cols (don’t reindex the whole frame)
missing_in_test = [c for c in available_features if c not in df_test.columns]
for c in missing_in_test:
    df_test[c] = 0.0
df_test[available_features] = df_test[available_features].fillna(train_means)

if not test_had_target and 'target_25' in df_test.columns:
    df_test['target_25'] = np.nan

X_df     = df_train[available_features].astype(float)
y_series = df_train['target_25'].astype(int)
Xtest_df = df_test[available_features].astype(float)

print(f"✅ Train shape: {X_df.shape} | Test shape: {Xtest_df.shape}")
if X_df.size == 0 or Xtest_df.size == 0:
    raise SystemExit("🚫 No usable training or test data. Check feature NA handling.")

# -----------------------------
# OOF blend weight learning (alpha for LR share)
# -----------------------------
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_lr   = np.zeros(len(X_df))
oof_lgbm = np.zeros(len(X_df))

for tr_idx, va_idx in skf.split(X_df, y_series):
    X_tr, X_va = X_df.iloc[tr_idx], X_df.iloc[va_idx]
    y_tr, y_va = y_series.iloc[tr_idx], y_series.iloc[va_idx]

    # LR fold (scale inside fold to avoid leakage)
    sc = StandardScaler().fit(X_tr)
    lr_fold = LogisticRegression(max_iter=2000)
    lr_cal_fold = CalibratedClassifierCV(estimator=lr_fold, method='sigmoid', cv=5)
    lr_cal_fold.fit(sc.transform(X_tr), y_tr)
    oof_lr[va_idx] = lr_cal_fold.predict_proba(sc.transform(X_va))[:, 1]

    # LGBM fold
    lgbm_fold = LGBMClassifier(
        n_estimators=1200, learning_rate=0.03, max_depth=-1, num_leaves=63,
        min_child_samples=25, subsample=0.9, colsample_bytree=0.9,
        reg_alpha=0.1, reg_lambda=0.2, is_unbalance=True,
        random_state=42, n_jobs=-1, verbose=-1
    )
    lgbm_cal_fold = CalibratedClassifierCV(estimator=lgbm_fold, method='isotonic', cv=5)
    lgbm_cal_fold.fit(X_tr, y_tr)
    oof_lgbm[va_idx] = lgbm_cal_fold.predict_proba(X_va)[:, 1]

alphas = np.linspace(0, 1, 21)  # 0.00, 0.05, ..., 1.00
best_alpha, best_ll = None, 1e9
for a in alphas:
    p = a * oof_lr + (1 - a) * oof_lgbm
    ll = log_loss(y_series, p)
    if ll < best_ll:
        best_ll, best_alpha = ll, a

print(f"🧮 Learned blend α (LR share): {best_alpha:.2f} | OOF LogLoss: {best_ll:.4f}")

# -----------------------------
# Final models on ALL training data
# -----------------------------
# Logistic (scale on full train)
scaler = StandardScaler().fit(X_df)
X_lr     = scaler.transform(X_df)
Xtest_lr = scaler.transform(Xtest_df)

lr = LogisticRegression(max_iter=2000)
lr_cal = CalibratedClassifierCV(estimator=lr, method='sigmoid', cv=5)
lr_cal.fit(X_lr, y_series)

# LightGBM on full train
lgbm = LGBMClassifier(
    n_estimators=1200, learning_rate=0.03, max_depth=-1, num_leaves=63,
    min_child_samples=25, subsample=0.9, colsample_bytree=0.9,
    reg_alpha=0.1, reg_lambda=0.2, is_unbalance=True,
    random_state=42, n_jobs=-1, verbose=-1
)
lgbm_cal = CalibratedClassifierCV(estimator=lgbm, method='isotonic', cv=5)
lgbm_cal.fit(X_df, y_series)

# -----------------------------
# Predict probabilities (test)
# -----------------------------
p_lr    = lr_cal.predict_proba(Xtest_lr)[:, 1]
p_lgbm  = lgbm_cal.predict_proba(Xtest_df)[:, 1]
p_blend = best_alpha * p_lr + (1 - best_alpha) * p_lgbm

df_test['pred_prob_lr']    = p_lr
df_test['pred_prob_lgbm']  = p_lgbm
df_test['pred_prob_blend'] = p_blend
df_test['blend_alpha']     = best_alpha

# -----------------------------
# Threshold tuning & evaluation (if test has targets)
# -----------------------------
if 'target_25' in df_test.columns and not df_test['target_25'].isna().all():
    y_test = df_test['target_25'].astype(int).values

    def eval_block(name, probs, y_true):
        auc   = roc_auc_score(y_true, probs)
        ll    = log_loss(y_true, probs)
        brier = brier_score_loss(y_true, probs)
        print(f"✅ {name} | AUC: {auc:.4f} | LogLoss: {ll:.4f} | Brier: {brier:.4f}")
    eval_block("Logistic (cal)", p_lr, y_test)
    eval_block("LightGBM (cal)", p_lgbm, y_test)
    eval_block(f"BLEND (α={best_alpha:.2f})", p_blend, y_test)

    precisions, recalls, thresholds = precision_recall_curve(y_test, p_blend)
    f1_scores = (2 * precisions * recalls) / (precisions + recalls + 1e-9)
    best_idx = int(np.nanargmax(f1_scores))
    thr_f1 = thresholds[max(best_idx - 1, 0)]
    print(f"📈 Blend best F1 threshold: {thr_f1:.3f} | "
          f"P={precisions[best_idx]:.3f} R={recalls[best_idx]:.3f} F1={f1_scores[best_idx]:.3f}")

    df_test['pred_class_blend_f1']  = (p_blend >= thr_f1).astype(int)
    df_test['pred_class_blend_0_5'] = (p_blend >= 0.5).astype(int)

    print("\n📋 Classification Report (BLEND tuned):")
    print(classification_report(y_test, df_test['pred_class_blend_f1']))
    print("📋 Classification Report (BLEND @0.5):")
    print(classification_report(y_test, df_test['pred_class_blend_0_5']))
else:
    print("✅ Predictions complete. No test targets to evaluate.")
    df_test['pred_class_blend_f1']  = (p_blend >= 0.5).astype(int)
    df_test['pred_class_blend_0_5'] = (p_blend >= 0.5).astype(int)

# -----------------------------
# EV-based betting filter (optional, on BLEND)
# -----------------------------
odds_col_candidates = ['odds_25_over', 'odds_over_25', 'odds_25p', 'odds_25', 'odds_over25']
odds_col = next((c for c in df_test.columns if c in odds_col_candidates), None)

if odds_col is not None:
    implied = 1.0 / df_test[odds_col].astype(float)
    buffer  = 0.02
    df_test['ev_threshold'] = implied + buffer
    df_test['bet_flag']     = (df_test['pred_prob_blend'] > df_test['ev_threshold']).astype(int)

    bets = df_test[df_test['bet_flag'] == 1]
    print(f"\n🎯 EV filter using '{odds_col}' with +{buffer:.2f} buffer. Bets flagged: {len(bets)} / {len(df_test)}")
    if 'target_25' in df_test.columns and not df_test['target_25'].isna().all() and len(bets) > 0:
        pnl = np.where(bets['target_25'] == 1, bets[odds_col] - 1.0, -1.0)
        roi = pnl.mean()
        hit_rate = bets['target_25'].mean()
        mean_edge = (bets['pred_prob_blend'] - (1.0 / bets[odds_col])).mean()
        print(f"💰 Mean ROI per bet: {roi:.3f} | Hit Rate: {hit_rate:.3f} | Mean Edge: {mean_edge:.3f}")
else:
    print("\nℹ️ No odds column found for EV thresholding. Skipping ROI step.")

# -----------------------------
# Interpretability
# -----------------------------
# LR coefficients (mean across calibrated folds if available)
try:
    lr_fold_ests = [cc.estimator for cc in lr_cal.calibrated_classifiers_]
    lr_coef = np.vstack([est.coef_[0] for est in lr_fold_ests]).mean(axis=0)
except Exception as e:
    print(f"⚠️ LR fold coeffs unavailable ({e}); fitting plain LR for coeffs.")
    lr_coef = LogisticRegression(max_iter=2000).fit(X_lr, y_series).coef_[0]

lr_imp = sorted(zip(available_features, lr_coef, np.abs(lr_coef)), key=lambda x: x[2], reverse=True)
print("\n🔍 Top Logistic Features (mean coefficients):")
for f, c, ac in lr_imp[:20]:
    print(f"{f.ljust(35)} {'↑' if c>0 else '↓'} ({c:.4f})")

# LGBM gain importances (mean across calibrated folds)
try:
    lgbm_fold_models = [cc.estimator for cc in lgbm_cal.calibrated_classifiers_]
    gains = np.vstack([fm.booster_.feature_importance(importance_type='gain') for fm in lgbm_fold_models])
    gain_mean = gains.mean(axis=0)
    lgbm_imp = sorted(zip(available_features, gain_mean), key=lambda x: x[1], reverse=True)
    print("\n🌲 Top LightGBM Features (mean gain):")
    for f, g in lgbm_imp[:20]:
        print(f"{f.ljust(35)} {g:.2f}")
except Exception as e:
    print(f"⚠️ Could not extract LightGBM importances ({e}).")

# -----------------------------
# Save outputs
# -----------------------------
out_cols = ["Date", "Team", "Player",
            "pred_prob_lr", "pred_prob_lgbm", "pred_prob_blend",
            "pred_class_blend_0_5", "pred_class_blend_f1", "blend_alpha"]
if 'bet_flag' in df_test.columns:
    out_cols += ['bet_flag', 'ev_threshold']
df_test[out_cols].to_sql("blend_preds", engine, if_exists="replace", index=False)
print("✅ Predictions saved to table: blend_preds")

os.makedirs("models", exist_ok=True)
joblib.dump(lr_cal,       "models/logreg_calibrated.pkl")
joblib.dump(lgbm_cal,     "models/lgbm_calibrated.pkl")
joblib.dump(scaler,       "models/logreg_scaler.pkl")
joblib.dump({"alpha": float(best_alpha)}, "models/blend_alpha.pkl")
print("✅ Models, scaler, and blend weight saved.")
