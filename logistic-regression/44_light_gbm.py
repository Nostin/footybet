import os
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    roc_auc_score, log_loss, classification_report,
    brier_score_loss, precision_recall_curve
)
from sklearn.preprocessing import StandardScaler
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

# Ensure test has all feature cols (don’t reindex whole frame)
missing_in_test = [c for c in available_features if c not in df_test.columns]
for c in missing_in_test:
    df_test[c] = 0.0
df_test[available_features] = df_test[available_features].fillna(train_means)

if not test_had_target and 'target_25' in df_test.columns:
    df_test['target_25'] = np.nan

# Use DataFrames/Series to keep feature names for LGBM (kills warnings)
X = df_train[available_features].astype(float)
y = df_train['target_25'].astype(int)
X_test = df_test[available_features].astype(float)


print(f"✅ Train shape: {X.shape} | Test shape: {X_test.shape}")
if X.size == 0 or X_test.size == 0:
    raise SystemExit("🚫 No usable training or test data. Check feature NA handling.")

# -----------------------------
# Model: LightGBM (plain) + isotonic calibration
# -----------------------------
# Solid starting params for tabular probabilities
lgbm = LGBMClassifier(
    n_estimators=1200,
    learning_rate=0.03,
    max_depth=-1,
    num_leaves=63,
    min_child_samples=25,
    subsample=0.9,
    colsample_bytree=0.9,
    reg_alpha=0.1,
    reg_lambda=0.2,
    is_unbalance=True,   # <— handle class imbalance
    random_state=42,
    n_jobs=-1,
    verbose=-1           # <— quiet logs
)

# Calibrate with CV=5; isotonic often calibrates tree ensembles better than sigmoid
calibrated_model = CalibratedClassifierCV(estimator=lgbm, method='isotonic', cv=5)
calibrated_model.fit(X, y)

# -----------------------------
# Predict probabilities
# -----------------------------
probs = calibrated_model.predict_proba(X_test)[:, 1]
df_test['pred_prob'] = probs

# -----------------------------
# Threshold tuning (if test has targets)
# -----------------------------
if 'target_25' in df_test.columns and not df_test['target_25'].isna().all():
    y_test = df_test['target_25'].astype(int).values

    precisions, recalls, thresholds = precision_recall_curve(y_test, probs)
    f1_scores = (2 * precisions * recalls) / (precisions + recalls + 1e-9)

    best_idx = int(np.nanargmax(f1_scores))
    best_threshold_f1 = thresholds[max(best_idx - 1, 0)]  # thresholds is one shorter

    print(f"📈 Best threshold by F1: {best_threshold_f1:.3f} | "
          f"Precision={precisions[best_idx]:.3f}, Recall={recalls[best_idx]:.3f}, F1={f1_scores[best_idx]:.3f}")

    # Optional: first threshold that reaches recall target
    recall_target = 0.65
    idxs = np.where(recalls >= recall_target)[0]
    if len(idxs) > 0:
        i = int(idxs[0])
        thr_recall = thresholds[max(i - 1, 0)]
        print(f"🎯 First threshold to reach {recall_target:.2f} recall: {thr_recall:.3f} | "
              f"Precision={precisions[i]:.3f}, Recall={recalls[i]:.3f}")

    df_test['pred_class_f1']  = (probs >= best_threshold_f1).astype(int)
    df_test['pred_class_0_5'] = (probs >= 0.5).astype(int)

    # Reference metrics
    auc   = roc_auc_score(y_test, probs)
    ll    = log_loss(y_test, probs)
    brier = brier_score_loss(y_test, probs)
    print(f"\n✅ FINAL EVAL | AUC: {auc:.4f} | LogLoss: {ll:.4f} | Brier: {brier:.4f}\n")
    print("📋 Classification Report (tuned threshold):")
    print(classification_report(y_test, df_test['pred_class_f1']))
    print("📋 Classification Report (0.5 threshold):")
    print(classification_report(y_test, df_test['pred_class_0_5']))
else:
    print("✅ Predictions complete. No test targets to evaluate.")
    df_test['pred_class_f1']  = (probs >= 0.5).astype(int)
    df_test['pred_class_0_5'] = (probs >= 0.5).astype(int)

# -----------------------------
# EV-based betting filter (optional)
# -----------------------------
odds_col_candidates = ['odds_25_over', 'odds_over_25', 'odds_25p', 'odds_25', 'odds_over25']
odds_col = next((c for c in odds_col_candidates if c in df_test.columns), None)

if odds_col is not None:
    implied = 1.0 / df_test[odds_col].astype(float)
    buffer  = 0.02  # tweak as you like
    df_test['ev_threshold'] = implied + buffer
    df_test['bet_flag']     = (df_test['pred_prob'] > df_test['ev_threshold']).astype(int)

    bets = df_test[df_test['bet_flag'] == 1]
    print(f"\n🎯 EV filter using column '{odds_col}' with +{buffer:.2f} buffer.")
    print(f"🧾 Bets flagged: {len(bets)} / {len(df_test)} rows")

    if 'target_25' in df_test.columns and not df_test['target_25'].isna().all() and len(bets) > 0:
        pnl = np.where(bets['target_25'] == 1, bets[odds_col] - 1.0, -1.0)
        roi = pnl.mean()
        hit_rate = bets['target_25'].mean()
        mean_edge = (bets['pred_prob'] - (1.0 / bets[odds_col])).mean()
        print(f"💰 Mean ROI per bet: {roi:.3f} | Hit Rate: {hit_rate:.3f} | Mean Edge: {mean_edge:.3f}")
else:
    print("\nℹ️ No odds column found for EV thresholding. Skipping ROI step.")

# -----------------------------
# Feature importance (gain)
# -----------------------------
# Average gain importances across calibrated CV folds' estimators
try:
    fold_models = [cc.estimator for cc in calibrated_model.calibrated_classifiers_]
    # If using isotonic, estimator is an underlying fitted LGBM per fold
    importances = np.vstack([fm.booster_.feature_importance(importance_type='gain') for fm in fold_models])
    fi_mean = importances.mean(axis=0)
    fi = sorted(zip(available_features, fi_mean), key=lambda x: x[1], reverse=True)
    print("\n🔍 Top Features by Gain (mean across folds):")
    for name, val in fi[:25]:
        print(f"{name.ljust(35)} {val:.2f}")
except Exception as e:
    print(f"⚠️ Could not extract LightGBM importances ({e}).")

# -----------------------------
# Save outputs
# -----------------------------
out_cols = ["Date", "Team", "Player", "pred_prob", "pred_class_0_5", "pred_class_f1"]
if 'bet_flag' in df_test.columns:
    out_cols += ['bet_flag', 'ev_threshold']
df_test[out_cols].to_sql("lgbm_preds", engine, if_exists="replace", index=False)
print("✅ Predictions saved to table: lgbm_preds")

os.makedirs("models", exist_ok=True)
joblib.dump(calibrated_model, "models/lgbm_calibrated.pkl")
print("✅ Calibrated LightGBM model saved.")
