import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, log_loss, classification_report, brier_score_loss
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from db_connect import get_engine
import joblib
import os
import optuna

engine = get_engine()

# ------------------- Data Loading --------------------
df_train = pd.read_sql("SELECT * FROM model_feed_train", engine)
df_test = pd.read_sql("SELECT * FROM model_feed_test", engine)

# ------------- Feature Engineering (as before) --------------
def bin_days(val):
    if pd.isna(val):
        return np.nan
    if val <= 5:
        return '≤5 days'
    elif val == 6:
        return '6 days'
    elif val == 7:
        return '7 days'
    elif val == 8:
        return '8 days'
    elif val == 9:
        return '9 days'
    elif val == 10:
        return '10 days'
    elif val == 11:
        return '11 days'
    elif val in [12, 13]:
        return '12–13 days'
    elif val in [14, 15]:
        return '14–15 days'
    else:
        return '16+ days'

df_train["days_bin"] = df_train["days_since_last_game"].apply(bin_days)
df_test["days_bin"] = df_test["days_since_last_game"].apply(bin_days)

df_train = pd.get_dummies(df_train, columns=["days_bin"], drop_first=True)
df_test = pd.get_dummies(df_test, columns=["days_bin"], drop_first=True)
df_test = df_test.reindex(columns=df_train.columns, fill_value=0)

base_features = [
    'disposals_trend_last_5', 'disposals_delta_5',
    'disposals_max_last_5', 'disposals_min_last_5', 'disposals_std_last_5',
    'is_home_game', 'is_away_game',
    'avg_disposals_all', 'avg_team_disposals_last_4',
    'is_wet_game', 'wet_disposals_last_3', 'dry_disposals_last_3'
]
bin_features = [col for col in df_train.columns if col.startswith("days_bin_")]
available_features = base_features + bin_features

df_train = df_train.dropna(subset=['target_25'])
train_means = df_train[available_features].mean()
df_train[available_features] = df_train[available_features].fillna(train_means)
df_test[available_features] = df_test[available_features].fillna(train_means)

X = df_train[available_features]
y = df_train['target_25']
X_test = df_test[available_features]

print(f"✅ Using {len(available_features)} features: {available_features}")
print(f"✅ Train shape: {X.shape}, Test shape: {X_test.shape}")

if X.empty or X_test.empty:
    print("🚫 No usable training or test data. Check feature NA handling.")
    exit()

# --------------- Standardize ---------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)

# Split for Optuna tuning and calibration
X_train_opt, X_val_opt, y_train_opt, y_val_opt = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# -------------- Optuna Tuning ---------------
def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 300),
        "max_depth": trial.suggest_int("max_depth", 2, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "gamma": trial.suggest_float("gamma", 0, 5),
        "reg_alpha": trial.suggest_float("reg_alpha", 0, 5),
        "reg_lambda": trial.suggest_float("reg_lambda", 0, 5),
        "use_label_encoder": False,
        "eval_metric": "logloss",
        "random_state": 42,
        "verbosity": 0,
    }
    model = XGBClassifier(**params)
    model.fit(X_train_opt, y_train_opt)
    y_pred = model.predict_proba(X_val_opt)[:, 1]
    auc = roc_auc_score(y_val_opt, y_pred)
    return auc

print("🔬 Starting Optuna hyperparameter search...")
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=40)  # Increase n_trials for even better tuning!

print("\n🏆 Best Optuna trial:")
print(study.best_trial)
print("Best params:", study.best_trial.params)

# ------------- Retrain Model on Full Training Set ---------------
best_params = study.best_trial.params
best_params.update({"use_label_encoder": False, "eval_metric": "logloss", "random_state": 42})
best_model = XGBClassifier(**best_params)
# We'll use the entire scaled training set for final model
X_train_scaled, X_calib_scaled, y_train, y_calib = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)
best_model.fit(X_train_scaled, y_train)

# -------------- Calibration ---------------
calibrated_best = CalibratedClassifierCV(best_model, method='sigmoid', cv='prefit')
calibrated_best.fit(X_calib_scaled, y_calib)

# -------------- Predict and Evaluate ---------------
df_test['pred_prob'] = calibrated_best.predict_proba(X_test_scaled)[:, 1]

if 'target_25' in df_test.columns:
    y_test = df_test['target_25']
    auc = roc_auc_score(y_test, df_test['pred_prob'])
    logloss = log_loss(y_test, df_test['pred_prob'])
    brier = brier_score_loss(y_test, df_test['pred_prob'])
    preds = (df_test['pred_prob'] >= 0.5).astype(int)
    print(f"\n✅ FINAL EVAL | AUC: {auc:.4f} | Log Loss: {logloss:.4f} | Brier Score: {brier:.4f}\n")
    print("📋 Classification Report:")
    print(classification_report(y_test, preds))
else:
    print("✅ Predictions complete. No test targets to evaluate.")

# ------------- Feature Importance -------------
importances = best_model.feature_importances_
feature_importance = sorted(
    zip(available_features, importances, np.abs(importances)),
    key=lambda x: x[2],
    reverse=True
)
print("\n🔍 Top Predictive Features (XGBoost feature importances):")
for feature, importance, abs_importance in feature_importance:
    print(f"{feature.ljust(35)} ({importance:.4f})")

# ------------- Save Results and Model -------------
leaky_cols = ['Disposals', 'Kicks', 'Handballs', 'Goals', 'Behinds', 'Time on Ground %', 'Game Result']
df_test.drop(columns=[col for col in leaky_cols if col in df_test.columns], inplace=True)

df_test[["Date", "Team", "Player", "pred_prob"]].to_sql("xgb_preds", engine, if_exists="replace", index=False)
print("✅ Predictions saved to table: xgb_preds")

os.makedirs("models", exist_ok=True)
joblib.dump(calibrated_best, "models/xgb_model_calibrated.pkl")
joblib.dump(scaler, "models/xgb_scaler.pkl")
print("✅ Calibrated XGBoost model and scaler saved.")

