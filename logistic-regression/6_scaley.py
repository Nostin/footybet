import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, log_loss, classification_report, brier_score_loss
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from db_connect import get_engine
import joblib
import os

engine = get_engine()

# Load data
df_train = pd.read_sql("SELECT * FROM model_feed_train", engine)
df_test = pd.read_sql("SELECT * FROM model_feed_test", engine)

# Bin days_since_last_game
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

# One-hot encode the binned feature
df_train = pd.get_dummies(df_train, columns=["days_bin"], drop_first=True)
df_test = pd.get_dummies(df_test, columns=["days_bin"], drop_first=True)

# Align test columns to train (to cover edge cases)
df_test = df_test.reindex(columns=df_train.columns, fill_value=0)

# Feature list
base_features = [
    'disposals_trend_last_5', 'disposals_delta_5',
    'disposals_max_last_5', 'disposals_min_last_5', 'disposals_std_last_5',
    'is_home_game', 'is_away_game',
    'avg_disposals_all', 'avg_team_disposals_last_4',
    'is_wet_game',
    'wet_disposals_last_3',
    'dry_disposals_last_3'
]
bin_features = [col for col in df_train.columns if col.startswith("days_bin_")]
available_features = base_features + bin_features

# Drop missing targets only
df_train = df_train.dropna(subset=['target_25'])

# Fill NaNs in features
train_means = df_train[available_features].mean()
df_train[available_features] = df_train[available_features].fillna(train_means)
df_test[available_features] = df_test[available_features].fillna(train_means)

# Extract X/y
X = df_train[available_features]
y = df_train['target_25']
X_test = df_test[available_features]

print(f"✅ Using {len(available_features)} features: {available_features}")
print(f"✅ Train shape: {X.shape}, Test shape: {X_test.shape}")

if X.empty or X_test.empty:
    print("🚫 No usable training or test data. Check feature NA handling.")
    exit()

# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)

# Split off calibration set
X_train_scaled, X_calib_scaled, y_train, y_calib = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train base model
base_model = LogisticRegression(max_iter=1000)
base_model.fit(X_train_scaled, y_train)

# Calibrate
calibrated_model = CalibratedClassifierCV(base_model, method='sigmoid', cv='prefit')
calibrated_model.fit(X_calib_scaled, y_calib)

# Predict
df_test['pred_prob'] = calibrated_model.predict_proba(X_test_scaled)[:, 1]

# Evaluate if possible
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

# Drop leaky columns
leaky_cols = ['Disposals', 'Kicks', 'Handballs', 'Goals', 'Behinds', 'Time on Ground %', 'Game Result']
df_test.drop(columns=[col for col in leaky_cols if col in df_test.columns], inplace=True)

# Feature importance from coefficients
coef = base_model.coef_[0]
feature_importance = sorted(
    zip(available_features, coef, np.abs(coef)),
    key=lambda x: x[2],
    reverse=True
)

print("\n🔍 Top Predictive Features (logistic regression coefficients):")
for feature, raw_coef, abs_coef in feature_importance:
    direction = "↑" if raw_coef > 0 else "↓"
    print(f"{feature.ljust(35)} {direction} ({raw_coef:.4f})")

print("\n📊 Days Bin Coefficients (from logistic regression):")
for feature, coef in zip(available_features, base_model.coef_[0]):
    if feature.startswith("days_bin_"):
        print(f"{feature:20} → {coef:.4f}")

# Save predictions
df_test[["Date", "Team", "Player", "pred_prob"]].to_sql("logistic_preds", engine, if_exists="replace", index=False)
print("✅ Predictions saved to table: logistic_preds")

# Save model and scaler
os.makedirs("models", exist_ok=True)
joblib.dump(calibrated_model, "models/logreg_model_calibrated.pkl")
joblib.dump(scaler, "models/logreg_scaler.pkl")
print("✅ Calibrated model and scaler saved.")
