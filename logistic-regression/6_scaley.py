import pandas as pd
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

# Feature list
candidate_features = [
    'disposals_trend_last_5', 'disposals_delta_5',
    'disposals_max_last_5', 'disposals_min_last_5', 'disposals_std_last_5',
    'days_since_last_game',
    'is_home_game', 'is_away_game',
    'avg_disposals_all', 'avg_team_disposals_last_4'
]

# Filter features available in both sets
available_features = [f for f in candidate_features if f in df_train.columns and f in df_test.columns]

# Drop only rows missing target (not features)
df_train = df_train.dropna(subset=['target_25'])

# Fill feature NaNs with training means
train_means = df_train[available_features].mean()
df_train[available_features] = df_train[available_features].fillna(train_means)
df_test[available_features] = df_test[available_features].fillna(train_means)

# Extract features/target
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

# Split off a calibration set
X_train_scaled, X_calib_scaled, y_train, y_calib = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train base model
base_model = LogisticRegression(max_iter=1000)
base_model.fit(X_train_scaled, y_train)

# Calibrate with Platt scaling
calibrated_model = CalibratedClassifierCV(base_model, method='sigmoid', cv='prefit')
calibrated_model.fit(X_calib_scaled, y_calib)

# Predict on test set
df_test['pred_prob'] = calibrated_model.predict_proba(X_test_scaled)[:, 1]

# Evaluate if test target exists
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

# Leaky columns — drop them from test only
leaky_cols = ['Disposals', 'Kicks', 'Handballs', 'Goals', 'Behinds', 'Time on Ground %', 'Game Result']
df_test.drop(columns=[col for col in leaky_cols if col in df_test.columns], inplace=True)

# Save predictions
df_test[["Date", "Team", "Player", "pred_prob"]].to_sql("logistic_preds", engine, if_exists="replace", index=False)
print("✅ Predictions saved to table: logistic_preds")

# Save model and scaler
os.makedirs("models", exist_ok=True)
joblib.dump(calibrated_model, "models/logreg_model_calibrated.pkl")
joblib.dump(scaler, "models/logreg_scaler.pkl")

print("✅ Calibrated model and scaler saved.")
