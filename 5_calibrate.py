import pandas as pd
from sqlalchemy import create_engine
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss
from sklearn.calibration import CalibratedClassifierCV
import xgboost as xgb
from db_connect import get_engine
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
import joblib
from sklearn.preprocessing import OneHotEncoder
from model_config import categorical_cols, numerical_cols

# Connect to your database
engine = get_engine()

# Load data
df = pd.read_sql("SELECT * FROM model_feed", engine)
df = df.dropna(subset=['target_25'])

# Sort chronologically
df = df.sort_values('Date')

# Time-based train/test split
cutoff_index = int(len(df) * 0.8)
train_df = df.iloc[:cutoff_index]
test_df = df.iloc[cutoff_index:]

# Split features and target
X_train = train_df[categorical_cols + numerical_cols]
y_train = train_df['target_25']
X_test = test_df[categorical_cols + numerical_cols]
y_test = test_df['target_25']

# One-hot encode after splitting
X_train_encoded = pd.get_dummies(X_train, columns=categorical_cols, drop_first=True)
X_test_encoded = pd.get_dummies(X_test, columns=categorical_cols, drop_first=True)

# Align columns
X_train_encoded, X_test_encoded = X_train_encoded.align(X_test_encoded, join='left', axis=1, fill_value=0)

# 🔍 Train/Test overlap check by player-date
train_players = set(train_df['Player'] + "_" + train_df['Date'].astype(str))
test_players = set(test_df['Player'] + "_" + test_df['Date'].astype(str))
overlap = train_players & test_players
print(f"🔍 Train/Test overlap: {len(overlap)} overlapping player-date entries")

# ✅ Fit base model first to get feature importances
base_model = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    random_state=42,
    max_depth=3,
    learning_rate=0.21004834872098999,
    n_estimators=59,
    subsample=0.8627525692021224,
    colsample_bytree=0.6953413042335951,
    reg_alpha=0.17338469138614787,
    reg_lambda=6.102224578842963
)

base_model.fit(X_train_encoded, y_train)

# 📊 Feature importances
importances = pd.Series(base_model.feature_importances_, index=X_train_encoded.columns)
print("\n📊 Top Features:")
print(importances.sort_values(ascending=False).head(20))

# 📏 Now calibrate using trained model
calibrated_model = CalibratedClassifierCV(base_model, method='isotonic', cv=5)
calibrated_model.fit(X_train_encoded, y_train)

# 📈 Evaluate
y_pred_proba = calibrated_model.predict_proba(X_test_encoded)[:, 1]
auc = roc_auc_score(y_test, y_pred_proba)
logloss = log_loss(y_test, y_pred_proba)
brier = brier_score_loss(y_test, y_pred_proba)

print(f"\n✅ AUC: {auc:.4f} | Log Loss: {logloss:.4f} | Brier Score: {brier:.4f}")

# Manually fit and save the encoder on training categorical data
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
encoder.fit(X_train[categorical_cols])  # use original categorical cols before encoding

# Save fitted model, encoder, and actual trained columns
joblib.dump(calibrated_model, "calibrated_model.joblib")
joblib.dump(encoder, "encoder.joblib")
joblib.dump(X_train_encoded.columns.tolist(), "trained_columns.joblib")


print("💾 Saved calibrated model, encoder, and trained columns")

# 🧪 Calibration Curve
prob_true, prob_pred = calibration_curve(y_test, y_pred_proba, n_bins=10, strategy='uniform')

plt.figure(figsize=(8, 6))
plt.plot(prob_pred, prob_true, marker='o', label='Model')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly Calibrated')
plt.xlabel('Mean Predicted Probability')
plt.ylabel('Fraction of Positives')
plt.title('Calibration Curve')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()