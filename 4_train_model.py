import pandas as pd
from sqlalchemy import create_engine
from sklearn.metrics import roc_auc_score, log_loss
import xgboost as xgb
from db_connect import get_engine
from model_config import categorical_cols, numerical_cols
import joblib

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

trained_columns = list(X_train_encoded.columns)
joblib.dump(trained_columns, "trained_columns.joblib")

# 🔍 Train/Test overlap check by player-date
train_players = set(train_df['Player'] + "_" + train_df['Date'].astype(str))
test_players = set(test_df['Player'] + "_" + test_df['Date'].astype(str))
overlap = train_players & test_players
print(f"🔍 Train/Test overlap: {len(overlap)} overlapping player-date entries")

# ✅ Train XGBoost model with best Optuna hyperparameters
model = xgb.XGBClassifier(
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

model.fit(X_train_encoded, y_train)

# 📈 Evaluate
y_pred_proba = model.predict_proba(X_test_encoded)[:, 1]
auc = roc_auc_score(y_test, y_pred_proba)
logloss = log_loss(y_test, y_pred_proba)

print(f"✅ AUC: {auc:.4f} | Log Loss: {logloss:.4f}")

# 📊 Feature importances
importances = pd.Series(model.feature_importances_, index=X_train_encoded.columns)
print("\n📊 Top Features:")
print(importances.sort_values(ascending=False).head(20))
