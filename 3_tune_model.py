import pandas as pd
import optuna
import xgboost as xgb
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from sqlalchemy import create_engine
from db_connect import get_engine
from model_config import categorical_cols, numerical_cols

# Connect to your database
engine = get_engine()
df = pd.read_sql("SELECT * FROM model_feed", engine)

# Drop rows with missing target
df = df.dropna(subset=['target_25'])

# Sort by date to maintain temporal order
df = df.sort_values('Date')

# Train/Test split (time-based)
cutoff_index = int(len(df) * 0.8)
train_df = df.iloc[:cutoff_index]
test_df = df.iloc[cutoff_index:]

# Define features and target
X_train = train_df[categorical_cols + numerical_cols]
y_train = train_df['target_25']
X_test = test_df[categorical_cols + numerical_cols]
y_test = test_df['target_25']

# One-hot encode after split
X_train_encoded = pd.get_dummies(X_train, columns=categorical_cols, drop_first=True)
X_test_encoded = pd.get_dummies(X_test, columns=categorical_cols, drop_first=True)

# Align columns
X_train_encoded, X_test_encoded = X_train_encoded.align(X_test_encoded, join='left', axis=1, fill_value=0)

def objective(trial):
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 10.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 10.0),
        'random_state': 42,
        'use_label_encoder': False
    }

    model = xgb.XGBClassifier(**params)
    model.fit(X_train_encoded, y_train)
    y_pred_proba = model.predict_proba(X_test_encoded)[:, 1]
    return log_loss(y_test, y_pred_proba)

if __name__ == '__main__':
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=50)

    print("\n🎯 Best trial:")
    print(study.best_trial)
    print("\n📌 Best parameters:")
    for key, value in study.best_trial.params.items():
        print(f"{key}: {value}")
