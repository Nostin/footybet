import optuna
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OneHotEncoder
from db_connect import get_engine
import joblib

# Connect to DB
engine = get_engine()
train_df = pd.read_sql('SELECT * FROM model_feed_train', engine)
test_df = pd.read_sql('SELECT * FROM model_feed_test', engine)

y_train = train_df["target_25"]
y_test = test_df["target_25"]

categorical_cols = ['Team', 'Opponent', 'Venue', 'timeslot_category']
numerical_cols = [
    'disposals_last_3', 'disposals_last_5',
    'kicks_last_3', 'kicks_last_5',
    'handballs_last_3', 'handballs_last_5',
    'clearances_last_5',
    'time_on_ground_%_last_3', 'time_on_ground_%_last_5',
    'disposals_std_last_5',
    'opp_avg_disposals_allowed_last_3',
    'is_home_game', 'is_away_game', 'form_diff',
    # 'avg_team_disposals_last_4'
]


X_train = train_df[categorical_cols + numerical_cols]
X_test = test_df[categorical_cols + numerical_cols]

encoder = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
X_train_cat = encoder.fit_transform(X_train[categorical_cols])
X_test_cat = encoder.transform(X_test[categorical_cols])

X_train_encoded = pd.DataFrame(
    np.hstack([X_train[numerical_cols].astype(float).values, X_train_cat]),
    columns=numerical_cols + encoder.get_feature_names_out(categorical_cols).tolist()
)
X_test_encoded = pd.DataFrame(
    np.hstack([X_test[numerical_cols].astype(float).values, X_test_cat]),
    columns=numerical_cols + encoder.get_feature_names_out(categorical_cols).tolist()
)
X_test_encoded = X_test_encoded.reindex(columns=X_train_encoded.columns, fill_value=0)

def objective(trial):
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 5),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 5),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 5),
        'random_state': 42
    }

    model = xgb.XGBClassifier(**params)
    model.fit(X_train_encoded, y_train)
    preds = model.predict_proba(X_test_encoded)[:, 1]
    return roc_auc_score(y_test, preds)

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)

# Save best params to file
joblib.dump(study.best_params, "optuna_best_params.pkl")
print("✅ Saved best parameters to optuna_best_params.pkl")

print("✅ Best AUC:", study.best_value)
print("✅ Best Params:", study.best_params)
