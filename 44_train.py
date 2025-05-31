import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.preprocessing import OneHotEncoder
import joblib
from db_connect import get_engine

# Connect to DB
engine = get_engine()

# Load data
train_df = pd.read_sql('SELECT * FROM model_feed_train', engine)
test_df = pd.read_sql('SELECT * FROM model_feed_test', engine)

# Target
y_train = train_df["target_25"]
y_test = test_df["target_25"]

# Feature columns
categorical_cols = ['Team', 'Opponent', 'Venue', 'timeslot_category']
numerical_cols = [
    'disposals_last_3', 'disposals_last_5',
    'kicks_last_3', 'kicks_last_5',
    'handballs_last_3', 'handballs_last_5',
    'marks_last_3', 'marks_last_5',
    'tackles_last_3', 'tackles_last_5',
    'clearances_last_3', 'clearances_last_5',
    'time_on_ground_%_last_3', 'time_on_ground_%_last_5',
    'disposals_std_last_5',
    'days_since_last_game',
    'opp_avg_disposals_allowed_last_3',
    'is_home_game', 'is_away_game'
]

# Prepare features
X_train = train_df[categorical_cols + numerical_cols]
X_test = test_df[categorical_cols + numerical_cols]

encoder = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
X_train_cat = encoder.fit_transform(X_train[categorical_cols])
X_test_cat = encoder.transform(X_test[categorical_cols])

X_train_encoded = pd.DataFrame(
    np.hstack([X_train[numerical_cols].values, X_train_cat]),
    columns=numerical_cols + encoder.get_feature_names_out(categorical_cols).tolist()
)
X_test_encoded = pd.DataFrame(
    np.hstack([X_test[numerical_cols].values, X_test_cat]),
    columns=numerical_cols + encoder.get_feature_names_out(categorical_cols).tolist()
)
X_test_encoded = X_test_encoded.reindex(columns=X_train_encoded.columns, fill_value=0)

X_train_encoded = X_train_encoded.astype(float)
X_test_encoded = X_test_encoded.astype(float)

# 🔥 Load best params
best_params = joblib.load("optuna_best_params.pkl")  # You must save it in the tuning script like this!

# 👇 Optional: add required fixed values
best_params.update({
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'random_state': 42
})

# Train final model
model = xgb.XGBClassifier(**best_params)
model.fit(X_train_encoded, y_train)

# Predict
y_pred_proba = model.predict_proba(X_test_encoded)[:, 1]

# Evaluate
auc = roc_auc_score(y_test, y_pred_proba)
logloss = log_loss(y_test, y_pred_proba)
print(f"✅ FINAL EVAL | AUC: {auc:.4f} | Log Loss: {logloss:.4f}")

# 💾 Save model
joblib.dump(model, "xgb_model_target25.pkl")
