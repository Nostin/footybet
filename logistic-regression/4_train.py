import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, log_loss, classification_report
from sklearn.preprocessing import StandardScaler
from db_connect import get_engine

engine = get_engine()

df_train = pd.read_sql("SELECT * FROM model_feed_train", engine)
df_test = pd.read_sql("SELECT * FROM model_feed_test", engine)

# Drop any leaky fields here (safe AFTER enrichment)
leaky_cols = ['Disposals', 'Kicks', 'Handballs', 'Goals', 'Behinds', 'Time on Ground %', 'Game Result']
df_test.drop(columns=[col for col in leaky_cols if col in df_test.columns], inplace=True)

# Candidate features
candidate_features = [
    'disposals_trend_last_5', 'disposals_delta_5',
    'disposals_max_last_5', 'disposals_min_last_5', 'disposals_std_last_5',
    'days_since_last_game',
    'is_home_game', 'is_away_game',
    'avg_disposals_all', 'avg_team_disposals_last_4'
]

# Only keep features present in both sets
available_features = [f for f in candidate_features if f in df_train.columns and f in df_test.columns]

# Drop NA
df_train = df_train.dropna(subset=available_features + ['target_20'])
df_test = df_test.dropna(subset=available_features)

# Extract features and target
X_train = df_train[available_features]
y_train = df_train['target_20']
X_test = df_test[available_features]

print(f"✅ Using {len(available_features)} features: {available_features}")
print(f"✅ Train shape after NA drop: {X_train.shape}, Test shape: {X_test.shape}")

if X_train.shape[0] == 0 or X_test.shape[0] == 0:
    print("🚫 No usable training or test data. Check feature NA handling.")
    exit()

# Standardize
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train + Predict
model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)
df_test["pred_prob"] = model.predict_proba(X_test_scaled)[:, 1]

# Evaluate if target exists
if "target_20" in df_test.columns:
    y_test = df_test["target_20"]
    auc = roc_auc_score(y_test, df_test["pred_prob"])
    logloss = log_loss(y_test, df_test["pred_prob"])
    preds = (df_test["pred_prob"] >= 0.5).astype(int)

    print(f"\n✅ FINAL EVAL | AUC: {auc:.4f} | Log Loss: {logloss:.4f}\n")
    print("📋 Classification Report:")
    print(classification_report(y_test, preds))
else:
    print("✅ Predictions complete. No test targets to evaluate.")

# Save predictions
df_test[["Date", "Team", "Player", "pred_prob"]].to_sql("logistic_preds", engine, if_exists="replace", index=False)
print("✅ Predictions saved to table: logistic_preds")
