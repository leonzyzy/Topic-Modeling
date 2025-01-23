import xgboost as xgb
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import numpy as np

# Assuming train, val, and test datasets are already split
# X_train, y_train -> Training data
# X_val, y_val -> Validation data (for hyperparameter tuning)
# X_test, y_test -> Test data (for final evaluation)

# Define the parameter grid
param_grid = {
    "max_depth": [4, 6, 8],
    "learning_rate": [0.01, 0.1, 0.2],
    "n_estimators": [50, 100, 200],
    "subsample": [0.6, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
}

# Initialize variables to store the best parameters and score
best_params = None
best_score = float("-inf")

# Perform grid search manually (no CV)
for max_depth in param_grid["max_depth"]:
    for learning_rate in param_grid["learning_rate"]:
        for n_estimators in param_grid["n_estimators"]:
            for subsample in param_grid["subsample"]:
                for colsample_bytree in param_grid["colsample_bytree"]:
                    # Train the model with the current set of hyperparameters
                    model = xgb.XGBClassifier(
                        max_depth=max_depth,
                        learning_rate=learning_rate,
                        n_estimators=n_estimators,
                        subsample=subsample,
                        colsample_bytree=colsample_bytree,
                        use_label_encoder=False,
                        eval_metric="logloss",
                        random_state=42,
                    )
                    
                    model.fit(X_train, y_train)  # Train on the training set
                    
                    # Evaluate on the validation set
                    y_val_pred = model.predict_proba(X_val)[:, 1]
                    val_score = roc_auc_score(y_val, y_val_pred)  # Using ROC AUC as the metric
                    
                    # Update best parameters if current score is better
                    if val_score > best_score:
                        best_score = val_score
                        best_params = {
                            "max_depth": max_depth,
                            "learning_rate": learning_rate,
                            "n_estimators": n_estimators,
                            "subsample": subsample,
                            "colsample_bytree": colsample_bytree,
                        }

print("Best Parameters:", best_params)
print("Best Validation Score (AUC):", best_score)

# Train the final model on train + val data using the best parameters
final_model = xgb.XGBClassifier(
    **best_params,
    use_label_encoder=False,
    eval_metric="logloss",
    random_state=42,
)
final_model.fit(np.concatenate([X_train, X_val]), np.concatenate([y_train, y_val]))

# Evaluate the final model on the test set
y_test_pred = final_model.predict(X_test)
y_test_proba = final_model.predict_proba(X_test)[:, 1]

# Metrics
accuracy = accuracy_score(y_test, y_test_pred)
f1 = f1_score(y_test, y_test_pred)
roc_auc = roc_auc_score(y_test, y_test_proba)

print("Test Accuracy:", accuracy)
print("Test F1 Score:", f1)
print("Test ROC AUC:", roc_auc)
