import xgboost as xgb
from sklearn.metrics import f1_score, accuracy_score
import numpy as np

# Assuming train, val, and test datasets are already split
# X_train, y_train -> Training data
# X_val, y_val -> Validation data (for hyperparameter tuning)
# X_test, y_test -> Test data (for final evaluation)

# Define a list of hyperparameter combinations to evaluate (linear search)
hyperparameter_combinations = [
    {"max_depth": 4, "learning_rate": 0.01, "n_estimators": 100, "subsample": 0.8, "colsample_bytree": 0.8},
    {"max_depth": 6, "learning_rate": 0.1, "n_estimators": 150, "subsample": 1.0, "colsample_bytree": 0.6},
    {"max_depth": 8, "learning_rate": 0.2, "n_estimators": 200, "subsample": 0.6, "colsample_bytree": 1.0},
    {"max_depth": 4, "learning_rate": 0.1, "n_estimators": 50, "subsample": 0.8, "colsample_bytree": 0.8},
    {"max_depth": 6, "learning_rate": 0.05, "n_estimators": 100, "subsample": 0.9, "colsample_bytree": 0.7},
]

# Initialize variables to store the best configuration and its score
best_params = None
best_score = float("-inf")

# Linear search over the predefined hyperparameter combinations
for params in hyperparameter_combinations:
    # Train the model with the current hyperparameters
    model = xgb.XGBClassifier(
        max_depth=params["max_depth"],
        learning_rate=params["learning_rate"],
        n_estimators=params["n_estimators"],
        subsample=params["subsample"],
        colsample_bytree=params["colsample_bytree"],
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
    )
    
    # Fit the model on the training data
    model.fit(X_train, y_train)
    
    # Predict on the validation set
    y_val_pred = model.predict(X_val)
    
    # Compute F1-Score for validation
    val_score = f1_score(y_val, y_val_pred)  # Using F1 as the metric
    
    print(f"Params: {params}, Validation F1-Score: {val_score}")
    
    # Update the best parameters if the current model is better
    if val_score > best_score:
        best_score = val_score
        best_params = params

print("\nBest Parameters:", best_params)
print("Best Validation Score (F1):", best_score)

# Train the final model on train + val data with the best parameters
final_model = xgb.XGBClassifier(
    **best_params,
    use_label_encoder=False,
    eval_metric="logloss",
    random_state=42,
)
final_model.fit(np.concatenate([X_train, X_val]), np.concatenate([y_train, y_val]))

# Evaluate the final model on the test set
y_test_pred = final_model.predict(X_test)

# Compute test metrics
test_f1 = f1_score(y_test, y_test_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

print("\nTest Accuracy:", test_accuracy)
print("Test F1 Score:", test_f1)
