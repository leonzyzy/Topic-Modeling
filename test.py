import xgboost as xgb
from sklearn.metrics import make_scorer, f1_score
from sklearn.model_selection import GridSearchCV

# Assuming train, val, and test datasets are already split
# X_train, y_train -> Training data
# X_val, y_val -> Validation data (for hyperparameter tuning)
# X_test, y_test -> Test data (for final evaluation)

# Define the parameter grid
param_grid = {
    "max_depth": [4, 6, 8],
    "learning_rate": [0.01, 0.1, 0.2],
    "n_estimators": [50, 100, 150],
    "subsample": [0.6, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
}

# Custom scorer for F1-Score
f1_scorer = make_scorer(f1_score)

# Create an XGBoost classifier
model = xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42)

# GridSearchCV with validation set (no cross-validation)
grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    scoring=f1_scorer,
    cv=[(slice(None), slice(None))],  # Trick to skip cross-validation
    verbose=2,
    n_jobs=-1,
)

# Fit the grid search on the training set
grid_search.fit(X_train, y_train)

# Evaluate on the validation set
y_val_pred = grid_search.best_estimator_.predict(X_val)
val_f1 = f1_score(y_val, y_val_pred)

print("\nBest Parameters:", grid_search.best_params_)
print("Best Validation F1-Score:", val_f1)

# Train the final model on the training set with the best parameters
final_model = xgb.XGBClassifier(
    **grid_search.best_params_,
    use_label_encoder=False,
    eval_metric="logloss",
    random_state=42,
)
final_model.fit(X_train, y_train)

# Evaluate the final model on the test set
y_test_pred = final_model.predict(X_test)

# Compute test metrics
test_f1 = f1_score(y_test, y_test_pred)
print("\nTest F1 Score:", test_f1)
