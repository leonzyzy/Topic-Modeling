import pandas as pd
import numpy as np

# Sample feature DataFrame (transaction data)
df_features = pd.DataFrame({
    "account_id": [1, 1, 2, 2, 3, 3],
    "feature1": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
    "feature2": [1.1, 1.2, 1.3, 1.4, 1.5, 1.6]
})

# Sample target DataFrame (may have different account IDs)
df_target = pd.DataFrame({
    "account_id": [1, 2, 3],  # Only one target per account
    "target": [0, 1, 0]
})

# Convert features into a dictionary
features_dict = (
    df_features.groupby("account_id")
    .apply(lambda x: np.array(x.drop(columns=["account_id"])))
    .to_dict()
)

# Convert targets into a dictionary
target_dict = df_target.set_index("account_id")["target"].to_dict()

# Combine into final dictionary
final_dict = {
    account_id: {
        "features": [features_dict[account_id]] if account_id in features_dict else [],
        "target": [np.array([target_dict[account_id]])] if account_id in target_dict else []
    }
    for account_id in set(features_dict.keys()).union(target_dict.keys())
}

# Output result
print(final_dict)
