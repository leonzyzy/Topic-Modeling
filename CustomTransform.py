import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.preprocessing import KBinsDiscretizer

class CustomTransformer:
    def __init__(self):
        self.label_encoder = LabelEncoder()
        self.onehot_encoder = OneHotEncoder(sparse=False)
        self.standard_scaler = StandardScaler()
        self.min_max_scaler = MinMaxScaler()
        self.discretizer = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='uniform')
    
    def fit_transform(self, X_train, column_map):
        """
        Fit and transform the training set based on the column_map dictionary.
        
        column_map: Dictionary specifying the transformations for each column.
            Example:
                {
                    'Category': 'label_encode',
                    'Value': 'standardize',
                    'Time': 'cyclical_encode'
                }
        """
        X_train_transformed = X_train.copy()

        for col, transform_type in column_map.items():
            if transform_type == 'label_encode':
                X_train_transformed[col] = self.label_encoder.fit_transform(X_train[col])
            elif transform_type == 'onehot_encode':
                one_hot_encoded = self.onehot_encoder.fit_transform(X_train[[col]])
                one_hot_df = pd.DataFrame(one_hot_encoded, columns=self.onehot_encoder.categories_[0])
                X_train_transformed = pd.concat([X_train_transformed, one_hot_df], axis=1)
                X_train_transformed.drop(col, axis=1, inplace=True)
            elif transform_type == 'standardize':
                X_train_transformed[col] = self.standard_scaler.fit_transform(X_train[[col]])
            elif transform_type == 'normalize':
                X_train_transformed[col] = self.min_max_scaler.fit_transform(X_train[[col]])
            elif transform_type == 'discretize':
                X_train_transformed[col] = self.discretizer.fit_transform(X_train[[col]])
            elif transform_type == 'cyclical_encode':
                # Assuming cyclical encoding applies to columns with numeric values
                X_train_transformed = self.cyclical_encode(X_train_transformed, col)
        
        return X_train_transformed
    
    def transform(self, X_test, column_map):
        """
        Transform the test set based on the fitted transformations from the training set.
        
        column_map: Dictionary specifying the transformations for each column.
        """
        X_test_transformed = X_test.copy()

        for col, transform_type in column_map.items():
            if transform_type == 'label_encode':
                X_test_transformed[col] = self.label_encoder.transform(X_test[col])
            elif transform_type == 'onehot_encode':
                one_hot_encoded = self.onehot_encoder.transform(X_test[[col]])
                one_hot_df = pd.DataFrame(one_hot_encoded, columns=self.onehot_encoder.categories_[0])
                X_test_transformed = pd.concat([X_test_transformed, one_hot_df], axis=1)
                X_test_transformed.drop(col, axis=1, inplace=True)
            elif transform_type == 'standardize':
                X_test_transformed[col] = self.standard_scaler.transform(X_test[[col]])
            elif transform_type == 'normalize':
                X_test_transformed[col] = self.min_max_scaler.transform(X_test[[col]])
            elif transform_type == 'discretize':
                X_test_transformed[col] = self.discretizer.transform(X_test[[col]])
            elif transform_type == 'cyclical_encode':
                X_test_transformed = self.cyclical_encode(X_test_transformed, col)
        
        return X_test_transformed

    def cyclical_encode(self, X, col):
        """
        Encodes cyclical features like time of day, day of week, etc., using sin/cos transformations.
        Assumes the column contains numeric cyclical data (e.g., hours of the day).
        """
        X[col + '_sin'] = np.sin(2 * np.pi * X[col] / X[col].max())
        X[col + '_cos'] = np.cos(2 * np.pi * X[col] / X[col].max())
        X.drop(columns=[col], inplace=True)
        return X


# Example Usage:

# Simulating some data for demonstration:
data = {
    'Category': ['A', 'B', 'A', 'C', 'B'],
