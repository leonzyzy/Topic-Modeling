import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, MinMaxScaler, KBinsDiscretizer

class DataTransformer:
    def __init__(self):
        self.encoders = {}  # To store transformers for each column
        self.scalers = {}
        self.column_strategy = {}  # To track transformation strategy
        self.fill_values = {}  # To store fill values for missing data

    def fit_transform(self, train, column_name):
        """
        Fit and transform the train data based on the specified transformation type.
        Args:
            train (pd.DataFrame): The training dataset.
            column_name (dict): Dictionary mapping column names to transformation types.
        Returns:
            pd.DataFrame: Transformed train data.
        """
        self.column_strategy = column_name  # Save the strategies for use in transform
        train_transformed = train.copy()

        for col, strategy in column_name.items():
            # Fill missing values
            if train_transformed[col].isnull().any():
                if train_transformed[col].dtype == 'object':
                    self.fill_values[col] = "NA"
                    train_transformed[col] = train_transformed[col].fillna("NA")
                else:
                    mean_value = train_transformed[col].mean()
                    self.fill_values[col] = mean_value
                    train_transformed[col] = train_transformed[col].fillna(mean_value)
            
            if strategy == "no_transform":
                continue  # No transformation applied

            elif strategy == "label_encode":
                le = LabelEncoder()
                train_transformed[col] = le.fit_transform(train_transformed[col])
                self.encoders[col] = le  # Save encoder

            elif strategy == "onehot_encode":
                ohe = OneHotEncoder(sparse=False, drop='first')
                encoded = ohe.fit_transform(train_transformed[[col]])
                col_names = [f"{col}_{cat}" for cat in ohe.categories_[0][1:]]
                onehot_df = pd.DataFrame(encoded, columns=col_names, index=train_transformed.index)
                train_transformed = train_transformed.drop(columns=[col])
                train_transformed = pd.concat([train_transformed, onehot_df], axis=1)
                self.encoders[col] = ohe

            elif strategy == "standardize":
                scaler = StandardScaler()
                train_transformed[col] = scaler.fit_transform(train_transformed[[col]])
                self.scalers[col] = scaler

            elif strategy == "normalize":
                scaler = MinMaxScaler()
                train_transformed[col] = scaler.fit_transform(train_transformed[[col]])
                self.scalers[col] = scaler

            elif strategy == "discretize":
                kbins = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='uniform')
                train_transformed[col] = kbins.fit_transform(train_transformed[[col]]).astype(int)
                self.encoders[col] = kbins

            else:
                raise ValueError(f"Unknown transformation strategy: {strategy}")

        return train_transformed

    def transform(self, test, column_name=None):
        """
        Transform the test data using previously fitted transformers.
        Args:
            test (pd.DataFrame): The test dataset.
            column_name (dict): Optional, for consistency check with strategies.
        Returns:
            pd.DataFrame: Transformed test data.
        """
        if column_name is not None:
            # Verify that strategies match
            if column_name != self.column_strategy:
                raise ValueError("Column transformation strategy mismatch between train and test.")

        test_transformed = test.copy()

        for col, strategy in self.column_strategy.items():
            # Fill missing values using train fill values
            if test_transformed[col].isnull().any():
                fill_value = self.fill_values.get(col, "NA")
                test_transformed[col] = test_transformed[col].fillna(fill_value)

            if strategy == "no_transform":
                continue

            elif strategy == "label_encode":
                le = self.encoders.get(col)
                if le is not None:
                    test_transformed[col] = le.transform(test_transformed[col])

            elif strategy == "onehot_encode":
                ohe = self.encoders.get(col)
                if ohe is not None:
                    encoded = ohe.transform(test_transformed[[col]])
                    col_names = [f"{col}_{cat}" for cat in ohe.categories_[0][1:]]
                    onehot_df = pd.DataFrame(encoded, columns=col_names, index=test_transformed.index)
                    test_transformed = test_transformed.drop(columns=[col])
                    test_transformed = pd.concat([test_transformed, onehot_df], axis=1)

            elif strategy == "standardize":
                scaler = self.scalers.get(col)
                if scaler is not None:
                    test_transformed[col] = scaler.transform(test_transformed[[col]])

            elif strategy == "normalize":
                scaler = self.scalers.get(col)
                if scaler is not None:
                    test_transformed[col] = scaler.transform(test_transformed[[col]])

            elif strategy == "discretize":
                kbins = self.encoders.get(col)
                if kbins is not None:
                    test_transformed[col] = kbins.transform(test_transformed[[col]]).astype(int)

            else:
                raise ValueError(f"Unknown transformation strategy: {strategy}")

        return test_transformed
