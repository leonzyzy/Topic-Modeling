import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, MinMaxScaler, KBinsDiscretizer

class DataTransformer:
    def __init__(self):
        self.transform_params = {}  # Store transformation details for each column
        self.encoders = {}          # Store encoders or scalers for columns
        self.means = {}             # Store mean values for columns

    def fit_transform(self, train_set, column_name):
        """
        Fits and applies transformations on the train set.

        Parameters:
        - train_set: pd.DataFrame
        - column_name: dict, key = column, value = transform type

        Returns:
        - Transformed train set (pd.DataFrame)
        """
        transformed_df = train_set.copy()

        # Apply transformations based on column_name and drop 'timestamp_decompose' and 'timestamp_encode' columns
        columns_to_drop = []
        
        for col, transform_type in column_name.items():
            if transform_type == 'timestamp_decompose':
                transformed_df = pd.concat([transformed_df, self.timestamp_decompose(transformed_df[col])], axis=1)
                columns_to_drop.append(col)  # Drop the original timestamp column
            elif transform_type == 'timestamp_encode':
                transformed_df = pd.concat([transformed_df, self.timestamp_encode(transformed_df[col])], axis=1)
                columns_to_drop.append(col)  # Drop the original timestamp column

        # Drop timestamp columns after transformation
        transformed_df.drop(columns=columns_to_drop, inplace=True)

        for col, transform_type in column_name.items():
            if transform_type == "no_transform":
                continue

            elif transform_type == "label_encode":
                le = LabelEncoder()
                transformed_df[col] = le.fit_transform(transformed_df[col].fillna("NA"))
                self.encoders[col] = le

            elif transform_type == "onehot_encode":
                ohe = OneHotEncoder(sparse=False, handle_unknown='ignore')
                ohe_result = ohe.fit_transform(transformed_df[[col]].fillna("NA"))
                ohe_columns = [f"{col}_{cat}" for cat in ohe.categories_[0]]
                ohe_df = pd.DataFrame(ohe_result, columns=ohe_columns, index=train_set.index)
                transformed_df = pd.concat([transformed_df.drop(columns=[col]), ohe_df], axis=1)
                self.encoders[col] = ohe

            elif transform_type == "standardize":
                # Fill missing values with column mean before scaling
                col_mean = transformed_df[col].mean()
                transformed_df[col].fillna(col_mean, inplace=True)
                scaler = StandardScaler()
                transformed_df[col] = scaler.fit_transform(transformed_df[[col]])
                self.encoders[col] = scaler
                self.means[col] = col_mean

            elif transform_type == "normalize":
                # Fill missing values with column mean before scaling
                col_mean = transformed_df[col].mean()
                transformed_df[col].fillna(col_mean, inplace=True)
                scaler = MinMaxScaler()
                transformed_df[col] = scaler.fit_transform(transformed_df[[col]])
                self.encoders[col] = scaler
                self.means[col] = col_mean

            elif transform_type == "discretize":
                discretizer = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile')
                transformed_df[col] = discretizer.fit_transform(transformed_df[[col]].fillna(0)).astype(int)
                self.encoders[col] = discretizer

            else:
                raise ValueError(f"Unknown transformation: {transform_type}")
        
        return transformed_df

    def transform(self, test_set, column_name):
        """
        Applies transformations to the test set using saved parameters.

        Parameters:
        - test_set: pd.DataFrame
        - column_name: dict, key = column, value = transform type

        Returns:
        - Transformed test set (pd.DataFrame)
        """
        transformed_df = test_set.copy()

        for col, transform_type in column_name.items():
            if transform_type == "no_transform":
                continue

            elif transform_type == "label_encode":
                le = self.encoders.get(col)
                if not le:
                    raise ValueError(f"LabelEncoder for column '{col}' not fitted.")
                transformed_df[col] = le.transform(transformed_df[col].fillna("NA"))

            elif transform_type == "onehot_encode":
                ohe = self.encoders.get(col)
                if not ohe:
                    raise ValueError(f"OneHotEncoder for column '{col}' not fitted.")
                ohe_result = ohe.transform(transformed_df[[col]].fillna("NA"))
                ohe_columns = [f"{col}_{cat}" for cat in ohe.categories_[0]]
                ohe_df = pd.DataFrame(ohe_result, columns=ohe_columns, index=test_set.index)
                transformed_df = pd.concat([transformed_df.drop(columns=[col]), ohe_df], axis=1)

            elif transform_type == "standardize":
                # Use train mean to fill missing values
                col_mean = self.means.get(col)
                if col_mean is None:
                    raise ValueError(f"StandardScaler for column '{col}' not fitted.")
                transformed_df[col].fillna(col_mean, inplace=True)
                scaler = self.encoders.get(col)
                if not scaler:
                    raise ValueError(f"StandardScaler for column '{col}' not fitted.")
                transformed_df[col] = scaler.transform(transformed_df[[col]])

            elif transform_type == "normalize":
                # Use train mean to fill missing values
                col_mean = self.means.get(col)
                if col_mean is None:
                    raise ValueError(f"MinMaxScaler for column '{col}' not fitted.")
                transformed_df[col].fillna(col_mean, inplace=True)
                scaler = self.encoders.get(col)
                if not scaler:
                    raise ValueError(f"MinMaxScaler for column '{col}' not fitted.")
                transformed_df[col] = scaler.transform(transformed_df[[col]])

            elif transform_type == "discretize":
                discretizer = self.encoders.get(col)
                if not discretizer:
                    raise ValueError(f"Discretizer for column '{col}' not fitted.")
                transformed_df[col] = discretizer.transform(transformed_df[[col]].fillna(0)).astype(int)

            else:
                raise ValueError(f"Unknown transformation: {transform_type}")
        
        return transformed_df
