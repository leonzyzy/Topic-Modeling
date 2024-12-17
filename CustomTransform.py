import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, MinMaxScaler

class DataTransformer:
    def __init__(self):
        self.transform_params = {}  # Store transformation details for each column
        self.encoders = {}          # Store encoders or scalers for columns

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
                scaler = StandardScaler()
                transformed_df[col] = scaler.fit_transform(transformed_df[[col]].fillna(0))
                self.encoders[col] = scaler

            elif transform_type == "normalize":
                scaler = MinMaxScaler()
                transformed_df[col] = scaler.fit_transform(transformed_df[[col]].fillna(0))
                self.encoders[col] = scaler

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
                scaler = self.encoders.get(col)
                if not scaler:
                    raise ValueError(f"StandardScaler for column '{col}' not fitted.")
                transformed_df[col] = scaler.transform(transformed_df[[col]].fillna(0))

            elif transform_type == "normalize":
                scaler = self.encoders.get(col)
                if not scaler:
                    raise ValueError(f"MinMaxScaler for column '{col}' not fitted.")
                transformed_df[col] = scaler.transform(transformed_df[[col]].fillna(0))

            else:
                raise ValueError(f"Unknown transformation: {transform_type}")
        
        return transformed_df
