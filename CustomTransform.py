import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, MinMaxScaler, KBinsDiscretizer

class DataTransformer:
    def __init__(self):
        self.encoders = {}  # To store encoders/scalers for each column
        
    def fit_transform(self, train, column_name):
        transformed_train = train.copy()
        
        for col, transform_type in column_name.items():
            if transform_type == "no_transform":
                continue  # Do nothing
            
            elif transform_type == "label_encode":
                encoder = LabelEncoder()
                transformed_train[col] = encoder.fit_transform(train[col])
                self.encoders[col] = encoder  # Save encoder
                
            elif transform_type == "onehot_encode":
                encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")
                transformed_data = encoder.fit_transform(train[[col]])
                transformed_train = self._replace_column_with_encoded_data(transformed_train, col, transformed_data, encoder)
                self.encoders[col] = encoder
                
            elif transform_type == "standarize":
                scaler = StandardScaler()
                transformed_train[col] = scaler.fit_transform(train[[col]])
                self.encoders[col] = scaler  # Save scaler
                
            elif transform_type == "normalize":
                scaler = MinMaxScaler()
                transformed_train[col] = scaler.fit_transform(train[[col]])
                self.encoders[col] = scaler  # Save scaler
                
            elif transform_type == "discretize":
                discretizer = KBinsDiscretizer(n_bins=5, encode="ordinal", strategy="uniform")
                transformed_train[col] = discretizer.fit_transform(train[[col]])
                self.encoders[col] = discretizer  # Save discretizer
                
            else:
                raise ValueError(f"Unknown transform type: {transform_type}")
                
        return transformed_train
    
    def transform(self, test, column_name):
        transformed_test = test.copy()
        
        for col, transform_type in column_name.items():
            if transform_type == "no_transform":
                continue  # Do nothing
            
            elif transform_type == "label_encode":
                encoder = self.encoders.get(col)
                if encoder:
                    transformed_test[col] = encoder.transform(test[col])
                else:
                    raise ValueError(f"No encoder found for column {col}")
                    
            elif transform_type == "onehot_encode":
                encoder = self.encoders.get(col)
                if encoder:
                    transformed_data = encoder.transform(test[[col]])
                    transformed_test = self._replace_column_with_encoded_data(transformed_test, col, transformed_data, encoder)
                else:
                    raise ValueError(f"No encoder found for column {col}")
                    
            elif transform_type == "standarize":
                scaler = self.encoders.get(col)
                if scaler:
                    transformed_test[col] = scaler.transform(test[[col]])
                else:
                    raise ValueError(f"No scaler found for column {col}")
                    
            elif transform_type == "normalize":
                scaler = self.encoders.get(col)
                if scaler:
                    transformed_test[col] = scaler.transform(test[[col]])
                else:
                    raise ValueError(f"No scaler found for column {col}")
                    
            elif transform_type == "discretize":
                discretizer = self.encoders.get(col)
                if discretizer:
                    transformed_test[col] = discretizer.transform(test[[col]])
                else:
                    raise ValueError(f"No discretizer found for column {col}")
                    
            else:
                raise ValueError(f"Unknown transform type: {transform_type}")
                
        return transformed_test

    def _replace_column_with_encoded_data(self, df, col, transformed_data, encoder):
        """Helper function to replace a column with encoded data for one-hot encoding."""
        encoded_cols = [f"{col}_{category}" for category in encoder.categories_[0]]
        transformed_df = pd.DataFrame(transformed_data, columns=encoded_cols, index=df.index)
        df = df.drop(columns=[col])
        return pd.concat([df, transformed_df], axis=1)
