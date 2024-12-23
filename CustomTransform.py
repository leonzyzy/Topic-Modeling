
import os
import pickle
from typing import Dict, Any
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

class DataManipulator:
    """
    A class for preprocessing tabular data, including encoding and saving/loading transformations.

    Attributes:
        encoders (dict): A dictionary to store fitted encoders for each column.
    """

    def __init__(self):
        """Initialize the DataManipulator with an empty dictionary for encoders."""
        self.encoders: Dict[str, Any] = {}
        self.pkl_filepath = "transformations.pkl"  # Default filepath for saving/loading encoders

    def fit_transform(self, train: pd.DataFrame, column_map: Dict[str, str]) -> pd.DataFrame:
        """
        Fit transformations on training data and apply them. Save encoders to a pickle file.

        Args:
            train (pd.DataFrame): The training data to transform.
            column_map (dict): A dictionary mapping column names to transformation types ('label', 'onehot', etc.).

        Returns:
            pd.DataFrame: Transformed training data.
        """
        if os.path.exists(self.pkl_filepath):
            print(f"Loading transformations from {self.pkl_filepath}.")
            with open(self.pkl_filepath, 'rb') as f:
                self.encoders = pickle.load(f)
        else:
            print(f"Fitting new transformations and saving to {self.pkl_filepath}.")
            transformed = train.copy(deep=True)
            for col, transform_type in column_map.items():
                if transform_type == 'label':
                    le = LabelEncoder()
                    transformed[col] = le.fit_transform(transformed[col].astype(str))
                    self.encoders[col] = le
                elif transform_type == 'onehot':
                    ohe = OneHotEncoder(sparse=False, handle_unknown='ignore')
                    ohe_transformed = ohe.fit_transform(transformed[col].values.reshape(-1, 1))
                    ohe_columns = [f"{col}_{category}" for category in ohe.categories_[0]]
                    transformed = pd.concat(
                        [transformed, pd.DataFrame(ohe_transformed, columns=ohe_columns, index=transformed.index)], 
                        axis=1
                    )
                    transformed.drop(columns=[col], inplace=True)
                    self.encoders[col] = ohe
            with open(self.pkl_filepath, 'wb') as f:
                pickle.dump(self.encoders, f)
            return transformed

    def transform(self, test: pd.DataFrame, column_map: Dict[str, str]) -> pd.DataFrame:
        """
        Apply fitted transformations to new data.

        Args:
            test (pd.DataFrame): The test data to transform.
            column_map (dict): A dictionary mapping column names to transformation types ('label', 'onehot', etc.).

        Returns:
            pd.DataFrame: Transformed test data.
        """
        if not os.path.exists(self.pkl_filepath):
            raise FileNotFoundError(f"The specified file does not exist: {self.pkl_filepath}")
        
        print(f"Loading transformations from {self.pkl_filepath}.")
        with open(self.pkl_filepath, 'rb') as f:
            self.encoders = pickle.load(f)
        
        transformed = test.copy(deep=True)
        for col, transform_type in column_map.items():
            encoder = self.encoders.get(col)
            if transform_type == 'label' and isinstance(encoder, LabelEncoder):
                transformed[col] = encoder.transform(transformed[col].astype(str))
            elif transform_type == 'onehot' and isinstance(encoder, OneHotEncoder):
                ohe_transformed = encoder.transform(transformed[col].values.reshape(-1, 1))
                ohe_columns = [f"{col}_{category}" for category in encoder.categories_[0]]
                transformed = pd.concat(
                    [transformed, pd.DataFrame(ohe_transformed, columns=ohe_columns, index=transformed.index)], 
                    axis=1
                )
                transformed.drop(columns=[col], inplace=True)
        return transformed
