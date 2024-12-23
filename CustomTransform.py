
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

def timestamp_decompose(self, series: pd.Series, column: str) -> pd.DataFrame:
    """
    Decomposes a pandas Series of timestamps into separate components (year, month, day, etc.).

    Args:
        series (pd.Series): A pandas Series containing datetime-like objects.
        column (str): The base column name for the decomposed components.

    Returns:
        pd.DataFrame: A DataFrame containing the decomposed components as separate columns.
    """
    if not pd.api.types.is_datetime64_any_dtype(series):
        raise ValueError("The input series must have a datetime64 dtype.")

    # Decomposing the datetime series into separate components
    decomposed = pd.DataFrame()
    decomposed[f"{column}_year"] = series.dt.year
    decomposed[f"{column}_month"] = series.dt.month
    decomposed[f"{column}_day"] = series.dt.day
    decomposed[f"{column}_hour"] = series.dt.hour
    decomposed[f"{column}_minute"] = series.dt.minute
    decomposed[f"{column}_second"] = series.dt.second

    return decomposed

def timestamp_encode(self, series: pd.Series, column: str, freq: str = "D") -> pd.DataFrame:
    """
    Encodes a pandas Series of timestamps into cyclic features (sin and cos) based on a given frequency.

    Args:
        series (pd.Series): A pandas Series containing datetime-like objects.
        column (str): The base column name for the encoded features.
        freq (str): The frequency to encode the timestamps. Supported values are:
            - "D" for days in a year (cyclic encoding for day-of-year).
            - "M" for months in a year (cyclic encoding for month-of-year).
            - "W" for days in a week (cyclic encoding for day-of-week).

    Returns:
        pd.DataFrame: A DataFrame containing cyclically encoded features (sin and cos) for the specified frequency.

    Raises:
        ValueError: If the input `series` is not of datetime64 dtype or if the `freq` is invalid.
    """
    if not pd.api.types.is_datetime64_any_dtype(series):
        raise ValueError("The input series must have a datetime64 dtype.")
    
    encoded = pd.DataFrame()

    if freq == "D":  # Days in a year
        days_in_year = 365.25
        day_of_year = series.dt.dayofyear
        encoded[f"{column}_sin_day"] = np.sin(2 * np.pi * day_of_year / days_in_year)
        encoded[f"{column}_cos_day"] = np.cos(2 * np.pi * day_of_year / days_in_year)

    elif freq == "M":  # Months in a year

def cyclical_encode(self, series: pd.Series, column: str, max_value: int = None) -> pd.DataFrame:
    """
    Encodes a pandas Series of cyclic values into sine and cosine components.

    Args:
        series (pd.Series): A pandas Series containing numerical or categorical cyclic values (e.g., hours, months, or days).
        column (str): The base column name for the encoded features.
        max_value (int, optional): The maximum value of the cyclic feature. If None, it will be inferred from the data (series.max() + 1).

    Returns:
        pd.DataFrame: A DataFrame containing sine and cosine encoded features for the cyclic values.

    Raises:
        ValueError: If the input `series` is not numeric or if `max_value` is not specified and cannot be inferred.
    """
    if not pd.api.types.is_numeric_dtype(series):
        raise ValueError("The input series must contain numeric data.")

    if max_value is None:
        max_value = series.max() + 1
        if max_value <= 1:
            raise ValueError("Invalid max_value. It must be greater than 1 or properly inferred.")

    encoded = pd.DataFrame()
 
