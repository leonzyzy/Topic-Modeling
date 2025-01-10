import os
import pickle
from typing import Dict
from pyspark.sql import DataFrame
from pyspark.ml.feature import StandardScaler, OneHotEncoder, StringIndexer, VectorAssembler
from pyspark.ml import Pipeline
from pyspark.sql.functions import col, unix_timestamp
from pyspark.ml.linalg import Vectors

class DataTransformer:
    def __init__(self, pkl_file: str = 'preprocessor.pkl'):
        self.pkl_file = pkl_file
        self.encoder = {}  # Dictionary to store encoders for columns

    def fit_transform(self, train: DataFrame, column_map: Dict[str, str]) -> DataFrame:
        """
        Fit transformers on training data and apply the transformations.
        """
        # Check if preprocessor.pkl exists
        if os.path.exists(self.pkl_file):
            with open(self.pkl_file, 'rb') as f:
                self.encoder = pickle.load(f)
            print("Loaded existing preprocessor from pkl.")
            return self._apply_transformations(train, column_map)
        else:
            print("Fitting new encoders and transformations.")
            # Initialize encoders based on column_map
            for col_name, transform_type in column_map.items():
                if transform_type == "standardize":
                    self.encoder[col_name] = self._create_standardizer(train, col_name)
                elif transform_type == "onehot_encode":
                    self.encoder[col_name] = self._create_onehot_encoder(train, col_name)
                elif transform_type == "timestamp_encode":
                    self.encoder[col_name] = self._create_timestamp_encoder(train, col_name)
                # For "no_transform", we do nothing (this will be handled in transform function)
            
            # Save the encoders to a file
            with open(self.pkl_file, 'wb') as f:
                pickle.dump(self.encoder, f)
            
            return self._apply_transformations(train, column_map)

    def transform(self, test: DataFrame, column_map: Dict[str, str]) -> DataFrame:
        """
        Apply saved transformations to the test data.
        """
        if os.path.exists(self.pkl_file):
            with open(self.pkl_file, 'rb') as f:
                self.encoder = pickle.load(f)
            print("Loaded existing preprocessor from pkl.")
            return self._apply_transformations(test, column_map)
        else:
            raise FileNotFoundError(f"{self.pkl_file} not found. Please fit the model first.")

    def _apply_transformations(self, df: DataFrame, column_map: Dict[str, str]) -> DataFrame:
        """
        Apply the transformations to the DataFrame based on the given column_map.
        """
        for col_name, transform_type in column_map.items():
            if transform_type == "standardize" and col_name in self.encoder:
                df = self.encoder[col_name].transform(df)
            elif transform_type == "onehot_encode" and col_name in self.encoder:
                df = self.encoder[col_name].transform(df)
            elif transform_type == "timestamp_encode" and col_name in self.encoder:
                df = self.encoder[col_name].transform(df)
            # For "no_transform", we do nothing

        return df

    def _create_standardizer(self, df: DataFrame, col_name: str):
        """
        Create and fit a standardizer (StandardScaler) for a given column.
        """
        assembler = VectorAssembler(inputCols=[col_name], outputCol=f"{col_name}_vec")
        df = assembler.transform(df)
        scaler = StandardScaler(inputCol=f"{col_name}_vec", outputCol=f"{col_name}_scaled")
        scaler_model = scaler.fit(df)
        return scaler_model

    def _create_onehot_encoder(self, df: DataFrame, col_name: str):
        """
        Create and fit a one-hot encoder for a given column.
        """
        indexer = StringIndexer(inputCol=col_name, outputCol=f"{col_name}_index")
        indexed_df = indexer.fit(df).transform(df)
        encoder = OneHotEncoder(inputCol=f"{col_name}_index", outputCol=f"{col_name}_onehot")
        encoder_model = encoder.fit(indexed_df)
        return encoder_model

    def _create_timestamp_encoder(self, df: DataFrame, col_name: str):
        """
        Create and fit a timestamp encoder for a given column.
        """
        return df.withColumn(f"{col_name}_encoded", unix_timestamp(col(col_name), 'yyyy-MM-dd HH:mm:ss').cast("double"))
