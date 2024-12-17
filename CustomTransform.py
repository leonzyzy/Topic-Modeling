import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, MinMaxScaler, KBinsDiscretizer

class ColumnTransformer:
    def __init__(self, transformations):
        """
        Initialize with a dictionary of column transformations.
        :param transformations: dict, key=column_name, value=transformation type
               Supported types: "no_transform", "label_encode", "onehot_encode", "standarize", "normalize", "discretize"
        """
        self.transformations = transformations
        self.encoders = {}
        self.scalers = {}

    def fit_transform(self, trainset, column_name):
        """
        Fit and transform the specified column in the trainset.
        :param trainset: pd.DataFrame, the training dataset
        :param column_name: str, the column to be transformed
        :return: Transformed trainset with the specified column
        """
        if column_name not in trainset.columns:
            raise ValueError(f"Column '{column_name}' not found in the dataset.")

        transformation = self.transformations.get(column_name, "no_transform")
        
        if transformation == "no_transform":
            # Do nothing
            return trainset

        elif transformation == "label_encode":
            le = LabelEncoder()
            trainset[column_name] = le.fit_transform(trainset[column_name].astype(str))
            self.encoders[column_name] = le

        elif transformation == "onehot_encode":
            ohe = OneHotEncoder(sparse=False, drop='first')  # drop_first to avoid dummy variable trap
            transformed = ohe.fit_transform(trainset[[column_name]])
            transformed_df = pd.DataFrame(transformed, 
                                          columns=[f"{column_name}_{i}" for i in range(transformed.shape[1])])
            self.encoders[column_name] = ohe
            trainset = trainset.drop(column_name, axis=1).reset_index(drop=True)
            trainset = pd.concat([trainset, transformed_df], axis=1)

        elif transformation == "standarize":
            scaler = StandardScaler()
            trainset[column_name] = scaler.fit_transform(trainset[[column_name]])
            self.scalers[column_name] = scaler

        elif transformation == "normalize":
            scaler = MinMaxScaler()
            trainset[column_name] = scaler.fit_transform(trainset[[column_name]])
            self.scalers[column_name] = scaler

        elif transformation == "discretize":
            kb = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='uniform')
            trainset[column_name] = kb.fit_transform(trainset[[column_name]]).astype(int)
            self.scalers[column_name] = kb
        
        else:
            raise ValueError(f"Transformation '{transformation}' is not supported.")

        return trainset

    def transform(self, testset, column_name):
        """
        Transform the specified column in the testset using fitted encoders/scalers.
        :param testset: pd.DataFrame, the test dataset
        :param column_name: str, the column to be transformed
        :return: Transformed testset with the specified column
        """
        if column_name not in testset.columns:
            raise ValueError(f"Column '{column_name}' not found in the dataset.")

        transformation = self.transformations.get(column_name, "no_transform")
        
        if transformation == "no_transform":
            return testset

        elif transformation == "label_encode":
            le = self.encoders.get(column_name)
            if le is None:
                raise ValueError(f"LabelEncoder for '{column_name}' is not fitted.")
            testset[column_name] = le.transform(testset[column_name].astype(str))

        elif transformation == "onehot_encode":
            ohe = self.encoders.get(column_name)
            if ohe is None:
                raise ValueError(f"OneHotEncoder for '{column_name}' is not fitted.")
            transformed = ohe.transform(testset[[column_name]])
            transformed_df = pd.DataFrame(transformed, 
                                          columns=[f"{column_name}_{i}" for i in range(transformed.shape[1])])
            testset = testset.drop(column_name, axis=1).reset_index(drop=True)
            testset = pd.concat([testset, transformed_df], axis=1)

        elif transformation == "standarize" or transformation == "normalize":
            scaler = self.scalers.get(column_name)
            if scaler is None:
                raise ValueError(f"Scaler for '{column_name}' is not fitted.")
            testset[column_name] = scaler.transform(testset[[column_name]])

        elif transformation == "discretize":
            kb = self.scalers.get(column_name)
            if kb is None:
                raise ValueError(f"KBinsDiscretizer for '{column_name}' is not fitted.")
            testset[column_name] = kb.transform(testset[[column_name]]).astype(int)
        
        else:
            raise ValueError(f"Transformation '{transformation}' is not supported.")

        return testset

# Example usage:
if __name__ == "__main__":
    # Sample dataset
    train = pd.DataFrame({
        'A': ['cat', 'dog', 'mouse'],
        'B': [1.0, 2.5, 3.7],
        'C': [10, 20, 30]
    })
    test = pd.DataFrame({
        'A': ['dog', 'mouse', 'cat'],
        'B': [2.0, 3.5, 1.5],
        'C': [15, 25, 35]
    })

    transformations = {
        'A': 'label_encode',
        'B': 'standarize',
        'C': 'discretize'
    }

    transformer = ColumnTransformer(transformations)

    # Fit and transform trainset
    print("Trainset Before Transformation:")
    print(train)
    train = transformer.fit_transform(train, 'A')
    train = transformer.fit_transform(train, 'B')
    train = transformer.fit_transform(train, 'C')
    print("\nTrainset After Transformation:")
    print(train)

    # Transform testset
    print("\nTestset Before Transformation:")
    print(test)
    test = transformer.transform(test, 'A')
    test = transformer.transform(test, 'B')
    test = transformer.transform(test, 'C')
    print("\nTestset After Transformation:")
    print(test)
