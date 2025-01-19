from pyspark.ml import Pipeline
from pyspark.ml.feature import StandardScaler, OneHotEncoder, StringIndexer
from pyspark.ml.functions import vector_to_array
from pyspark.sql import functions as F

def fit_transform(df, col_map):
    """
    Perform transformations (standardize, one-hot encode, no transform) on the columns specified in col_map
    and return the modified DataFrame with expanded columns replacing original columns.

    Parameters:
    df: pyspark.sql.DataFrame - Input DataFrame.
    col_map: dict - Dictionary where keys are column names and values are transformation types ('standardize', 'onehot_encode', 'no_transform').
    
    Returns:
    pyspark.sql.DataFrame - DataFrame with transformed columns.
    """
    
    # List to store the transformations
    stages = []
    
    for col_name, transform_type in col_map.items():
        if transform_type == 'standardize':
            # StandardScaler requires a vector column, so we need to assemble the features first
            scaler = StandardScaler(inputCol=col_name, outputCol=f"{col_name}_scaled")
            stages.append(scaler)
            
        elif transform_type == 'onehot_encode':
            # StringIndexer and OneHotEncoder will be used to convert categorical values to one-hot encoding
            indexer = StringIndexer(inputCol=col_name, outputCol=f"{col_name}_index")
            encoder = OneHotEncoder(inputCol=f"{col_name}_index", outputCol=f"{col_name}_onehot")
            stages.append(indexer)
            stages.append(encoder)
    
    # Create and fit the pipeline
    pipeline = Pipeline(stages=stages)
    model = pipeline.fit(df)
    transformed_df = model.transform(df)
    
    # Expand the standardized features into individual columns
    for col_name, transform_type in col_map.items():
        if transform_type == 'standardize':
            # Convert the scaled feature vector to array and expand
            transformed_df = expanded_scaled_features(transformed_df, f"{col_name}_scaled")
        
        elif transform_type == 'onehot_encode':
            # Extract and expand the one-hot encoded feature columns
            transformed_df = expanded_onehot_features(transformed_df, f"{col_name}_onehot")

    # Drop the original feature columns and return the transformed dataframe
    for col_name in col_map.keys():
        transformed_df = transformed_df.drop(col_name)
    
    return transformed_df

def expanded_scaled_features(df, feature_column):
    """
    Expand the scaled feature column (vector) into multiple columns and drop the original vector column.
    
    Parameters:
    df: pyspark.sql.DataFrame - Input DataFrame.
    feature_column: str - The name of the scaled feature column.
    
    Returns:
    pyspark.sql.DataFrame - DataFrame with expanded columns from the scaled feature.
    """
    # Convert the vector to array of values
    df_with_array = df.withColumn(f"{feature_column}_array", vector_to_array(F.col(feature_column)))
    
    # Get the number of features in the vector
    num_features = len(df_with_array.select(f"{feature_column}_array").first()[0])
    
    # Create new columns from the array (each index of the array becomes a new column)
    expanded_columns = []
    for i in range(num_features):
        expanded_columns.append(F.col(f"{feature_column}_array")[i].alias(f"{feature_column}_{i}"))
    
    # Select the original columns (excluding the vector column) and the new expanded columns
    df_expanded = df_with_array.select(*df.columns, *expanded_columns).drop(feature_column, f"{feature_column}_array")
    
    return df_expanded

def expanded_onehot_features(df, feature_column):
    """
    Expands the one-hot encoded vector into multiple columns and drops the original vector column.
    
    Parameters:
    df: pyspark.sql.DataFrame - Input DataFrame.
    feature_column: str - The name of the one-hot encoded feature column.
    
    Returns:
    pyspark.sql.DataFrame - DataFrame with expanded one-hot encoded columns.
    """
    # Get the length of the one-hot encoded vector
    df_with_array = df.withColumn(f"{feature_column}_array", vector_to_array(F.col(feature_column)))
    
    # Get the number of features (size of the one-hot vector)
    num_features = len(df_with_array.select(f"{feature_column}_array").first()[0])
    
    # Create new columns from the array (each index of the array becomes a new column)
    expanded_columns = []
    for i in range(num_features):
        expanded_columns.append(F.col(f"{feature_column}_array")[i].alias(f"{feature_column}_{i}"))
    
    # Select the original columns (excluding the vector column) and the new expanded columns
    df_expanded = df_with_array.select(*df.columns, *expanded_columns).drop(feature_column, f"{feature_column}_array")
    
    return df_expanded

# Example Usage:
col_map = {
    "age": "standardize",
    "category": "onehot_encode",
    "income": "no_transform"
}

df_transformed = fit_transform(df, col_map)
df_transformed.show()
