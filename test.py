from pyspark.ml.feature import StringIndexer, OneHotEncoder, StandardScaler
from pyspark.sql import functions as F
from pyspark.ml import Pipeline

def fit_transform(df, col_map):
    """
    Transforms the DataFrame based on the column transformation map (col_map).
    
    Parameters:
    df: pyspark.sql.DataFrame
    col_map: dict - A dictionary where the keys are column names and the values are transformation types.
        - 'standardize': Apply standard scaling.
        - 'onehot_encode': Apply one-hot encoding.
        - 'no_transform': No transformation (just pass the column).
    
    Returns:
    pyspark.sql.DataFrame: Transformed DataFrame.
    """
    indexers = []
    encoders = []
    scalers = []
    output_columns = []
    
    # Process the col_map to generate the appropriate stages
    for col_name, transform_type in col_map.items():
        if transform_type == 'standardize':
            # For 'standardize', we apply StandardScaler
            scaler = StandardScaler(inputCol=col_name, outputCol=f"{col_name}_scaled")
            scalers.append(scaler)
            output_columns.append(f"{col_name}_scaled")
        
        elif transform_type == 'onehot_encode':
            # For 'onehot_encode', we apply StringIndexer followed by OneHotEncoder
            indexer = StringIndexer(inputCol=col_name, outputCol=f"{col_name}_index")
            encoder = OneHotEncoder(inputCol=f"{col_name}_index", outputCol=f"{col_name}_encoded")
            indexers.append(indexer)
            encoders.append(encoder)
            output_columns.append(f"{col_name}_encoded")
        
        elif transform_type == 'no_transform':
            # For 'no_transform', we do nothing (just pass through the column)
            output_columns.append(col_name)
    
    # Create pipeline with indexers, encoders, and scalers
    stages = indexers + encoders + scalers
    
    # Apply the pipeline to fit and transform the data
    pipeline = Pipeline(stages=stages)
    df_transformed = pipeline.fit(df).transform(df)
    
    # Drop intermediate columns created during transformations
    final_columns = [col for col in df_transformed.columns if col in output_columns]
    return df_transformed.select(*final_columns)

# Example usage
col_map = {
    'num_feature1': 'standardize',
    'num_feature2': 'standardize',
    'cat_feature1': 'onehot_encode',
    'cat_feature2': 'no_transform'
}

df_transformed = fit_transform(df, col_map)
df_transformed.show()
