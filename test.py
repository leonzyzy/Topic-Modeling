from pyspark.ml.functions import vector_to_array
from pyspark.sql import functions as F

def expand_feature_column(df, feature_column):
    """
    Expands a feature vector column into multiple individual columns.
    Drops the original feature vector column.

    Parameters:
    df: pyspark.sql.DataFrame - Input DataFrame with a vector column.
    feature_column: str - The name of the feature column that contains a vector.
    
    Returns:
    pyspark.sql.DataFrame - DataFrame with expanded columns and original vector column dropped.
    """
    # Convert the vector to an array of values
    df_with_array = df.withColumn(f"{feature_column}_array", vector_to_array(F.col(feature_column)))
    
    # Get the number of features in the vector
    num_features = len(df_with_array.select(f"{feature_column}_array").first()[0])  # Getting the length of the first row's array
    
    # Create new columns from the array (each index of the array becomes a new column)
    expanded_columns = []
    for i in range(num_features):
        expanded_columns.append(F.col(f"{feature_column}_array")[i].alias(f"{feature_column}_{i}"))
    
    # Select the original columns (excluding the vector column) and the new expanded columns
    df_expanded = df_with_array.select(*df.columns, *expanded_columns).drop(feature_column, f"{feature_column}_array")
    
    return df_expanded

# Example usage
df_expanded = expand_feature_column(df, 'scaled_feature')
df_expanded.show()
