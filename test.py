from pyspark.ml.functions import vector_to_array
from pyspark.sql import functions as F

def expand_scaled_features(df, feature_column):
    """
    Expand a feature column (which is a vector) into multiple individual columns.
    
    Parameters:
    df: pyspark.sql.DataFrame - Input DataFrame.
    feature_column: str - The name of the feature column to be expanded (which contains the scaled vector).
    
    Returns:
    pyspark.sql.DataFrame - DataFrame with the vector column expanded into multiple columns.
    """
    # Convert the vector column to an array of values
    df_with_array = df.withColumn(f"{feature_column}_array", vector_to_array(F.col(feature_column)))
    
    # Get the number of features in the vector (length of the array)
    num_features = len(df_with_array.select(f"{feature_column}_array").first()[0])
    
    # Create new columns from the array (each index of the array becomes a new column)
    expanded_columns = []
    for i in range(num_features):
        expanded_columns.append(F.col(f"{feature_column}_array")[i].alias(f"{feature_column}_{i}"))
    
    # Select the original columns (excluding the vector column) and the new expanded columns
    df_expanded = df_with_array.select(*df.columns, *expanded_columns).drop(feature_column, f"{feature_column}_array")
    
    return df_expanded

# Example Usage
# Assuming df has a column called 'scaled_features' that contains the vector of scaled values
df_expanded = expand_scaled_features(df, "scaled_features")
df_expanded.show()
