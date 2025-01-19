from pyspark.ml.functions import vector_to_array
from pyspark.sql import functions as F

def expand_scaled_features(df, feature_column, new_feature_name):
    """
    Expand a feature column (which is a vector) into multiple individual columns.
    
    Parameters:
    df: pyspark.sql.DataFrame - Input DataFrame.
    feature_column: str - The name of the feature column to be expanded (which contains the scaled vector).
    new_feature_name: str - The original column name to replace in the expanded feature names.
    
    Returns:
    pyspark.sql.DataFrame - DataFrame with the vector column expanded into multiple columns and original columns dropped.
    """
    # Convert the vector column to an array of values
    df_with_array = df.withColumn(f"{feature_column}_array", vector_to_array(F.col(feature_column)))
    
    # Get the number of features in the vector (length of the array)
    num_features = len(df_with_array.select(f"{feature_column}_array").first()[0])
    
    # Create new columns from the array (each index of the array becomes a new column)
    expanded_columns = []
    for i in range(num_features):
        expanded_columns.append(F.col(f"{feature_column}_array")[i].alias(f"{new_feature_name}_{i}"))
    
    # Select the original columns (excluding the vector column) and the new expanded columns
    df_expanded = df_with_array.select(*df.columns, *expanded_columns).drop(feature_column, f"{feature_column}_array")
    
    return df_expanded


def fit_transform(df, col_map):
    """
    Process columns based on the transformation type (standardize, onehot_encode, or no_transform),
    expand the scaled feature vectors into individual columns, and drop the original columns.
    
    Parameters:
    df: pyspark.sql.DataFrame - Input DataFrame.
    col_map: dict - Dictionary containing column names and their corresponding transformation types.
    
    Returns:
    pyspark.sql.DataFrame - DataFrame with transformed columns.
    """
    for col_name, transform_type in col_map.items():
        if transform_type == 'standardize':
            # Get the new feature name for each transformation (use the original name from col_map)
            df = expand_scaled_features(df, col_name, col_name)
        
        elif transform_type == 'no_transform':
            continue  # No transformation needed
        
        elif transform_type == 'onehot_encode':
            # Placeholder for one-hot encoding logic (you can add StringIndexer, OneHotEncoder here)
            pass
    
    return df

# Example Usage
col_map = {
    "feature1": "standardize",
    "feature2": "standardize",
    "feature3": "no_transform"
}

# Assuming df has columns 'feature1', 'feature2', 'feature3', with feature1 and feature2 being vectors
df_transformed = fit_transform(df, col_map)
df_transformed.show()
