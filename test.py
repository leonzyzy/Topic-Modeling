from pyspark.ml.feature import OneHotEncoder, StringIndexer, StandardScaler
from pyspark.ml import Pipeline
from pyspark.sql import functions as F
from pyspark.sql.types import ArrayType, FloatType
from pyspark.ml.linalg import Vectors
from pyspark.ml import PipelineModel

# Sample function to handle multiple transformations
def fit_transform(df, col_map):
    # Initialize list to store stages of the pipeline
    stages = []
    
    # List to track columns to be dropped after transformation
    drop_columns = []
    
    # Iterate over the columns in the col_map dictionary
    for col_name, transform_type in col_map.items():
        if transform_type == 'standardize':
            # Apply StandardScaler for standardization
            scaler = StandardScaler(inputCol=col_name, outputCol=f"{col_name}_scaled")
            stages.append(scaler)
            drop_columns.append(col_name)
            
        elif transform_type == 'onehot_encode':
            # Apply StringIndexer followed by OneHotEncoder
            indexer = StringIndexer(inputCol=col_name, outputCol=f"{col_name}_index")
            encoder = OneHotEncoder(inputCol=f"{col_name}_index", outputCol=f"{col_name}_onehot")
            stages.extend([indexer, encoder])
            drop_columns.append(col_name)
        
        elif transform_type == 'no_transform':
            # No transformation needed, just keep the column as is
            continue
            
    # Build and fit the pipeline
    pipeline = Pipeline(stages=stages)
    model = pipeline.fit(df)
    transformed_df = model.transform(df)
    
    # Process one-hot encoding to expand into multiple columns
    for col_name, transform_type in col_map.items():
        if transform_type == 'onehot_encode':
            # Convert sparse vector to dense array and expand into multiple columns
            onehot_col = f"{col_name}_onehot"
            transformed_df = transformed_df.withColumn(f"{col_name}_onehot_array", F.col(onehot_col).toArray())
            
            # Get number of categories (length of the one-hot encoded array)
            num_categories = len(transformed_df.select(f"{col_name}_onehot_array").first()[0])
            
            # Expand array into multiple columns for each category
            for i in range(num_categories):
                transformed_df = transformed_df.withColumn(f"{col_name}_category_{i}", F.col(f"{col_name}_onehot_array")[i])
            
            # Drop intermediate columns (index, onehot, array)
            transformed_df = transformed_df.drop(f"{col_name}_index", onehot_col, f"{col_name}_onehot_array")
    
    # Drop original columns for those that were transformed
    transformed_df = transformed_df.drop(*drop_columns)
    
    return transformed_df

# Sample DataFrame
df = spark.createDataFrame([
    (0, "cat", 1.0),
    (1, "dog", 2.0),
    (2, "cat", 3.0),
    (3, "dog", 4.0),
    (4, "bird", 5.0)
], ["id", "animal", "score"])

# Example col_map: 
# - 'animal': onehot_encode, 'score': standardize (apply transformations to these columns)
col_map = {
    "animal": "onehot_encode",
    "score": "standardize"
}

# Apply fit_transform
transformed_df = fit_transform(df, col_map)

# Show result
transformed_df.show(truncate=False)
