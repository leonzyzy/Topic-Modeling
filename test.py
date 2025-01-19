from pyspark.ml.feature import VectorAssembler, StringIndexer, OneHotEncoder
from pyspark.ml import Pipeline
from pyspark.sql.functions import col

def transform_with_vector(df, col_map):
    # Create a list to store the columns that need to be assembled into a vector
    vector_columns = []
    indexers = []
    encoders = []
    
    # Go through the col_map to see which columns need transformation
    for col_name, transform_type in col_map.items():
        if transform_type == 'standardize':  # Numeric column, to be included in vector assembler
            vector_columns.append(col_name)
        elif transform_type == 'onehot_encode':  # Categorical column, to be one-hot encoded
            # StringIndexer to convert categorical values into numeric indices
            indexer = StringIndexer(inputCol=col_name, outputCol=col_name + "_index")
            indexers.append(indexer)
            
            # OneHotEncoder to convert numeric indices into one-hot encoded vectors
            encoder = OneHotEncoder(inputCol=col_name + "_index", outputCol=col_name + "_onehot")
            encoders.append(encoder)
    
    # Apply VectorAssembler if there are any columns to be vectorized
    if vector_columns:
        assembler = VectorAssembler(inputCols=vector_columns, outputCol="features")
    
    # Combine all transformations in a Pipeline
    stages = indexers + encoders
    if vector_columns:
        stages.append(assembler)
    
    pipeline = Pipeline(stages=stages)
    
    # Fit and transform the DataFrame using the pipeline
    df_transformed = pipeline.fit(df).transform(df)

    # The resulting DataFrame will have the "features" column for standardized columns and "_onehot" columns for categorical ones.
    return df_transformed

# Example: Transforming the DataFrame with the col_map
col_map = {
    "accountAge": "standardize",  # Numeric column, will be included in VectorAssembler
    "gender": "onehot_encode",    # Categorical column, will be one-hot encoded
    "name": "no_transform"        # No transformation needed
}

df_transformed = transform_with_vector(df, col_map)

# Check the schema to ensure the vectorized and one-hot encoded columns are created
df_transformed.printSchema()

# Show the transformed DataFrame
df_transformed.show()
