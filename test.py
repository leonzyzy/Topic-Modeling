from pyspark.ml.feature import VectorAssembler
from pyspark.ml import Pipeline

def transform_with_vector(df, col_map):
    # Create a list to store the columns that need to be assembled into a vector
    vector_columns = []

    # Go through the col_map to see which columns need transformation
    for col_name, transform_type in col_map.items():
        if transform_type == 'standardize':  # Assume this column needs to be in vector format
            vector_columns.append(col_name)
    
    # Apply VectorAssembler if there are any columns to be vectorized
    if vector_columns:
        assembler = VectorAssembler(inputCols=vector_columns, outputCol="features")
        df = assembler.transform(df)

    # Now `df` will have a new column "features" that contains the vectorized form of your numerical columns.
    return df

# Example: Transforming the DataFrame with the col_map
col_map = {
    "accountAge": "standardize",
    "gender": "onehot_encode",
    "name": "no_transform"
}

df_transformed = transform_with_vector(df, col_map)

# Check the schema to ensure the vectorized column is created
df_transformed.printSchema()
