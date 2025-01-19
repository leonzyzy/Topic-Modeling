from pyspark.ml import Pipeline
from pyspark.ml.feature import StandardScaler, StringIndexer, OneHotEncoder
from pyspark.sql import functions as F
from pyspark.ml.feature import VectorAssembler

def fit_transform(df, col_map):
    stages = []
    
    # Iterate over each column in the col_map to apply the respective transformation
    for col_name, transform_type in col_map.items():
        
        if transform_type == 'standardize':
            # Standardize: Use StandardScaler to standardize the column
            assembler = VectorAssembler(inputCols=[col_name], outputCol=col_name + "_vec")
            scaler = StandardScaler(inputCol=col_name + "_vec", outputCol=col_name + "_scaled")
            
            stages += [assembler, scaler]
        
        elif transform_type == 'onehot_encode':
            # OneHot Encode: Use StringIndexer and OneHotEncoder
            indexer = StringIndexer(inputCol=col_name, outputCol=col_name + "_index")
            encoder = OneHotEncoder(inputCol=col_name + "_index", outputCol=col_name + "_onehot")
            
            stages += [indexer, encoder]
        
        elif transform_type == 'no_transform':
            # No transformation: Just keep the column as is (no-op)
            pass
    
    # Create a pipeline with the stages
    pipeline = Pipeline(stages=stages)
    
    # Fit the pipeline and transform the data
    model = pipeline.fit(df)
    transformed_df = model.transform(df)
    
    # Handle one-hot encoding conversion
    for col_name, transform_type in col_map.items():
        if transform_type == 'onehot_encode':
            # Convert the one-hot sparse vector to an array
            transformed_df = transformed_df.withColumn(col_name + "_onehot_array", F.col(col_name + "_onehot").toArray())
            
            # If you want to break the array into individual columns:
            num_categories = len(transformed_df.select(col_name + "_onehot_array").first()[0])  # Get number of categories
            for i in range(num_categories):
                transformed_df = transformed_df.withColumn(f"{col_name}_category_{i}", F.col(col_name + "_onehot_array")[i])
            
            # Drop intermediate columns like index and one-hot vector columns
            transformed_df = transformed_df.drop(col_name + "_index", col_name + "_onehot", col_name + "_onehot_array")
    
    # Drop any extra temporary columns like vector columns used for scaling
    drop_cols = []
    for col_name, transform_type in col_map.items():
        if transform_type == 'standardize':
            drop_cols += [col_name + "_vec"]  # Drop vector columns used for scaling

    # Drop unnecessary columns and return the final DataFrame
    transformed_df = transformed_df.drop(*drop_cols)
    
    return transformed_df
