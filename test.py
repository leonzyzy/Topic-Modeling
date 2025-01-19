def fit_transform(df, col_map):
    stages = []  # Store the transformations
    
    for col_name, transform_type in col_map.items():
        if transform_type == "standardize":
            # Standardize numerical columns
            scaler = StandardScaler(inputCol=col_name, outputCol=f"{col_name}_scaled")
            stages.append(scaler)
        
        elif transform_type == "onehot_encode":
            # OneHotEncoding categorical columns
            indexer = StringIndexer(inputCol=col_name, outputCol=f"{col_name}_indexed")
            encoder = OneHotEncoder(inputCol=f"{col_name}_indexed", outputCol=f"{col_name}_onehot")
            stages.append(indexer)
            stages.append(encoder)
        
        elif transform_type == "no_transform":
            # No transformation, just keep the column as is
            pass

    # Create a pipeline with the transformations
    pipeline = Pipeline(stages=stages)
    
    # Fit and transform the data
    model = pipeline.fit(df)
    transformed_df = model.transform(df)

    return transformed_df, model
