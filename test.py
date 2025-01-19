def sparse_to_array(sparse_vector):
    return sparse_vector.toArray().tolist() if sparse_vector is not None else []

sparse_to_array_udf = F.udf(sparse_to_array, ArrayType(FloatType()))

# Convert sparse vector to an array
encoded_df = encoded_df.withColumn("animal_onehot_array", sparse_to_array_udf(F.col("animal_onehot")))

# Get the number of categories (the length of the one-hot encoded array)
num_categories = len(encoded_df.select("animal_onehot_array").first()[0])

# Expand the array into separate columns for each category
for i in range(num_categories):
    encoded_df = encoded_df.withColumn(f"animal_category_{i}", F.col("animal_onehot_array")[i])

# Drop intermediate columns (e.g., onehot array column and index column)
encoded_df = encoded_df.drop("animal_index", "animal_onehot", "animal_onehot_array")

# Show the resulting DataFrame
encoded_df.show(truncate=False)
