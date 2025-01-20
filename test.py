from pyspark.sql.functions import col, when, count

# Count the number of NULLs in each column
null_counts = df.select([count(when(col(c).isNull(), c)).alias(c) for c in df.columns])

# Show the result
null_counts.show()
