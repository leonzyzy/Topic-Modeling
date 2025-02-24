from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

# Initialize Spark
spark = SparkSession.builder.getOrCreate()

# Sample Data
data = [
    (101, 10.0, 100.0),
    (101, 20.0, 200.0),
    (102, 5.0, 50.0),
    (102, 8.0, 80.0),
    (102, 12.0, 120.0),
    (103, 15.0, 150.0)
]
df = spark.createDataFrame(data, ["account_id", "feature1", "feature2"])

# Step 1: Count rows per account_id
count_df = df.groupBy("account_id").count()

# Step 2: Get max count across all accounts
max_count = count_df.agg(F.max("count")).collect()[0][0]

# Step 3: Generate row indices (1 to max_count) for each account_id
expanded_df = count_df.withColumn("index", F.expr(f"sequence(1, {max_count})")) \
                      .select("account_id", F.explode("index").alias("row_num"))

# Step 4: Add row number to original DataFrame for join
window_spec = Window.partitionBy("account_id").orderBy(F.lit(0))  # Keep original order
df = df.withColumn("row_num", F.row_number().over(window_spec))

# Step 5: Left join to fill missing rows
padded_df = expanded_df.join(df, ["account_id", "row_num"], "left").drop("row_num")

# Step 6: Show results
padded_df.show()
