from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, FloatType
import random

# Initialize SparkSession
spark = SparkSession.builder.appName("CreateSparkDF").getOrCreate()

# Define a schema
schema = StructType([
    StructField("Col1", FloatType(), True),
    StructField("Col2", FloatType(), True),
    StructField("Col3", FloatType(), True),
    StructField("Col4", FloatType(), True),
    StructField("Col5", FloatType(), True)
])

# Create data for the DataFrame
data = [[random.random() for _ in range(5)] for _ in range(10)]

# Create a Spark DataFrame
spark_df = spark.createDataFrame(data, schema)

# Show the DataFrame
spark_df.show()
