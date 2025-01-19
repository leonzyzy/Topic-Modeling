from pyspark.sql.functions import col

# Get the column names and types from the DataFrame schema
double_columns = [col_name for col_name, dtype in df.dtypes if dtype == "double"]
string_columns = [col_name for col_name, dtype in df.dtypes if dtype == "string"]

# Fill missing values for DoubleType columns with 0.0
df_filled_double = df.fillna({col: 0.0 for col in double_columns})

# Fill missing values for StringType columns with "NA"
df_filled = df_filled_double.fillna({col: "NA" for col in string_columns})

# Show the result
df_filled.show()
