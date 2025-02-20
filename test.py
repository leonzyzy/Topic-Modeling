# Generate a sequence of numbers up to max_transactions per account
seq_df = (transaction_counts
    .select("account_id", F.explode(F.array([F.lit(i+1) for i in range(max_transactions)])).alias("row_number")))

# Perform an outer join to ensure each account has max_transactions rows
padded_df = (seq_df
    .join(df, on=["account_id", "row_number"], how="left")
    .drop("transaction_count"))
