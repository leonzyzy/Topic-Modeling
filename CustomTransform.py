import pandas as pd
import numpy as np

# Create a 1000x5 DataFrame with random numerical values
df = pd.DataFrame(np.random.rand(1000, 5), columns=['Col1', 'Col2', 'Col3', 'Col4', 'Col5'])

# Save the DataFrame to a CSV file
file_path = '/mnt/data/numerical_df.csv'
df.to_csv(file_path, index=False)

file_path
