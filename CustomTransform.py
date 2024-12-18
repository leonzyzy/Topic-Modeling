import pandas as pd
import numpy as np

# Generate random numerical DataFrame
np.random.seed(42)  # For reproducibility
data = {
    "Column_A": np.random.randint(0, 100, size=10),                # Random integers
    "Column_B": np.random.uniform(0, 1, size=10),                 # Random floats
    "Column_C": np.random.normal(50, 10, size=10),                # Normally distributed data
    "Column_D": np.random.randint(1, 4, size=10),                 # Random integers (1 to 3)
    "Column_E": np.random.randint(1000, 2000, size=10)            # Random integers (1000 to 2000)
}

df = pd.DataFrame(data)
print(df)
