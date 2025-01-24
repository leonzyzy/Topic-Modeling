import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

class YourDatasetClass(Dataset):
    def __init__(self, ...):
        # Initialize and load data
        # Concatenate all data into one large DataFrame
        self.data = self.concatenate_data()

    def concatenate_data(self):
        # Initialize an empty list to store the DataFrames
        df_list = []
        
        for idx in range(5):  # Assuming 5 chunks of data
            df = pd.DataFrame(...)  # Fetch DataFrame for each index
            df_list.append(df)
        
        # Concatenate all DataFrames into one large DataFrame
        concatenated_df = pd.concat(df_list, ignore_index=True)
        return concatenated_df
    
    def __len__(self):
        # Return the number of rows in the concatenated DataFrame
        return len(self.data)
    
    def __getitem__(self, idx):
        # Return a single row as a sample (either as tensor or DataFrame)
        row = self.data.iloc[idx].values  # Convert row to numpy array
        return torch.tensor(row, dtype=torch.float)  # Or any other appropriate type

# Create DataLoader
dataset = YourDatasetClass()
data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Example of iterating through the DataLoader
for batch in data_loader:
    print(batch.shape)  # Should print (batch_size, 47)
