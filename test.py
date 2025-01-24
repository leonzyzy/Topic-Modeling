import numpy as np
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __len__(self):
        # Return the number of rows (samples) in the original dataset
        data = super().__getitem__(0)  # Get the first sample
        X, y = data  # Separate features and target
        return len(X)  # The length should be the number of rows in the dataset

    def __getitem__(self, idx):
        # Get the idx-th row of data (using super().__getitem__(idx))
        data = super().__getitem__(idx)  # Retrieves a row, assumed to be a pandas DataFrame
        
        # Extract the features (X) and target (y)
        X = data.drop('target').values  # All columns except 'target' for features
        y = data['target']  # The 'target' column as the target
        
        # Concatenate features (X) and target (y) into a single row (tabular)
        concatenated = np.concatenate([X, np.array([y])], axis=0)  # Concatenate X and y

        # Return the concatenated row (tabular data) and the label (y) separately
        return concatenated, y
