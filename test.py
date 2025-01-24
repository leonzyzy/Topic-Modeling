import torch
from torch.utils.data import Dataset

class TabularDataset(Dataset):
    def __init__(self, data):
        """
        Initialize the dataset.

        Args:
            data (list of Tensors): A list of 2D tensors, 
                                    each with shape (chunk_size, num_columns).
        """
        self.data = data  # List of 2D tensors
        self.chunk_sizes = [chunk.shape[0] for chunk in data]  # Number of rows in each chunk
        self.cumulative_sizes = torch.cumsum(torch.tensor(self.chunk_sizes), dim=0)  # Cumulative row indices

    def __len__(self):
        """
        Return the total number of rows across all chunks.
        """
        return self.cumulative_sizes[-1].item()

    def __getitem__(self, idx):
        """
        Get the row corresponding to the given index across all chunks.

        Args:
            idx (int): The global row index.

        Returns:
            Tensor: A single row (1D tensor) corresponding to the index.
        """
        # Find which chunk the row belongs to
        chunk_idx = torch.searchsorted(self.cumulative_sizes, idx, right=True)
        if chunk_idx > 0:
            row_idx = idx - self.cumulative_sizes[chunk_idx - 1]
        else:
            row_idx = idx
        
        # Extract the row from the appropriate chunk
        return self.data[chunk_idx][row_idx]

# Example Usage
if __name__ == "__main__":
    # Create dummy data (5 tensors, each with shape (70,000, 47))
    dummy_data = [torch.randn(70000, 47) for _ in range(5)]
    
    # Initialize the dataset
    dataset = TabularDataset(dummy_data)
    
    # Total length of the dataset (350,000 rows)
    print(f"Total number of rows: {len(dataset)}")
    
    # Access a specific row by its global index
    print(f"Shape of dataset[0]: {dataset[0].shape}")  # A single row: (47,)
    print(f"Shape of dataset[349999]: {dataset[349999].shape}")  # The last row
