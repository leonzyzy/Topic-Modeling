import torch
from torch.utils.data import Dataset, DataLoader

class TransactionDataset(Dataset):
    def __init__(self, accounts):
        """
        Dataset containing transaction histories per account.

        Args:
            accounts (list of lists): Each account contains a variable-length sequence of transactions.
        """
        self.accounts = accounts

    def __len__(self):
        return len(self.accounts)

    def __getitem__(self, idx):
        return torch.tensor(self.accounts[idx], dtype=torch.float32)  # Convert to tensor

def collate_fn(batch, max_len=50):
    """
    Custom collate function for DataLoader to handle padding and truncation.

    Args:
        batch (list of Tensors): Each tensor represents an account's transactions.
        max_len (int): Maximum sequence length after padding/truncation.

    Returns:
        torch.Tensor: Padded tensor of shape (batch_size, max_len, feature_dim).
    """
    feature_dim = batch[0].shape[1]  # Get feature dimension
    padded_batch = []

    for account in batch:
        seq_len = account.shape[0]

        if seq_len > max_len:
            truncated = account[-max_len:]  # Keep the last 50 transactions
        else:
            pad_size = (max_len - seq_len, feature_dim)
            padding = torch.zeros(pad_size, dtype=torch.float32)  # Pad with zeros
            truncated = torch.cat([padding, account], dim=0)  # Pad in the front to keep alignment

        padded_batch.append(truncated)

    return torch.stack(padded_batch)  # Shape: (batch_size, max_len, feature_dim)

# Example Transaction Data
accounts = [
    [[1, 2, 3]],  # 1 transaction
    [[1, 2, 3], [4, 5, 6]],  # 2 transactions
    [[1, 2, 3], [4, 5, 6], [7, 8, 9]],  # 3 transactions
    [[i, i+1, i+2] for i in range(0, 300, 3)],  # 100 transactions (will keep last 50)
]

# Create Dataset
dataset = TransactionDataset(accounts)

# Create DataLoader with Custom Collate Function
dataloader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)

# Fetch a batch
for batch in dataloader:
    print(batch.shape)  # Expected: (batch_size, 50, feature_dim)
    print(batch)  # Check padding & truncation
    break  # Just print the first batch
