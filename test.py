import torch
from torch.utils.data import Dataset, DataLoader

class AccountDataset(Dataset):
    def __init__(self, data_dict):
        self.account_ids = data_dict["account_id"]  # List of account IDs
        self.features = torch.tensor(data_dict["features"], dtype=torch.float32)  # Convert to tensor
        self.targets = torch.tensor(data_dict["target"], dtype=torch.float32)  # Convert to tensor

    def __len__(self):
        return len(self.account_ids)  # Total number of samples

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]  # Return feature-target pair

# Example Data Dictionary
data_dict = {
    "account_id": ["acc_1", "acc_2", "acc_3"],
    "features": [[1.2, 3.4, 5.6], [4.5, 6.7, 8.9], [7.8, 9.0, 10.1]],  # Each entry is a feature vector
    "target": [0, 1, 0]  # Binary classification targets
}

# Create Dataset
dataset = AccountDataset(data_dict)

print(len(dataset))  # Should print 3
print(dataset[0])  # Should print (features tensor, target tensor)
