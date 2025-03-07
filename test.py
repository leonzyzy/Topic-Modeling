class AccountDataset(Dataset):
    def __init__(self, data_dict):
        self.account_ids = list(data_dict.keys())  # List of account IDs
        self.features = [data_dict[acc]['features'] for acc in self.account_ids]  # Extract features
        self.targets = [data_dict[acc]['target'] for acc in self.account_ids]  # Extract targets

        # Convert to tensors if necessary
        self.features = torch.tensor(self.features, dtype=torch.float32)
        self.targets = torch.tensor(self.targets, dtype=torch.float32)

    def __len__(self):
        return len(self.account_ids)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]
