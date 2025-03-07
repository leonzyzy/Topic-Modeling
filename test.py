class AccountDataset(Dataset):
    def __init__(self, data_dict, targets):
        self.account_ids = list(data_dict.keys())
        self.features = [torch.tensor(data_dict[acc], dtype=torch.float32) for acc in self.account_ids]
        self.targets = [torch.tensor(targets[acc], dtype=torch.float32) for acc in self.account_ids]

        # Find the max length of the feature matrices and target sequences
        self.max_feature_length = max([feature.shape[0] for feature in self.features])
        self.max_target_length = 1  # Assuming scalar targets, no need to pad

    def __len__(self):
        return len(self.account_ids)

    def __getitem__(self, idx):
        feature = self.features[idx]
        target = self.targets[idx]
        
        # Pad feature matrix to global maximum length
        padded_feature = torch.cat([feature, torch.zeros(self.max_feature_length - feature.shape[0], feature.shape[1])], dim=0)
        
        # Return padded feature and target
        return padded_feature, target
