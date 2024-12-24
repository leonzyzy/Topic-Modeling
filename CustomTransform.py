import torch
from litdata import StreamingDataset, StreamingDataLoader

class CustomTabularDataset(StreamingDataset):
    def __init__(self, data_dict):
        # Assuming data_dict contains 'features' and 'target'
        self.data_dict = data_dict

    def __getitem__(self, index):
        # Get the features and target at the given index
        X = self.data_dict['features'][index]  # Features: a numpy array of shape (num_features,)
        y = self.data_dict['target'][index]    # Target: a numpy array (e.g., scalar or one-hot)

        # Convert numpy arrays to torch tensors
        X = torch.from_numpy(X).float()  # Convert features to tensor, ensure float type
        y = torch.from_numpy(y).long()   # Convert target to tensor, ensure long type (for classification)

        return X, y


# Example data_dict with numpy arrays (replace with your actual data)
data_dict = {
    'features': np.random.randn(100, 10),  # Example feature data (100 samples, 10 features each)
    'target': np.random.randint(0, 2, 100)  # Example target data (binary classification, 100 samples)
}

# Instantiate the dataset and dataloader
dataset = CustomTabularDataset(data_dict)
dataloader = StreamingDataLoader(dataset, batch_size=4)

# Iterate through the dataloader
for X_batch, y_batch in dataloader:
    print(X_batch.shape)  # Should output (4, 10) for tabular data with 10 features
    print(y_batch.shape)  # Should output (4,) for binary targets (or (4, num_classes) for multi-class)
