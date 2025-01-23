import numpy as np
import torch

# Example: Convert DataLoader to NumPy array
def dataloader_to_numpy(dataloader):
    data_list = []
    
    for data, targets in dataloader:  # assuming the data is in the form (inputs, labels)
        data_list.append(data.numpy())  # Convert tensor to NumPy array
        # Optionally: you can also convert targets if needed
        # targets_list.append(targets.numpy())
    
    # Concatenate all batches into one NumPy array
    data_array = np.concatenate(data_list, axis=0)  # Adjust axis based on data shape
    
    return data_array

# Example usage
train_numpy = dataloader_to_numpy(train_loader)
