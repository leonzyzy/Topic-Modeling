from pyspark.sql.functions import col, when, count

# Count the number of NULLs in each column
null_counts = df.select([count(when(col(c).isNull(), c)).alias(c) for c in df.columns])

# Show the result
import numpy as np

def dataloader_to_numpy(dataloader):
    """
    Convert all data in a PyTorch DataLoader to a single NumPy array.

    Args:
        dataloader (torch.utils.data.DataLoader): DataLoader object.

    Returns:
        numpy.ndarray: Concatenated data as a NumPy array.
    """
    all_data = []
    
    for batch in dataloader:
        # If the batch contains both inputs and targets (X, y), handle them separately
        if isinstance(batch, (list, tuple)) and len(batch) == 2:
            inputs, targets = batch
            all_data.append((inputs.numpy(), targets.numpy()))
        else:  # Handle single tensors
            all_data.append(batch.numpy())

    # Concatenate all batches
    if isinstance(all_data[0], tuple):
        inputs = np.concatenate([x[0] for x in all_data], axis=0)
        targets = np.concatenate([x[1] for x in all_data], axis=0)
        return inputs, targets
    else:
        return np.concatenate(all_data, axis=0)

