# Padding function
def collate_fn(batch):
    features, targets = zip(*batch)  # Separate features and targets

    # Pad matrices to the same size (batch_first=True makes shape [batch, max_rows, num_features])
    padded_features = pad_sequence(features, batch_first=True, padding_value=0)
    
    # Stack targets as a tensor
    targets = torch.stack(targets)

    return padded_features, targets
