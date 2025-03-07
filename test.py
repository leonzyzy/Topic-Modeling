from torch.nn.utils.rnn import pad_sequence

def collate_fn(batch):
    features, targets = zip(*batch)  # Separate features and targets

    # Pad features
    padded_features = pad_sequence(features, batch_first=True, padding_value=0)

    # Pad targets (if necessary, assuming they are 1D lists or tensors)
    padded_targets = pad_sequence([torch.tensor(tgt, dtype=torch.float32) for tgt in targets], 
                                  batch_first=True, padding_value=0)
    
    return padded_features, padded_targets
