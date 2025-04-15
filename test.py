import torch
from torch.utils.data import DataLoader

MAX_LEN = 222
PAD_VALUE = 0  # for inputs
PAD_LABEL = -1  # for labels

def collate_fn(batch):
    # Find feature dimension
    feat_dim = batch[0]['input'].size(-1)
    
    batch_inputs = []
    batch_labels = []
    attention_masks = []
    label_masks = []

    for sample in batch:
        input_seq = sample['input']
        label_seq = sample['label']
        
        seq_len = input_seq.size(0)

        # Pad input
        if seq_len < MAX_LEN:
            pad_len = MAX_LEN - seq_len
            padded_input = torch.cat([
                input_seq,
                torch.full((pad_len, feat_dim), PAD_VALUE)
            ], dim=0)
        else:
            padded_input = input_seq[:MAX_LEN]

        # Pad label
        if seq_len < MAX_LEN:
            padded_label = torch.cat([
                label_seq,
                torch.full((MAX_LEN - seq_len,), PAD_LABEL)
            ])
        else:
            padded_label = label_seq[:MAX_LEN]

        # Create attention and label masks
        attention_mask = torch.zeros(MAX_LEN)
        attention_mask[:min(seq_len, MAX_LEN)] = 1

        label_mask = (padded_label != PAD_LABEL).float()

        # Append to batch lists
        batch_inputs.append(padded_input)
        batch_labels.append(padded_label)
        attention_masks.append(attention_mask)
        label_masks.append(label_mask)

    return {
        'input': torch.stack(batch_inputs),         # [batch, MAX_LEN, feat_dim]
        'label': torch.stack(batch_labels),         # [batch, MAX_LEN]
        'attention_mask': torch.stack(attention_masks),  # [batch, MAX_LEN]
        'label_mask': torch.stack(label_masks),     # [batch, MAX_LEN]
    }
