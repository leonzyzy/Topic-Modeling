import torch

MAX_LEN = 222
PAD_VALUE = 0.0   # input padding
PAD_LABEL = -1    # label padding

def collate_fn(batch):
    batch_inputs = []
    batch_labels = []
    attention_masks = []
    label_masks = []

    feat_dim = batch[0][0].size(-1)

    for X, y in batch:
        seq_len = X.size(0)

        # Pad input
        if seq_len < MAX_LEN:
            pad_len = MAX_LEN - seq_len
            padded_X = torch.cat([
                X,
                torch.full((pad_len, feat_dim), PAD_VALUE)
            ], dim=0)
        else:
            padded_X = X[:MAX_LEN]

        # Pad label
        if seq_len < MAX_LEN:
            padded_y = torch.cat([
                y,
                torch.full((MAX_LEN - seq_len,), PAD_LABEL, dtype=y.dtype)
            ])
        else:
            padded_y = y[:MAX_LEN]

        # Attention mask (1 where input is real)
        attn_mask = torch.zeros(MAX_LEN)
        attn_mask[:min(seq_len, MAX_LEN)] = 1

        # Label mask (1 where label is valid)
        label_mask = (padded_y != PAD_LABEL).float()

        batch_inputs.append(padded_X)
        batch_labels.append(padded_y)
        attention_masks.append(attn_mask)
        label_masks.append(label_mask)

    return {
        'input': torch.stack(batch_inputs),         # [B, 222, feat_dim]
        'label': torch.stack(batch_labels),         # [B, 222]
        'attention_mask': torch.stack(attention_masks),  # [B, 222]
        'label_mask': torch.stack(label_masks),     # [B, 222]
    }
