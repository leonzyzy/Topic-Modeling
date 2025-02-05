import torch
import torch.nn as nn

class FTTransformer(nn.Module):
    def __init__(self, num_features, cat_cardinalities, embed_dim=32, num_heads=4, num_layers=2, num_classes=2):
        super().__init__()
        
        # Feature tokenization
        self.cat_emb = nn.ModuleList([nn.Embedding(cat, embed_dim) for cat in cat_cardinalities])
        self.num_emb = nn.Linear(1, embed_dim)  # Learnable embedding for numerical features

        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))  # CLS token

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads)
        self.transformer = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)

        self.fc = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x_cat, x_num):
        # Encode categorical features
        x_cat = [emb(x_cat[:, i]) for i, emb in enumerate(self.cat_emb)]
        x_cat = torch.stack(x_cat, dim=1)  # Shape: (batch, num_cat, embed_dim)

        # Encode numerical features
        x_num = self.num_emb(x_num.unsqueeze(-1))  # Shape: (batch, num_num, embed_dim)

        # Combine categorical & numerical tokens
        x = torch.cat([x_cat, x_num], dim=1)  # Shape: (batch, num_features, embed_dim)

        # Add CLS token
        cls_token = self.cls_token.expand(x.size(0), -1, -1)  # Shape: (batch, 1, embed_dim)
        x = torch.cat([cls_token, x], dim=1)  # Shape: (batch, num_features + 1, embed_dim)

        # Pass through Transformer
        x = self.transformer(x)

        # Use CLS token for classification
        x = x[:, 0, :]  # Extract CLS token output
        return self.fc(x)

# Example Usage:
num_categorical = [10, 20, 15]  # Assume 3 categorical features with these cardinalities
num_numerical = 5  # Assume 5 numerical features
num_classes = 2
model = FTTransformer(num_features=num_numerical, cat_cardinalities=num_categorical, num_classes=num_classes)
