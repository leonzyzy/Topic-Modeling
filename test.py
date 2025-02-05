# Example dataset
data = pd.DataFrame({
    'Loan_Type': ['Car Loan', 'Home Loan', 'Personal Loan', 'Car Loan', 'Home Loan', 'Personal Loan'],
    'Employment_Status': ['Employed', 'Unemployed', 'Self-Employed', 'Employed', 'Unemployed', 'Employed'],
    'State': ['CA', 'TX', 'NY', 'CA', 'TX', 'NY'],
    'Income': [50000, 60000, 45000, 52000, 70000, 48000],  # Numerical feature
    'Default': [0, 1, 0, 0, 1, 0]  # Binary target
})

# Separate features and target
X = data.drop(columns=['Default'])
y = data['Default']

# Identify categorical and numerical columns
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Encode categorical variables
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le

# Standardize numerical columns
scaler = StandardScaler()
X[numerical_cols] = scaler.fit_transform(X[numerical_cols])

# Convert to numpy
X = X.values
y = y.values


# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)  # LongTensor for classification
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# Create DataLoader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)



# Define input parameters
num_numerical_features = len(numerical_cols)  # Number of numerical features
cat_cardinalities = [len(label_encoders[col].classes_) for col in categorical_cols]  # Unique categories per categorical column

# Create FT-Transformer model
model = rtdl.FTTransformer.make_default(
    n_num_features=num_numerical_features,
    cat_cardinalities=cat_cardinalities,
    d_out=2  # Binary classification (2 classes: 0 or 1)
)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-3)


num_epochs = 20

for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        output = model(X_batch)
        loss = criterion(output, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")



model.eval()
correct = 0
total = 0

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        outputs = model(X_batch)
        _, predicted = torch.max(outputs, 1)
        total += y_batch.size(0)
        correct += (predicted == y_batch).sum().item()

accuracy = 100 * correct / total
print(f"Test Accuracy: {accuracy:.2f}%")



from torch.utils.data import Dataset, DataLoader
import torch

class TabularDataset(Dataset):
    def __init__(self, X_num, X_cat, y):
        """
        X_num: Tensor of numerical features (shape: [num_samples, num_numerical_features])
        X_cat: Tensor of categorical features (shape: [num_samples, num_categorical_features])
        y: Target labels (shape: [num_samples])
        """
        self.X_num = torch.tensor(X_num, dtype=torch.float32)  # Numerical features
        self.X_cat = torch.tensor(X_cat, dtype=torch.long)      # Categorical features (for embeddings)
        self.y = torch.tensor(y, dtype=torch.long)              # Target labels

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return (self.X_num[idx], self.X_cat[idx]), self.y[idx]  # Return as tuple



import torch
import torch.nn as nn
import torch.optim as optim
import rtdl

class FTTransformerWithAttention(rtdl.FTTransformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.attention_weights = None  # Store attention matrix

    def forward(self, x_num, x_cat=None):
        """Forward pass with attention storage."""
        x = self.embed(x_num, x_cat)  # Embed numerical & categorical features

        # Hook function to capture attention matrix
        def hook(module, input, output):
            self.attention_weights = module.attn.attn_weights  # Store attention

        # Register hook on the first transformer block
        self.blocks[0].attn.register_forward_hook(hook)

        return super().forward(x_num, x_cat)

# Example: Define model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FTTransformerWithAttention.make_default(
    n_num_features=len(numerical_cols),
    cat_cardinalities=cat_cardinalities,
    d_out=2
).to(device)

# Define loss & optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-3)

# Training Loop (Updated)
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for (X_batch_num, X_batch_cat), y_batch in train_loader:
        X_batch_num, X_batch_cat, y_batch = (
            X_batch_num.to(device),
            X_batch_cat.to(device),
            y_batch.to(device),
        )

        optimizer.zero_grad()
        output = model(X_batch_num, X_batch_cat)  # ✅ Pass both numerical & categorical features
        loss = criterion(output, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

# Extract Attention Matrix After Training
with torch.no_grad():
    sample_num, sample_cat = next(iter(test_loader))  # Get a batch
    sample_num, sample_cat = sample_num.to(device), sample_cat.to(device)

    _ = model(sample_num, sample_cat)  # Run model once to get attention

attention_matrix = model.attention_weights  # Shape: (batch_size, num

