import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from tqdm import tqdm

# Define your model architecture (update based on your actual model)
class MyModel(torch.nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.fc = torch.nn.Linear(128, 1)  # Example model

    def forward(self, x):
        return self.fc(x)  # Output is logits

# Initialize model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MyModel().to(device)

# Load state_dict (OrderedDict)
state_dict = torch.load("model.pt", map_location=device)
model.load_state_dict(state_dict)
model.eval()

# Function to compute metrics
def evaluate_model(model, dataloader):
    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Evaluating"):
            inputs, labels = inputs.to(device), labels.to(device)
            logits = model(inputs)  # Model outputs logits
            probs = torch.sigmoid(logits).squeeze()  # Convert logits to probabilities
            preds = (probs > 0.5).long()  # Apply threshold at 0.5

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    # Compute metrics
    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average="binary", zero_division=0)
    recall = recall_score(all_labels, all_preds, average="binary", zero_division=0)
    f1 = f1_score(all_labels, all_preds, average="binary", zero_division=0)
    auc = roc_auc_score(all_labels, all_probs)

    return {"accuracy": acc, "precision": precision, "recall": recall, "f1-score": f1, "auc": auc}

# Evaluate
metrics = evaluate_model(model, test_dataloader)
print(metrics)
