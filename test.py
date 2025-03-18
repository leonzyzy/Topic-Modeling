import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from tqdm import tqdm

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load("model.pt", map_location=device)
model.eval()

# Define evaluation function
def evaluate_model(model, dataloader):
    all_labels = []
    all_preds = []
    all_probs = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Evaluating"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            
            if outputs.shape[-1] > 1:  # Multi-class classification
                probs = F.softmax(outputs, dim=1)
                preds = torch.argmax(probs, dim=1)
            else:  # Binary classification
                probs = torch.sigmoid(outputs).squeeze()
                preds = (probs > 0.5).long()
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    # Compute metrics
    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average="binary")
    recall = recall_score(all_labels, all_preds, average="binary")
    f1 = f1_score(all_labels, all_preds, average="binary")
    auc = roc_auc_score(all_labels, all_probs)

    return {"accuracy": acc, "precision": precision, "recall": recall, "f1-score": f1, "auc": auc}

# Evaluate the model
metrics = evaluate_model(model, test_dataloader)
print(metrics)
