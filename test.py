import torch
import torch.nn as nn
import torchmetrics
from torch.utils.data import DataLoader, Dataset
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from sklearn.metrics import roc_auc_score, average_precision_score
import argparse

# Sample dataset class for testing purposes
class SampleDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# Dummy model (replace with your actual model)
class DummyModel(nn.Module):
    def __init__(self):
        super(DummyModel, self).__init__()
        self.fc = nn.Linear(10, 2)  # Change as per your model's structure

    def forward(self, x):
        return self.fc(x)

# Function to initialize the distributed environment
def init_distributed_mode(args):
    dist.init_process_group(backend="nccl", init_method="env://")
    torch.cuda.set_device(args.local_rank)
    print(f"Running on rank {args.rank} in {args.world_size} processes")

# Evaluation function to compute the metrics
def evaluate(model, dataloader, device):
    model.eval()
    accuracy_metric = torchmetrics.classification.Accuracy().to(device)
    precision_metric = torchmetrics.classification.Precision(num_classes=2).to(device)
    recall_metric = torchmetrics.classification.Recall(num_classes=2).to(device)
    f1_metric = torchmetrics.classification.F1Score(num_classes=2).to(device)
    f05_metric = torchmetrics.classification.FBetaScore(beta=0.5, num_classes=2).to(device)
    roc_auc_metric = torchmetrics.classification.ROCAUC(num_classes=2).to(device)
    pr_auc_metric = torchmetrics.classification.AveragePrecision(num_classes=2).to(device)

    # Collect metrics
    with torch.no_grad():
        all_preds = []
        all_labels = []

        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)

            # Update metrics
            accuracy_metric.update(preds, labels)
            precision_metric.update(preds, labels)
            recall_metric.update(preds, labels)
            f1_metric.update(preds, labels)
            f05_metric.update(preds, labels)
            roc_auc_metric.update(preds, labels)
            pr_auc_metric.update(preds, labels)

            all_preds.append(preds)
            all_labels.append(labels)

    # Aggregate metrics
    accuracy = accuracy_metric.compute()
    precision = precision_metric.compute()
    recall = recall_metric.compute()
    f1 = f1_metric.compute()
    f05 = f05_metric.compute()
    roc_auc = roc_auc_metric.compute()
    pr_auc = pr_auc_metric.compute()

    return accuracy, precision, recall, f1, f05, roc_auc, pr_auc

def main(args):
    # Initialize the distributed environment
    init_distributed_mode(args)

    # Prepare dataset and dataloaders
    data = torch.randn(1000, 10)  # Dummy data (change to your dataset)
    labels = torch.randint(0, 2, (1000,))  # Dummy labels (binary classification)

    dataset = SampleDataset(data, labels)
    train_sampler = torch.utils.data.DistributedSampler(dataset)
    test_loader = DataLoader(dataset, batch_size=32, sampler=train_sampler)

    # Initialize the model
    model = DummyModel().to(args.device)
    model = DDP(model, device_ids=[args.local_rank])

    # Evaluate the model
    accuracy, precision, recall, f1, f05, roc_auc, pr_auc = evaluate(model, test_loader, args.device)

    if args.rank == 0:  # Only print results on the main process
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"F0.5 Score: {f05:.4f}")
        print(f"ROC AUC: {roc_auc:.4f}")
        print(f"PR AUC: {pr_auc:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', type=int, default=-1, help='Local rank of the process')
    parser.add_argument('--device', type=str, default='cuda', help='Device to run on')
    parser.add_argument('--rank', type=int, default=0, help='Rank of the process')
    parser.add_argument('--world_size', type=int, default=1, help='Total number of processes')
    
    args = parser.parse_args()
    main(args)
