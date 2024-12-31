import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from sklearn.metrics import precision_score, recall_score, matthews_corrcoef
import os

# Dummy ToyModel for demonstration
class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.fc = nn.Linear(10, 5)

    def forward(self, x):
        return self.fc(x)

# Compute metrics for accuracy, precision, recall, specificity, and MCC
def compute_metrics(pred, true, rank, world_size):
    """Compute precision, recall, specificity, and MCC in a distributed manner."""
    # Convert the lists to tensors
    pred_tensor = torch.tensor(pred).to(rank)
    true_tensor = torch.tensor(true).to(rank)

    # Compute local TP, FP, TN, FN
    tp = torch.sum((pred_tensor == 1) & (true_tensor == 1)).float().to(rank)
    fp = torch.sum((pred_tensor == 1) & (true_tensor == 0)).float().to(rank)
    tn = torch.sum((pred_tensor == 0) & (true_tensor == 0)).float().to(rank)
    fn = torch.sum((pred_tensor == 0) & (true_tensor == 1)).float().to(rank)

    # Aggregate across all processes
    for tensor in [tp, fp, tn, fn]:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

    # Calculate metrics
    epsilon = 1e-8
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    specificity = tn / (tn + fp + epsilon)
    mcc = (tp * tn - fp * fn) / (
        torch.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) + epsilon
    )

    # Calculate accuracy
    accuracy = torch.sum(pred_tensor == true_tensor).float() / len(pred_tensor)

    # Convert to numpy for use in scikit-learn metrics
    pred_np = pred_tensor.cpu().numpy()
    true_np = true_tensor.cpu().numpy()

    # Using sklearn for additional metrics (precision, recall, MCC)
    precision_sklearn = precision_score(true_np, pred_np)
    recall_sklearn = recall_score(true_np, pred_np)
    mcc_sklearn = matthews_corrcoef(true_np, pred_np)

    return {
        "accuracy": accuracy.item(),
        "precision": precision_sklearn,
        "recall": recall_sklearn,
        "specificity": specificity.item(),
        "mcc": mcc_sklearn
    }

# Create DataLoader function
def create_dataloader(batch_size=32, num_workers=4):
    # Example function to create dataloaders (this can be modified for your dataset)
    from torch.utils.data import DataLoader, TensorDataset
    data = torch.randn(1000, 10)
    labels = torch.randint(0, 2, (1000, 5))
    dataset = TensorDataset(data, labels)
    return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)

# Training loop with evaluation
def train(rank, world_size, epochs=10):
    # Setup for DDP
    torch.cuda.set_device(rank)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    
    # Model, optimizer, and loss function
    model = ToyModel().to(rank)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[rank])
    optimizer = optim.SGD(model.parameters(), lr=0.001)
    loss_fn = nn.BCEWithLogitsLoss()

    # DataLoader
    train_loader = create_dataloader(batch_size=32, num_workers=4)
    
    # Training Loop
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        all_preds = []
        all_labels = []
        
        for data, labels in train_loader:
            data, labels = data.to(rank), labels.to(rank)
            optimizer.zero_grad()
            
            outputs = model(data)
            loss = loss_fn(outputs, labels.float())
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            # Collect predictions and true labels
            preds = torch.round(torch.sigmoid(outputs)).cpu().numpy()
            true = labels.cpu().numpy()
            
            all_preds.extend(preds)
            all_labels.extend(true)

        # Compute metrics after the epoch
        metrics = compute_metrics(all_preds, all_labels, rank, world_size)

        # Print results from rank 0
        if rank == 0:
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss / len(train_loader):.4f}")
            print(f"Accuracy: {metrics['accuracy']:.4f}, Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}")
            print(f"Specificity: {metrics['specificity']:.4f}, MCC: {metrics['mcc']:.4f}")

        # Save checkpoint after each epoch
        if rank == 0:
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss / len(train_loader),
            }
            torch.save(checkpoint, f"checkpoint_epoch_{epoch + 1}.pth")

    # Cleanup
    dist.destroy_process_group()

# Main function for running training with torchrun (distributed)
def main():
    world_size = 2  # Number of nodes (GPU devices)
    torchrun_args = {
        "backend": "nccl",
        "rank": 0,
        "world_size": world_size,
    }
    # This function should be run using `torchrun`
    train(**torchrun_args)

if __name__ == "__main__":
    main()
