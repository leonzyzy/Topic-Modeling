import torch
import torch.optim as optim
import torch.nn as nn
import torch.distributed as dist
import argparse
import os
from sklearn.metrics import precision_score, recall_score, matthews_corrcoef

# Configuration for the model and optimizer
def parse_args():
    parser = argparse.ArgumentParser(description="DDP Training with Hyperparameters")

    # Model and optimizer hyperparameters
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate for the optimizer")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs to train the model")
    parser.add_argument("--optimizer", type=str, default="SGD", choices=["SGD", "Adam"], help="Optimizer type")
    parser.add_argument("--momentum", type=float, default=0.9, help="Momentum for SGD optimizer")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay for the optimizer")
    parser.add_argument("--device", type=str, default="cuda", help="Device type (cpu or cuda)")
    parser.add_argument("--loss_fn", type=str, default="MSELoss", choices=["MSELoss", "CrossEntropyLoss"], help="Loss function")
    parser.add_argument("--logging_freq", type=int, default=10, help="Frequency of logging metrics")

    # Distributed training options
    parser.add_argument("--world_size", type=int, default=1, help="Total number of processes in distributed training")
    parser.add_argument("--rank", type=int, default=0, help="Rank of the current process")

    return parser.parse_args()

# DDP Setup function
def setup_ddp(rank, world_size, config):
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    model = ToyModel().to(rank)
    ddp_model = DDP(model, device_ids=[rank])
    return ddp_model

# Create DataLoader function for distributed training
def create_dataloader(batch_size, rank, world_size):
    train_dataset = CustomDataset()  # Replace with your dataset
    sampler = torch.utils.data.DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
    return train_loader

# Metrics computation function
def compute_metrics(pred, true, rank, world_size):
    pred_tensor = torch.tensor(pred).to(rank)
    true_tensor = torch.tensor(true).to(rank)
    
    tp = torch.sum((pred_tensor == 1) & (true_tensor == 1)).float().to(rank)
    fp = torch.sum((pred_tensor == 1) & (true_tensor == 0)).float().to(rank)
    tn = torch.sum((pred_tensor == 0) & (true_tensor == 0)).float().to(rank)
    fn = torch.sum((pred_tensor == 0) & (true_tensor == 1)).float().to(rank)

    for tensor in [tp, fp, tn, fn]:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

    epsilon = 1e-8
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    specificity = tn / (tn + fp + epsilon)
    mcc = (tp * tn - fp * fn) / (
        torch.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) + epsilon
    )

    accuracy = torch.sum(pred_tensor == true_tensor).float() / len(pred_tensor)

    return {
        "accuracy": accuracy.item(),
        "precision": precision.item(),
        "recall": recall.item(),
        "specificity": specificity.item(),
        "mcc": mcc.item()
    }

# Training function
def train(ddp_model, train_loader, config, rank, world_size):
    optimizer = optim.SGD(ddp_model.parameters(), lr=config.learning_rate, momentum=config.momentum, weight_decay=config.weight_decay)
    loss_fn = nn.MSELoss() if config.loss_fn == "MSELoss" else nn.CrossEntropyLoss()
    epoch_losses = []

    for epoch in range(config.epochs):
        ddp_model.train()
        total_loss = 0
        total_metrics = {"accuracy": 0, "precision": 0, "recall": 0, "specificity": 0, "mcc": 0}
        batch_count = 0

        for batch_idx, (data, labels) in enumerate(train_loader):
            data, labels = data.to(rank), labels.to(rank)
            optimizer.zero_grad()
            outputs = ddp_model(data)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            batch_metrics = compute_metrics(outputs, labels, rank, world_size)
            for metric in batch_metrics:
                total_metrics[metric] += batch_metrics[metric]
            
            batch_count += 1

            if batch_idx % config.logging_freq == 0:
                print(f"Epoch [{epoch}/{config.epochs}], Batch [{batch_idx}], Loss: {loss.item()}")

        # Average metrics over all batches
        for metric in total_metrics:
            total_metrics[metric] /= batch_count

        print(f"Epoch [{epoch}/{config.epochs}]: Loss = {total_loss / batch_count:.4f}, Accuracy = {total_metrics['accuracy']:.4f}, Precision = {total_metrics['precision']:.4f}, Recall = {total_metrics['recall']:.4f}, Specificity = {total_metrics['specificity']:.4f}, MCC = {total_metrics['mcc']:.4f}")
        
        # Save checkpoint
        if rank == 0:
            torch.save(ddp_model.state_dict(), f"model_epoch_{epoch}.pth")

# Evaluate function for validation
def evaluate(ddp_model, val_loader, config, rank, world_size):
    ddp_model.eval()
    total_metrics = {"accuracy": 0, "precision": 0, "recall": 0, "specificity": 0, "mcc": 0}
    batch_count = 0

    with torch.no_grad():
        for data, labels in val_loader:
            data, labels = data.to(rank), labels.to(rank)
            outputs = ddp_model(data)
            batch_metrics = compute_metrics(outputs, labels, rank, world_size)
            for metric in batch_metrics:
                total_metrics[metric] += batch_metrics[metric]
            batch_count += 1

    # Average metrics over all batches
    for metric in total_metrics:
        total_metrics[metric] /= batch_count

    print(f"Validation: Accuracy = {total_metrics['accuracy']:.4f}, Precision = {total_metrics['precision']:.4f}, Recall = {total_metrics['recall']:.4f}, Specificity = {total_metrics['specificity']:.4f}, MCC = {total_metrics['mcc']:.4f}")

# Main function to train and evaluate
def main():
    # Parse arguments
    args = parse_args()
    
    # Setup DDP and DataLoader
    rank = args.rank
    world_size = args.world_size
    ddp_model = setup_ddp(rank, world_size, args)
    train_loader = create_dataloader(args.batch_size, rank, world_size)

    # Train and evaluate
    train(ddp_model, train_loader, args, rank, world_size)

    # Optionally evaluate on validation data
    # val_loader = create_dataloader(args.batch_size, rank, world_size)  # Load validation data
    # evaluate(ddp_model, val_loader, args, rank, world_size)

if __name__ == "__main__":
    main()
