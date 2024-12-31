import torch
import torch.distributed as dist

def compute_metrics(outputs, labels, rank, world_size):
    # Convert predictions to binary (0 or 1)
    preds = (outputs > 0.5).int()
    
    # Calculate local TP, FP, TN, FN
    tp = torch.sum((preds == 1) & (labels == 1)).float().to(rank)
    fp = torch.sum((preds == 1) & (labels == 0)).float().to(rank)
    tn = torch.sum((preds == 0) & (labels == 0)).float().to(rank)
    fn = torch.sum((preds == 0) & (labels == 1)).float().to(rank)
    
    # Aggregate TP, FP, TN, FN across all processes
    dist.all_reduce(tp, op=dist.ReduceOp.SUM)
    dist.all_reduce(fp, op=dist.ReduceOp.SUM)
    dist.all_reduce(tn, op=dist.ReduceOp.SUM)
    dist.all_reduce(fn, op=dist.ReduceOp.SUM)
    
    # Compute global metrics
    epsilon = 1e-8
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    specificity = tn / (tn + fp + epsilon)
    mcc = (tp * tn - fp * fn) / (
        torch.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) + epsilon
    )
    
    # Print metrics (for rank 0)
    if rank == 0:
        print(f"Precision: {precision.item():.4f}")
        print(f"Recall: {recall.item():.4f}")
        print(f"Specificity: {specificity.item():.4f}")
        print(f"MCC: {mcc.item():.4f}")
    
    return precision.item(), recall.item(), specificity.item(), mcc.item()
