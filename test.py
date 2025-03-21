import os
import sys
import torch
import torch.distributed as dist
import logging
from logging.handlers import RotatingFileHandler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torchmetrics.classification import Accuracy, Precision, Recall, F1Score, AUROC, AveragePrecision

# Setup logging with rotating files
def setup_logging(rank):
    log_filename = "metrics.log"
    
    # Create a rotating file handler (10MB per file, keeps last 5 files)
    file_handler = RotatingFileHandler(log_filename, maxBytes=10 * 1024 * 1024, backupCount=5)
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s", "%Y-%m-%d %H:%M:%S")
    file_handler.setFormatter(formatter)

    logging.basicConfig(
        level=logging.INFO,
        handlers=[file_handler] if rank == 0 else []
    )

    if rank == 0:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logging.getLogger().addHandler(console_handler)

# Custom dataset class
class MyDataset(torch.utils.data.Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# Inference function
def evaluate():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    # Initialize process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    # Setup logging
    setup_logging(rank)

    # Load model
    model_path = "model.pt"
    model = torch.load(model_path, map_location=f"cuda:{rank}")
    model = model.to(rank)
    model = DDP(model, device_ids=[rank])

    # Create dataset and dataloader
    dataset = MyDataset(data=torch.rand(1000, 10), labels=torch.randint(0, 2, (1000,)))
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False)
    dataloader = DataLoader(dataset, batch_size=64, sampler=sampler, num_workers=4)

    # Initialize TorchMetrics
    accuracy = Accuracy(task="binary").to(rank)
    precision = Precision(task="binary").to(rank)
    recall = Recall(task="binary").to(rank)
    f1 = F1Score(task="binary").to(rank)
    f0_5 = F1Score(task="binary", beta=0.5).to(rank)  # F0.5-score
    auroc = AUROC(task="binary").to(rank)
    pr_auc = AveragePrecision(task="binary").to(rank)

    # Inference loop
    model.eval()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(rank), targets.to(rank)

            # Forward pass
            outputs = model(inputs)
            preds = torch.sigmoid(outputs) > 0.5  # Convert logits to binary predictions

            # Compute batch-wise metrics
            acc = accuracy(preds, targets)
            prec = precision(preds, targets)
            rec = recall(preds, targets)
            f1_score = f1(preds, targets)
            f0_5_score = f0_5(preds, targets)
            auroc_score = auroc(preds, targets)
            pr_auc_score = pr_auc(preds, targets)

            if rank == 0:
                batch_metrics_msg = (
                    f"[Batch {batch_idx}] Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, "
                    f"F1: {f1_score:.4f}, F0.5: {f0_5_score:.4f}, AUROC: {auroc_score:.4f}, PR AUC: {pr_auc_score:.4f}"
                )
                logging.info(batch_metrics_msg)
                print(batch_metrics_msg)

    # Compute final aggregated metrics
    acc = accuracy.compute()
    prec = precision.compute()
    rec = recall.compute()
    f1_score = f1.compute()
    f0_5_score = f0_5.compute()
    auroc_score = auroc.compute()
    pr_auc_score = pr_auc.compute()

    if rank == 0:
        final_metrics_msg = (
            f"Final Metrics - Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, "
            f"F1: {f1_score:.4f}, F0.5: {f0_5_score:.4f}, AUROC: {auroc_score:.4f}, PR AUC: {pr_auc_score:.4f}"
        )
        logging.info(final_metrics_msg)
        print(final_metrics_msg)

    # Clean up
    dist.destroy_process_group()

if __name__ == "__main__":
    evaluate()
