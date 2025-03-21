import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.nn.parallel import DistributedDataParallel as DDP
import torchmetrics

class DummyDataset(Dataset):
    def __init__(self, size=1000):
        self.data = torch.randn(size, 10)
        self.labels = torch.randint(0, 2, (size,))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

def setup(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def train(rank, world_size):
    setup(rank, world_size)

    # Dummy model
    model = nn.Linear(10, 1).to(rank)
    model = DDP(model, device_ids=[rank])

    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Metrics with DDP support
    metrics = torchmetrics.MetricCollection({
        "accuracy": torchmetrics.Accuracy(task="binary", sync_on_compute=True),
        "precision": torchmetrics.Precision(task="binary", sync_on_compute=True),
        "recall": torchmetrics.Recall(task="binary", sync_on_compute=True),
        "f1": torchmetrics.F1Score(task="binary", sync_on_compute=True),
        "f0.5": torchmetrics.FBetaScore(task="binary", beta=0.5, sync_on_compute=True),
        "auroc": torchmetrics.AUROC(task="binary", sync_on_compute=True),
        "pr_auc": torchmetrics.AveragePrecision(task="binary", sync_on_compute=True),
    }).to(rank)

    # DataLoader with DistributedSampler
    dataset = DummyDataset()
    sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(dataset, batch_size=32, sampler=sampler)

    for epoch in range(5):  # Training loop
        model.train()
        for batch in dataloader:
            x, y = batch
            x, y = x.to(rank), y.to(rank).float()

            optimizer.zero_grad()
            logits = model(x).squeeze()
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            # Compute metrics
            preds = torch.sigmoid(logits) > 0.5  # Convert logits to binary predictions
            metrics.update(preds, y.int())

        # Gather results across all GPUs
        dist.barrier()
        if rank == 0:  # Only print on main process
            results = metrics.compute()
            print(f"Epoch {epoch + 1}: {results}")
            metrics.reset()  # Reset metrics for next epoch

    cleanup()

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    torch.multiprocessing.spawn(train, args=(world_size,), nprocs=world_size, join=True)
