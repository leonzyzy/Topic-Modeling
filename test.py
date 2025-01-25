import os
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, TensorDataset

def setup_ddp():
    """Initializes the process group for distributed training."""
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

def cleanup_ddp():
    """Cleans up the process group."""
    dist.destroy_process_group()

class SimpleModel(nn.Module):
    """A simple feedforward neural network."""
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

def train(rank, world_size, epochs=5):
    setup_ddp()

    # Device setup
    device = torch.device(f"cuda:{int(os.environ['LOCAL_RANK'])}")

    # Dataset and DataLoader
    input_size, hidden_size, output_size = 10, 50, 1
    batch_size = 32

    # Create dummy data
    num_samples = 1000
    x = torch.randn(num_samples, input_size)
    y = torch.randn(num_samples, output_size)

    dataset = TensorDataset(x, y)
    sampler = DistributedSampler(dataset)
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)

    # Model, loss, and optimizer
    model = SimpleModel(input_size, hidden_size, output_size).to(device)
    model = DDP(model, device_ids=[int(os.environ["LOCAL_RANK"])])
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # Training loop
    for epoch in range(epochs):
        sampler.set_epoch(epoch)  # Shuffle data across ranks
        for batch_x, batch_y in dataloader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

        if rank == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}")

    cleanup_ddp()

if __name__ == "__main__":
    # Run with: torchrun --standalone --nproc_per_node=NUM_GPUS script.py
    rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    train(rank, world_size)
