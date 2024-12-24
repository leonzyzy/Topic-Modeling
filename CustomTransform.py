import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.nn.parallel import DistributedDataParallel as DDP


# 1. Simple Dataset
class SimpleDataset(Dataset):
    def __init__(self, num_samples=1000):
        self.data = torch.randn(num_samples, 10)
        self.labels = torch.randint(0, 2, (num_samples,))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


# 2. Define Simple Model
class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


# 3. Initialize DDP Environment
def setup_ddp(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    torch.distributed.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


# 4. Train Function
def train(rank, world_size):
    print(f"Running DDP on rank {rank}.")
    setup_ddp(rank, world_size)

    # Load Dataset
    dataset = SimpleDataset()
    sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(dataset, batch_size=32, sampler=sampler)

    # Model, Loss, and Optimizer
    model = SimpleNN(input_size=10, hidden_size=64, output_size=2).cuda(rank)
    model = DDP(model, device_ids=[rank])
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # Training Loop
    for epoch in range(5):
        sampler.set_epoch(epoch)  # Shuffle data for each epoch
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.cuda(rank), targets.cuda(rank)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            if batch_idx % 10 == 0 and rank == 0:
                print(f"Rank {rank}, Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item()}")

    # Cleanup
    torch.distributed.destroy_process_group()


# 5. Main Function (entry point)
def main():
    world_size = torch.cuda.device_count()  # Number of GPUs
    print(f"Using {world_size} GPUs")
    rank = int(os.environ['RANK'])  # This is set by torchrun
    world_size = int(os.environ['WORLD_SIZE'])  # This is set by torchrun
    train(rank, world_size)


if __name__ == "__main__":
    main()

