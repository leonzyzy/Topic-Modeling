import os
import torch
import torch.distributed as dist
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp

# Define a simple dataset for demonstration (replace with your real dataset)
class DummyDataset(Dataset):
    def __init__(self, num_samples=1000):
        self.num_samples = num_samples
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return torch.randn(3, 224, 224), torch.randint(0, 10, (1,))

# Define a simple model for demonstration (replace with your real model)
class DummyModel(nn.Module):
    def __init__(self):
        super(DummyModel, self).__init__()
        self.fc = nn.Linear(3 * 224 * 224, 10)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.fc(x)

def setup_ddp(rank, world_size):
    # Initialize the distributed environment
    dist.init_process_group(
        backend='nccl',
        init_method=f'tcp://{os.environ["MASTER_ADDR"]}:{os.environ["MASTER_PORT"]}',
        world_size=world_size,
        rank=rank
    )
    torch.cuda.set_device(rank)  # Set the device for each process

def create_dataloader(batch_size=32):
    # Create DataLoader and distribute across processes
    dataset = DummyDataset()
    sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)
    return dataloader

def train(model, dataloader, optimizer, epoch):
    model.train()
    for data, target in dataloader:
        data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
        optimizer.zero_grad()
        output = model(data)
        loss = nn.CrossEntropyLoss()(output, target)
        loss.backward()
        optimizer.step()
    
    print(f"Epoch {epoch} training complete")

def run(rank, world_size):
    # Set up distributed environment
    setup_ddp(rank, world_size)

    # Initialize the model, optimizer, and dataloader
    model = DummyModel().cuda()
    model = DDP(model, device_ids=[rank])

    optimizer = optim.SGD(model.parameters(), lr=0.001)

    dataloader = create_dataloader()

    # Training loop
    for epoch in range(10):
        train(model, dataloader, optimizer, epoch)

    dist.destroy_process_group()

def main():
    # Set environment variables for MASTER_ADDR and MASTER_PORT
    os.environ['MASTER_ADDR'] = 'localhost'  # Set to master node IP
    os.environ['MASTER_PORT'] = '29500'
    
    # Number of processes per node (GPUs)
    world_size = 2  # Change this to the total number of processes across all nodes

    # Spawn processes for multi-GPU DDP
    mp.spawn(run, args=(world_size,), nprocs=world_size, join=True)

if __name__ == "__main__":
    main()
