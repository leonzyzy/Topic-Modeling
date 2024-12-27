import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
from torch.nn.parallel import DistributedDataParallel as DDP
import argparse

# Simple Model Definition (Example MLP)
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(100, 100)
        self.fc2 = nn.Linear(100, 10)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Dataset (Dummy Data)
class DummyDataset(Dataset):
    def __init__(self, size=1000):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return torch.randn(100), torch.randint(0, 10, (1,))

# 1. Setup DDP
def setup_ddp(rank, world_size, args):
    """ Initializes the process group for DDP and assigns devices """
    dist.init_process_group(backend='nccl', 
                            init_method=f'tcp://{args.master_addr}:{args.master_port}', 
                            rank=rank, 
                            world_size=world_size)
    
    # Set the device based on rank
    device = torch.device(f'cuda:{rank % torch.cuda.device_count()}')
    torch.cuda.set_device(device)
    return device

# 2. Create DataLoader with Distributed Sampler
def create_dataloader(rank, world_size, batch_size=32):
    """ Creates DataLoader with DistributedSampler for multi-GPU """
    dataset = DummyDataset()
    train_sampler = torch.utils.data.DistributedSampler(dataset, 
                                                        num_replicas=world_size, 
                                                        rank=rank)
    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
    return train_loader, train_sampler

# 3. Train the model
def train(rank, world_size, args, device):
    """ Train the model with Distributed Data Parallel (DDP) """
    model = SimpleModel().to(device)
    model = DDP(model, device_ids=[rank % torch.cuda.device_count()])

    # Create DataLoader with Distributed Sampler
    train_loader, train_sampler = create_dataloader(rank, world_size)

    # Optimizer and Loss function
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(5):  # Training for 5 epochs
        train_sampler.set_epoch(epoch)
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target.squeeze())
            loss.backward()
            optimizer.step()

        if rank == 0:  # Only print from the master node
            print(f"Epoch [{epoch+1}/5], Loss: {loss.item():.4f}")

    dist.barrier()  # Synchronize all processes at the end of training

# 4. Main function to launch the training process
def main():
    """ Main function to initialize and start the distributed training process """
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--master_addr', type=str, default='127.0.0.1', help='Master node address')
    parser.add_argument('--master_port', type=str, default='29500', help='Master node port')
    parser.add_argument('--nnodes', type=int, default=2, help='Number of nodes')
    parser.add_argument('--nproc_per_node', type=int, default=4, help='Number of processes per node')
    args = parser.parse_args()

    world_size = args.nnodes * args.nproc_per_node

    # Launch DDP training on multi-node and multi-GPU setup using torchrun
    dist.init_process_group(backend='nccl', 
                            init_method=f'tcp://{args.master_addr}:{args.master_port}',
                            world_size=world_size, 
                            rank=0)  # Initialize process group

    rank = dist.get_rank()  # Get rank of the current process
    device = setup_ddp(rank, world_size, args)
    
    # Train the model
    train(rank, world_size, args, device)

if __name__ == "__main__":
    main()
