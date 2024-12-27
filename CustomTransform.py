import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import torch.nn as nn

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(10, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, x):
        return self.net(x)

def setup_ddp():
    """Initialize distributed training"""
    # Initialize process group
    dist.init_process_group("nccl")
    
    # Get rank and world size
    rank = int(os.environ["LOCAL_RANK"])
    world_size = dist.get_world_size()
    
    # Set device
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    
    return rank, world_size, device

def train(rank, world_size, device):
    """Training process"""
    # Create model
    model = SimpleModel().to(device)
    model = DDP(model, device_ids=[rank])
    
    # Create dataset (replace with your data)
    train_data = torch.randn(1000, 10)
    train_labels = torch.randn(1000, 1)
    dataset = torch.utils.data.TensorDataset(train_data, train_labels)
    
    # Setup data loader
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    train_loader = DataLoader(
        dataset, 
        batch_size=32, 
        sampler=sampler,
        pin_memory=True
    )
    
    # Setup optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    # Training loop
    num_epochs = 5
    for epoch in range(num_epochs):
        model.train()
        sampler.set_epoch(epoch)
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.to(device)
            target = target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            if rank == 0 and batch_idx % 100 == 0:
                print(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item():.4f}')

def main():
    # Setup distributed training
    rank, world_size, device = setup_ddp()
    
    try:
        # Run training
        train(rank, world_size, device)
    except Exception as e:
        print(f"Error in rank {rank}: {str(e)}")
        raise e
    finally:
        # Cleanup
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
