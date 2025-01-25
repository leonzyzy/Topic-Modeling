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

            # Print loss for each rank
            print(f"[Rank {rank}] Epoch {epoch + 1}, Loss: {loss.item()}")

    cleanup_ddp()
