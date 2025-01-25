for epoch in range(args["epochs"]):
    model.train()
    train_sampler.set_epoch(epoch)
    total_train_loss = 0
    total_val_loss = 0

    # Ensure validation loader is iterable
    val_iter = iter(val_loader)

    for batch_idx, (train_data, train_target) in enumerate(train_loader):
        # Training step
        train_data, train_target = train_data.cuda(rank), train_target.cuda(rank)
        optimizer.zero_grad()
        train_output = model(train_data)
        train_loss = criterion(train_output, train_target)
        train_loss.backward()
        optimizer.step()
        total_train_loss += train_loss.item()

        # Fetch the next validation batch
        try:
            val_data, val_target = next(val_iter)
            val_data, val_target = val_data.cuda(rank), val_target.cuda(rank)
            model.eval()  # Switch to evaluation mode for validation
            with torch.no_grad():
                val_output = model(val_data)
                val_loss = criterion(val_output, val_target)
                total_val_loss += val_loss.item()
            
            model.train()  # Switch back to training mode

            if rank == 0:  # Log the training and validation loss
                print(f"Epoch {epoch + 1}, Batch {batch_idx + 1}: "
                      f"Train Loss = {train_loss.item():.4f}, Val Loss = {val_loss.item():.4f}")
        
        except StopIteration:
            # If the validation dataset is smaller and we run out of batches, skip validation for remaining train batches
            if rank == 0:
                print(f"Epoch {epoch + 1}, Batch {batch_idx + 1}: "
                      f"Train Loss = {train_loss.item():.4f}, Val Loss = N/A")

    # Compute average loss for the epoch
    avg_train_loss = total_train_loss / len(train_loader)
    avg_val_loss = total_val_loss / len(val_loader)

    if rank == 0:
        print(f"Epoch {epoch + 1} Summary: Avg Train Loss = {avg_train_loss:.4f}, Avg Val Loss = {avg_val_loss:.4f}")
    
    # Save model at the end of each epoch
    if rank == 0:
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            torch.save(model.module.state_dict(), f"model_epoch_{epoch + 1}.pt")
        else:
            torch.save(model.state_dict(), f"model_epoch_{epoch + 1}.pt")
