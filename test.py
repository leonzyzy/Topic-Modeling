def evaluate(model, dataloader, device):
    model.eval()
    
    # Sync metrics across processes
    accuracy_metric = torchmetrics.classification.Accuracy(dist_sync_on_step=True).to(device)
    precision_metric = torchmetrics.classification.Precision(num_classes=2, dist_sync_on_step=True).to(device)
    recall_metric = torchmetrics.classification.Recall(num_classes=2, dist_sync_on_step=True).to(device)
    f1_metric = torchmetrics.classification.F1Score(num_classes=2, dist_sync_on_step=True).to(device)
    f05_metric = torchmetrics.classification.FBetaScore(beta=0.5, num_classes=2, dist_sync_on_step=True).to(device)
    roc_auc_metric = torchmetrics.classification.ROCAUC(num_classes=2, dist_sync_on_step=True).to(device)
    pr_auc_metric = torchmetrics.classification.AveragePrecision(num_classes=2, dist_sync_on_step=True).to(device)

    # Collect metrics
    with torch.no_grad():
        batch_counter = 0

        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)

            # Update metrics
            accuracy_metric.update(preds, labels)
            precision_metric.update(preds, labels)
            recall_metric.update(preds, labels)
            f1_metric.update(preds, labels)
            f05_metric.update(preds, labels)
            roc_auc_metric.update(preds, labels)
            pr_auc_metric.update(preds, labels)

            batch_counter += 1

            # Print metrics every 10 batches
            if batch_counter % 10 == 0 and dist.get_rank() == 0:  # Only print from rank 0
                print(f"Batch {batch_counter}:")
                print(f"  Accuracy: {accuracy_metric.compute():.4f}")
                print(f"  Precision: {precision_metric.compute():.4f}")
                print(f"  Recall: {recall_metric.compute():.4f}")
                print(f"  F1 Score: {f1_metric.compute():.4f}")
                print(f"  F0.5 Score: {f05_metric.compute():.4f}")
                print(f"  ROC AUC: {roc_auc_metric.compute():.4f}")
                print(f"  PR AUC: {pr_auc_metric.compute():.4f}")

        # Barrier to synchronize processes before final metric computation
        dist.barrier()

        # Final cumulative metrics after all batches
        if dist.get_rank() == 0:  # Only print from rank 0
            print("\nFinal Metrics after all batches:")
            print(f"  Accuracy: {accuracy_metric.compute():.4f}")
            print(f"  Precision: {precision_metric.compute():.4f}")
            print(f"  Recall: {recall_metric.compute():.4f}")
            print(f"  F1 Score: {f1_metric.compute():.4f}")
            print(f"  F0.5 Score: {f05_metric.compute():.4f}")
            print(f"  ROC AUC: {roc_auc_metric.compute():.4f}")
            print(f"  PR AUC: {pr_auc_metric.compute():.4f}")
