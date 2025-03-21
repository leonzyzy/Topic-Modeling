    accuracy = Accuracy(task="binary", dist_sync_on_step=True).to(rank)
    precision = Precision(task="binary", dist_sync_on_step=True).to(rank)
    recall = Recall(task="binary", dist_sync_on_step=True).to(rank)
    f1 = F1Score(task="binary", dist_sync_on_step=True).to(rank)
    f0_5 = FBetaScore(task="binary", beta=0.5, dist_sync_on_step=True).to(rank)
    auroc = AUROC(task="binary", dist_sync_on_step=True).to(rank)
    pr_auc = AveragePrecision(task="binary", dist_sync_on_step=True).to(rank)  # PR AUC
