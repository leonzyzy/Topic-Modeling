import torchmetrics
from torchmetrics.classification import Accuracy, Precision, Recall, F1Score, FBetaScore, AveragePrecision, AUROC

# Initialize binary classification metrics with task='binary'
accuracy_metric = Accuracy(task='binary', dist_sync_on_step=True).to(device)
precision_metric = Precision(task='binary', dist_sync_on_step=True).to(device)
recall_metric = Recall(task='binary', dist_sync_on_step=True).to(device)
f1_metric = F1Score(task='binary', dist_sync_on_step=True).to(device)
f05_metric = FBetaScore(beta=0.5, task='binary', dist_sync_on_step=True).to(device)
auroc_metric = AUROC(task='binary', dist_sync_on_step=True).to(device)
pr_auc_metric = AveragePrecision(task='binary', dist_sync_on_step=True).to(device)
