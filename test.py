import torch
import seaborn as sns
import matplotlib.pyplot as plt

# Assuming model is already on GPU (e.g., model.to(device))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Extract attention weights using model.named_parameters()
attention_weights = {}

# Loop through named parameters
for name, param in model.named_parameters():
    if 'self_attn' in name:  # Identify attention parameters
        print(f"Found {name} of size {param.size()}")
        attention_weights[name] = param

# Visualize one of the self-attention weights (for example, in_proj_weight)
# Assuming 'self_attn.in_proj_weight' exists and is a matrix of shape [3 * d_model, d_model]
if 'encoder.layers.0.self_attn.in_proj_weight' in attention_weights:
    in_proj_weight = attention_weights['encoder.layers.0.self_attn.in_proj_weight']

    # Move to CPU if it's on GPU
    in_proj_weight_cpu = in_proj_weight.cpu().detach().numpy()

    # Visualize as heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(in_proj_weight_cpu, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Attention Weights (in_proj_weight)")
    plt.xlabel("Input Features")
    plt.ylabel("Attention Weights")
    plt.show()
