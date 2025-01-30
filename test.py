import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Assuming `attn_matrix` is of shape [4, 286, 286] (4 heads)
attn_matrix = attention_matrices[0].cpu().detach().numpy()  # Move to CPU for visualization

# Select the top 47 weights for each attention matrix
top_k = 47

# Visualize each head's top 47 attention weights with larger labels and legend
for head in range(4):
    # Flatten the matrix to get all the attention weights
    flat_matrix = attn_matrix[head].flatten()
    
    # Get the indices of the top 47 weights based on absolute value
    top_indices = np.argsort(np.abs(flat_matrix))[-top_k:]
    
    # Print the indices (row, col) and corresponding attention values
    print(f"Top 47 Attention Weights for Head {head+1}:")
    for idx in top_indices:
        row, col = divmod(idx, attn_matrix[head].shape[1])  # Convert flat index to row and column
        print(f"Index: ({row}, {col}) - Attention Weight: {attn_matrix[head][row, col]:.4f}")
    
    # Create a matrix of the same shape but with only the top 47 weights
    top_matrix = np.zeros_like(attn_matrix[head])
    for idx in top_indices:
        row, col = divmod(idx, attn_matrix[head].shape[1])
        top_matrix[row, col] = attn_matrix[head][row, col]
    
    # Plot the heatmap of the top 47 weights
    plt.figure(figsize=(12, 10))  # Increase figure size for better readability
    ax = sns.heatmap(top_matrix, cmap="coolwarm", annot=False, fmt=".2f", linewidths=0.5)
    
    # Set axis labels and title
    ax.set_title(f"Top 47 Important Attention Weights - Head {head+1}", fontsize=18)
    ax.set_xlabel("Input Features (286)", fontsize=14)
    ax.set_ylabel("Input Features (286)", fontsize=14)
    
    # Increase the size of the ticks on both axes
    ax.tick_params(axis='x', labelsize=12)
    ax.tick_params(axis='y', labelsize=12)
    
    # Add a color bar (legend) to the heatmap
    cbar = ax.collections[0].colorbar
    cbar.set_label('Attention Weight', fontsize=14)
    
    # Show the plot
    plt.show()
