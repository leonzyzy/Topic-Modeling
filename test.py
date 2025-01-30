import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Assuming `attn_matrix` is of shape [286, 286] (a single attention matrix)
attn_matrix = attn_matrix.cpu().detach().numpy()  # If the matrix is on GPU, move it to CPU

# Select the top 50 weights based on absolute value
top_k = 50

# Flatten the matrix to get all the attention weights
flat_matrix = attn_matrix.flatten()

# Get the indices of the top 50 weights based on absolute value
top_indices = np.argsort(np.abs(flat_matrix))[-top_k:]

# Print the indices (row, col) and corresponding attention values
print(f"Top 50 Attention Weights:")
for idx in top_indices:
    row, col = divmod(idx, attn_matrix.shape[1])  # Convert flat index to row and column
    print(f"Index: ({row}, {col}) - Attention Weight: {attn_matrix[row, col]:.4f}")

# Create a matrix of the same shape but with only the top 50 weights
top_matrix = np.zeros_like(attn_matrix)
for idx in top_indices:
    row, col = divmod(idx, attn_matrix.shape[1])
    top_matrix[row, col] = attn_matrix[row, col]

# Plot the heatmap of the top 50 weights
plt.figure(figsize=(12, 10))  # Increase figure size for better readability
ax = sns.heatmap(top_matrix, cmap="coolwarm", annot=True, fmt=".2f", linewidths=0.5)

# Set axis labels and title
ax.set_title(f"Top 50 Important Attention Weights", fontsize=18)
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
