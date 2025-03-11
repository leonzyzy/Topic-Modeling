import torch

# Original input tensor
data = torch.tensor([[1, 2, 3],  
                     [4, 5, 6],  
                     [7, 8, 9]], dtype=torch.float)

# Define the max number of rows
max_rows = 3

# Initialize a list to store the transformed rows
expanded_data = []

# Iterate through the rows and build up the cumulative dataset
for i in range(1, data.shape[0] + 1):
    cumulative_rows = data[:i]
    
    # If the number of rows is less than max_rows, pad it
    padding_size = max_rows - cumulative_rows.shape[0]
    padded_rows = torch.cat([cumulative_rows, torch.zeros(padding_size, data.shape[1])], dim=0)
    
    # Add the padded rows as a separate entry in expanded_data
    expanded_data.append(padded_rows)

# Convert list to a tensor
expanded_data = torch.stack(expanded_data)

# Output the transformed tensor
for t in expanded_data:
    print(t)
