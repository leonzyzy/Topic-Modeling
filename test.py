import torch

# Example input tensor of shape (batch_size, num_transactions, num_features)
data = torch.tensor([[[1, 2, 3],  
                      [4, 5, 6],  
                      [7, 8, 9]],

                     [[1, 2, 3],  
                      [4, 5, 6],  
                      [7, 8, 9]],

                     [[1, 2, 3],  
                      [4, 5, 6],  
                      [7, 8, 9]]], dtype=torch.float)

# Define the max number of transactions
max_transactions = data.shape[1]  # Max transactions = 3 in this example

# Initialize a list to store the transformed rows
expanded_data = []

# Iterate over each batch (account)
for b in range(data.shape[0]):
    account_data = data[b]
    
    # Iterate through the number of transactions and build the cumulative sequence
    for i in range(1, max_transactions + 1):
        cumulative_transactions = account_data[:i]
        
        # If the number of transactions is less than max_transactions, pad it
        padding_size = max_transactions - cumulative_transactions.shape[0]
        padded_transactions = torch.cat([cumulative_transactions, torch.zeros(padding_size, data.shape[2])], dim=0)
        
        # Add the padded transactions as a separate entry in expanded_data
        expanded_data.append(padded_transactions)

# Convert list to a tensor
expanded_data = torch.stack(expanded_data)

# Output the transformed tensor
for t in expanded_data:
    print(t)
