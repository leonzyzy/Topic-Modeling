# Initialize an empty list to store the result
expanded_data = []

# For each item in the dataset (transactions)
for i in range(data.shape[0]):
    # Get current transaction
    transaction = data[i]

    # Create a tensor to hold the final output (50 rows, 46 features)
    padded_data = torch.zeros(desired_size, num_features)

    # Copy the transaction into the first rows
    padded_data[:transaction.shape[0], :] = transaction

    # Append the padded data to the list
    expanded_data.append(padded_data)

# Stack all the expanded data into a single tensor
expanded_data = torch.stack(expanded_data)
