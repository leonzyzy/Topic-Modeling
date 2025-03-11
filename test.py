# Initialize a list to accumulate the rows
expanded_data = []

# Accumulate the rows progressively
for i in range(1, desired_size + 1):
    # Limit the size to the number of available transactions (3 in this case)
    temp_data = data[:min(i, data.shape[0])]
    
    # Add padding (zero rows) if needed
    if temp_data.shape[0] < i:
        padding = torch.zeros(i - temp_data.shape[0], num_features)
        temp_data = torch.cat((temp_data, padding), dim=0)
    
    expanded_data.append(temp_data)

# Convert the accumulated data into a single tensor
expanded_data = torch.cat(expanded_data, dim=0)
