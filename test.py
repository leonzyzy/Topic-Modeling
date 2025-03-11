 # To hold the expanded data
        expanded_data = []

        # Iterate through each transaction
        for i in range(account_data.shape[0]):
            cumulative_data = account_data[:i + 1]  # Get all transactions up to the current one

            # Pad with zeros if necessary
            padding_size = self.max_transactions - cumulative_data.shape[0]
            padded_data = torch.cat([cumulative_data, torch.zeros(padding_size, account_data.shape[1])], dim=0)
            
            # Append to the expanded data list
            expanded_data.append(padded_data)

        # Convert the expanded data into a tensor (shape: (max_transactions, num_features))
        expanded_data = torch.stack(expanded_data)

        return expanded_data, target
