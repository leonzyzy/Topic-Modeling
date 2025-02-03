import pandas as pd
import json
from sklearn.model_selection import train_test_split

# Load your DataFrame (assuming it's already loaded)
# df = pd.read_csv("your_file.csv")  # Uncomment if loading from CSV

# Define a fixed instruction
instruction = "Given the following plan description, classify its containment type."

# Ensure at least one sample from each class is in the test set
test_samples = df.groupby("label").apply(lambda x: x.sample(1, random_state=42))
test_samples = test_samples.droplevel(0)  # Remove groupby index

# Remaining data after reserving one per class
remaining_df = df.drop(test_samples.index)

# Split remaining data into train (90%) and test (10%)
train_df, extra_test_df = train_test_split(remaining_df, test_size=0.1, random_state=42)

# Combine the manually selected test samples with extra test data
test_df = pd.concat([test_samples, extra_test_df])

# Function to convert DataFrame to JSON format for fine-tuning
def df_to_json(df, filename):
    data = [
        {
            "instruction": instruction,
            "input": row["input"],  # Adjust column names if necessary
            "output": row["label"]
        }
        for _, row in df.iterrows()
    ]
    with open(filename, "w") as f:
        json.dump(data, f, indent=4)
    print(f"{filename} saved successfully!")

# Save train and test sets
df_to_json(train_df, "train.json")
df_to_json(test_df, "test.json")
