import os
import logging
import torch
import pytz
from datetime import datetime

# Custom formatter to log time in US Eastern Time (EST)
class ESTFormatter(logging.Formatter):
    def formatTime(self, record, datefmt=None):
        # Define the EST timezone
        eastern = pytz.timezone('US/Eastern')
        # Get the current time in UTC and convert to EST
        utc_time = datetime.utcfromtimestamp(record.created)
        utc_time = pytz.utc.localize(utc_time)
        est_time = utc_time.astimezone(eastern)
        # Return formatted time string
        return est_time.strftime('%Y-%m-%d %H:%M:%S')

# Step 1: Create a directory for logs (if it doesn't exist)
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)  # Creates folder if it doesn't exist

# Step 2: Configure logging to save logs to a file in the "logs" folder
logging.basicConfig(
    level=logging.INFO,  # Set the logging level
    format='%(asctime)s - %(levelname)s - %(message)s',  # Log format
    handlers=[
        logging.StreamHandler(),  # Log to console
        logging.FileHandler(os.path.join(log_dir, 'training.log'))  # Log to file inside 'logs' folder
    ]
)

# Example: Setup dummy model and training loop
def train_model():
    model = torch.nn.Sequential(torch.nn.Linear(10, 32), torch.nn.ReLU(), torch.nn.Linear(32, 2)).cuda()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Example dummy data
    train_data = torch.randn(100, 10).cuda()
    train_labels = torch.randint(0, 2, (100,)).cuda()

    epochs = 10
    for epoch in range(epochs):
        model.train()

        # Simulate loss and metric calculation
        optimizer.zero_grad()
        outputs = model(train_data)
        loss = criterion(outputs, train_labels)
        loss.backward()
        optimizer.step()

        # Example metric (accuracy)
        _, predicted = torch.max(outputs, 1)
        accuracy = (predicted == train_labels).float().mean().item()

        # Step 3: Log epoch, loss, and accuracy to file and console
        logging.info(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}, Accuracy: {accuracy:.4f}")

# Train the model
train_model()
