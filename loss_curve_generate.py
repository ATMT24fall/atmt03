import re
import matplotlib.pyplot as plt

# !!change the log file path and the save path to your own!!

# Load the log file
log_file_path = (
    "assignment03/baseline/lr_0.00025/log.txt"  # Update this with your log file name
)
epochs = []
train_loss = []
valid_loss = []

# Define regex patterns to match loss and valid_loss lines in your logs
# train_loss_pattern = re.compile(r"Epoch\s+(\d+).*?loss:\s*([\d.]+)")
train_loss_pattern = re.compile(r"Epoch\s+(\d+):\s+loss\s+([\d.]+)")

# valid_loss_pattern = re.compile(r"valid_loss:\s*([\d.]+)")
valid_loss_pattern = re.compile(r"valid_loss\s+([\d.]+)")

# Read through the log and extract losses
with open(log_file_path, "r") as file:
    for line in file:
        train_match = train_loss_pattern.search(line)
        valid_match = valid_loss_pattern.search(line)

        # Extract training loss
        if train_match:
            epoch = int(train_match.group(1))
            loss = float(train_match.group(2))
            epochs.append(epoch)
            train_loss.append(loss)

        # Extract validation loss (assuming it appears after train loss for each epoch)
        if valid_match:
            val_loss = float(valid_match.group(1))
            valid_loss.append(val_loss)

# Plotting the curves
plt.figure(figsize=(10, 6))
plt.plot(epochs, train_loss, label="Training Loss", marker="o")
plt.plot(epochs, valid_loss, label="Validation Loss", marker="o")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss Curves")
plt.legend()
plt.grid(True)
plt.savefig("assignment03/baseline/lr_0.00025/loss_curves.png", format="png")
