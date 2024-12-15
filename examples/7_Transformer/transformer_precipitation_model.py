import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# ==========================================================
# Data Preparation
# ==========================================================

# Simulate latitude, longitude, and corresponding precipitation data
np.random.seed(42)
num_points = 1000

# Generate random latitudes [-90, 90] and longitudes [-180, 180]
latitude = np.random.uniform(-90, 90, num_points)
longitude = np.random.uniform(-180, 180, num_points)

# Simulate precipitation as a function of latitude and longitude (with added noise)
precipitation = np.sin(np.radians(latitude)) + np.cos(np.radians(longitude)) + np.random.normal(0, 0.1, num_points)

# Convert to PyTorch tensors
inputs = torch.tensor(np.stack([latitude, longitude], axis=1), dtype=torch.float32)
targets = torch.tensor(precipitation, dtype=torch.float32).unsqueeze(1)

# Split data into training and testing sets
train_size = int(0.8 * num_points)
test_size = num_points - train_size

train_dataset = TensorDataset(inputs[:train_size], targets[:train_size])
test_dataset = TensorDataset(inputs[train_size:], targets[train_size:])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)

# ==========================================================
# Model Definition
# ==========================================================

class TransformerPrecipitationModel(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, output_dim):
        super(TransformerPrecipitationModel, self).__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, output_dim)

    def forward(self, x):
        # Add sequence length dimension
        x = self.embedding(x.unsqueeze(1))  # Shape: (batch_size, sequence_length=1, d_model)
        x = self.transformer(x)  # Shape: (batch_size, sequence_length=1, d_model)
        x = x.squeeze(1)  # Remove sequence length dimension
        return self.fc(x)

# Initialize model
device = "cpu"  # Use CPU for computation
model = TransformerPrecipitationModel(input_dim=2, d_model=64, nhead=4, num_layers=2, output_dim=1).to(device)

# ==========================================================
# Training
# ==========================================================

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training function
def train_model(model, train_loader, criterion, optimizer, epochs=50):
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_inputs, batch_targets in train_loader:
            batch_inputs, batch_targets = batch_inputs.to(device), batch_targets.to(device)
            optimizer.zero_grad()
            predictions = model(batch_inputs)
            loss = criterion(predictions, batch_targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {total_loss / len(train_loader):.4f}")

# Train the model
train_model(model, train_loader, criterion, optimizer, epochs=50)

# ==========================================================
# Testing
# ==========================================================

# Evaluation function
def evaluate_model(model, test_loader):
    model.eval()
    predictions, actuals = [], []
    with torch.no_grad():
        for batch_inputs, batch_targets in test_loader:
            batch_inputs, batch_targets = batch_inputs.to(device), batch_targets.to(device)
            batch_predictions = model(batch_inputs)
            predictions.append(batch_predictions.cpu().numpy())
            actuals.append(batch_targets.cpu().numpy())

    predictions = np.concatenate(predictions)
    actuals = np.concatenate(actuals)
    return predictions, actuals

# Get test results
predictions, actuals = evaluate_model(model, test_loader)

# Print test results
print("\nTest Results:")
print("Mean Squared Error:", np.mean((predictions - actuals) ** 2))

# ==========================================================
# Testing with New Input Data
# ==========================================================

# Example test inputs (latitude and longitude)
test_latitudes = torch.tensor([0.0, 45.0, -45.0, 90.0, -90.0], dtype=torch.float32)
test_longitudes = torch.tensor([0.0, 90.0, -90.0, 180.0, -180.0], dtype=torch.float32)

new_inputs = torch.stack([test_latitudes, test_longitudes], dim=1).to(device)

# Inference
with torch.no_grad():
    new_predictions = model(new_inputs)

print("\nNew Test Inputs (Latitude and Longitude):")
print(new_inputs.cpu().numpy())
print("Predicted Precipitation:")
print(new_predictions.cpu().numpy())

# ==========================================================
# Save Model as TorchScript
# ==========================================================

# Convert model to TorchScript
scripted_model = torch.jit.script(model)
scripted_model.save("transformer_precipitation_model.pt")

print("\nTorchScript model saved as 'transformer_precipitation_model.pt'")
