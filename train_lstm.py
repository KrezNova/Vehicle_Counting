import torch
import torch.nn as nn
import torch.optim as optim
from lstm_model import LSTMModel

# Define dataset
class TrajectoryDataset(torch.utils.data.Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return torch.tensor(self.sequences[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.float32)

# Example data (replace with real labeled data)
sequences = [
    [[0, 0], [1, 1], [2, 2], [3, 3], [4, 4]],  # Crossing
    [[0, 0], [0, 1], [0, 2], [0, 3], [0, 4]],  # Not crossing
    [[2, 0], [3, 1], [4, 2], [5, 3], [6, 4]],  # Crossing
    [[4, 0], [4, 1], [4, 2], [4, 3], [4, 4]],  # Not crossing
]
labels = [1, 0, 1, 0]

dataset = TrajectoryDataset(sequences, labels)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True)

# Initialize LSTM model
lstm_model = LSTMModel(input_size=2, hidden_size=64, output_size=1, num_layers=2)
criterion = nn.BCELoss()  # Binary Cross-Entropy Loss
optimizer = optim.Adam(lstm_model.parameters(), lr=0.0001)

# Train the model
for epoch in range(50):  # Train for 50 epochs
    epoch_loss = 0
    for inputs, targets in dataloader:
        outputs = lstm_model(inputs)
        loss = criterion(outputs.squeeze(), targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    print(f"Epoch {epoch+1}/50, Loss: {epoch_loss:.4f}")

# Save the trained model
torch.save(lstm_model.state_dict(), 'lstm_model.pth')
print("LSTM model saved to 'lstm_model.pth'")