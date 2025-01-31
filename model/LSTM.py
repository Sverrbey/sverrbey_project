import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from model import LSTM
from sklearn.preprocessing import MinMaxScaler

# 1. Preprocessing the data and creating sequences
# Load your data
data = pd.read_csv('your_data.csv')

# Preprocess the data
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

# Convert to PyTorch tensors
data_tensor = torch.tensor(data_scaled, dtype=torch.float32)

# Create sequences  
def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:i+seq_length]
        y = data[i+seq_length]
        xs.append(x)
        ys.append(y)
    return torch.stack(xs), torch.stack(ys)

seq_length = 10
X, y = create_sequences(data_tensor, seq_length)

# Create DataLoader
dataset = TensorDataset(X, y)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# 2. Define the LSTM model
class LSTM_Model(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(LSTM_Model, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, self.hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(self.hidden_size, output_size)
    
    def forward(self, x):
        h_0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)
        c_0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)
        c_0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h_0, c_0))
        out = self.fc(out[:, -1, :])
        return out

input_size = data.shape[1]
hidden_size = 50
output_size = data.shape[1]
model = LSTM_Model(input_size, hidden_size, output_size)

# 3. Train the model
criterion = nn.MSELoss()
optimizer = optim.Adam(LSTM.parameters(),
                        lr=0.001)

num_epochs = 100
for epoch in range(num_epochs):
    for X_batch, y_batch in dataloader:
        optimizer.zero_grad()
        y_pred = LSTM(X_batch)
        loss = criterion(y_pred, y_batch)
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')

# 4. Evaluate the model
# Assuming you have a test dataset
test_data = pd.read_csv('your_test_data.csv')
test_data_scaled = scaler.transform(test_data)
test_data_tensor = torch.tensor(test_data_scaled, dtype=torch.float32)
X_test, y_test = create_sequences(test_data_tensor, seq_length)

LSTM.eval()
with torch.no_grad():
    y_pred = LSTM(X_test)
    test_loss = criterion(y_pred, y_test)
    print(f'Test Loss: {test_loss.item()}')

# Save the model 
torch.save(LSTM.state_dict(), 'model.pth')
