import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import os
from src.function import extract_altitude_points, generate_contour_map, create_tin_from_clustered_csv, cluster_elevation_data
from model import model
from sklearn.preprocessing import MinMaxScaler

## 0. Calibration
if os.path.exists('../data/clustered_altitude_points.csv') == False:
    # Extracting altitude poits from tif file
    tfw_file_path = "../data/dtm50/dtm50_7002_50m_33.tfw"
    tif_file_path = "../data/dtm50/dtm50_7002_50m_33.tif"
    output_csv_path = "../data/altitude_points.csv"  # Replace with desired CSV file path or set to None

    altitude_points = extract_altitude_points(tif_file_path, output_csv_path)

    # Display a preview of the altitude points
    if altitude_points is not None:
        print(altitude_points.head())

    ''' Create contour map from tif file 
    generate_contour_map(tif_file_path, contour_interval=20)
    '''

    # Clustering elevation points 
    csv_file_path = "../data/altitude_points.csv"  # Replace with your CSV file path
    output_csv_path = "../data/clustered_altitude_points.csv"  # Replace or set to None if not saving
    clustered_df = cluster_elevation_data(csv_file_path, n_clusters=100, output_csv=output_csv_path)

    # Display a preview of the clustered data
    if clustered_df is not None:
        print(clustered_df.head())


    # TIN from clusted data
    clustered_csv_file_path = "clustered_altitude_points.csv"  # Replace with your clustered CSV file path
    output_csv_path = "simplified_tin_points.csv"  # Replace or set to None if not saving
    create_tin_from_clustered_csv(clustered_csv_file_path, output_csv=output_csv_path)


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




criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 100
for epoch in range(num_epochs):
    for X_batch, y_batch in dataloader:
        optimizer.zero_grad()
        y_pred = model(X_batch)
        loss = criterion(y_pred, y_batch)
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')

    # Assuming you have a test dataset
test_data = pd.read_csv('your_test_data.csv')
test_data_scaled = scaler.transform(test_data)
test_data_tensor = torch.tensor(test_data_scaled, dtype=torch.float32)
X_test, y_test = create_sequences(test_data_tensor, seq_length)

model.eval()
with torch.no_grad():
    y_pred = model(X_test)
    test_loss = criterion(y_pred, y_test)
    print(f'Test Loss: {test_loss.item()}')

# Save the model 
torch.save(model.state_dict(), 'model.pth')