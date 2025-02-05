import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt

# 1. Preprocessing the data and creating sequences
# Load your data

## ptq.txt 
# Precipitation, Temperature, Runoff (10 years: 1981 01 01 - 1991 12 31)
df = pd.read_csv('/Users/SverreB/Github_Repo/sverrbey_project/src/data/HBV-land/Data/ptq.txt', header=1, delimiter='\t')
date_df = df[['date']]
data_0 = df[['Prec.','Temp','Qobs']]

## EVAP.txt 
# Evaporation (1 year: Jan - Dec)
df = pd.read_csv('/Users/SverreB/Github_Repo/sverrbey_project/src/data/HBV-land/Data/EVAP.txt')
data_1 = df

## T_MEAN.txt
# Mean temperature (1 year: Jan - Dec)
df = pd.read_csv('/Users/SverreB/Github_Repo/sverrbey_project/src/data/HBV-land/Data/T_MEAN.txt')   
data_2 = df

## clarea.xml
# Extract information from XML file
tree = ET.parse('/Users/SverreB/Github_Repo/sverrbey_project/src/data/HBV-land/Data/clarea.xml')
root = tree.getroot()

# Extract information
vegetation_zone_count = int(root.find('VegetationZoneCount').text)
elevation_zone_height = float(root.find('ElevationZoneHeight/double').text)

sub_catchment = root.find('SubCatchment')
vegetation_zone = float(sub_catchment.find('VegetationZone/EVU/Area').text)
lake_area = float(sub_catchment.find('Lake/Area').text)
lake_elevation = float(sub_catchment.find('Lake/Elevation').text)
lake_tt = float(sub_catchment.find('Lake/TT').text)
lake_sfcf = float(sub_catchment.find('Lake/SFCF').text)
absolute_area = float(sub_catchment.find('AbsoluteArea').text)

## Parameter_HBVland_start.xml
# Extract information from XML file
tree = ET.parse('/Users/SverreB/Github_Repo/sverrbey_project/src/data/HBV-land/Data/Parameter_HBVland_start.xml')
root = tree.getroot()

# Extract information
catchmentParameters = root.find('CatchmentParameters')
parameter_perc = int(catchmentParameters.find('PERC').text)
parameter_alpha = int(catchmentParameters.find('Alpha').text)
parameter_uzl = float(catchmentParameters.find('UZL').text)
parameter_k0 = float(catchmentParameters.find('K0').text)
parameter_k1 = float(catchmentParameters.find('K1').text)
parameter_k2 = float(catchmentParameters.find('K2').text)
parameter_maxbas = float(catchmentParameters.find('MAXBAS').text)
parameter_cet = float(catchmentParameters.find('Cet').text)
parameter_pcalt = float(catchmentParameters.find('PCALT').text)
parameter_tcalt = float(catchmentParameters.find('TCALT').text)
parameter_pelev = float(catchmentParameters.find('Pelev').text)
parameter_telev = float(catchmentParameters.find('Telev').text)
parameter_part = float(catchmentParameters.find('PART').text)
parameter_delay = float(catchmentParameters.find('DELAY').text)

vegetationZoneParamteres = root.find('VegetationZone/VegetationZoneParameters')
v_parameter_tt = int(vegetationZoneParamteres.find('TT').text)
v_parameter_cfmax = int(vegetationZoneParamteres.find('CFMAX').text)
v_parameter_sp= int(vegetationZoneParamteres.find('SP').text)
v_parameter_sfcf= int(vegetationZoneParamteres.find('SFCF').text)
v_parameter_cfr= float(vegetationZoneParamteres.find('CFR').text)
v_parameter_cwh= float(vegetationZoneParamteres.find('CWH').text)
v_parameter_cfglacier= int(vegetationZoneParamteres.find('CFGlacier').text)
v_parameter_cfslope= int(vegetationZoneParamteres.find('CFSlope').text)
v_parameter_fc= int(vegetationZoneParamteres.find('FC').text)
v_parameter_lp= int(vegetationZoneParamteres.find('LP').text)
v_parameter_beta= int(vegetationZoneParamteres.find('BETA').text)

subcatchmentParameters = root.find('SubCatchment/SubCatchmentParameters')
parameter_perc = int(subcatchmentParameters.find('PERC').text)
parameter_alpha = int(subcatchmentParameters.find('Alpha').text)
parameter_uzl = float(subcatchmentParameters.find('UZL').text)
parameter_k0 = float(subcatchmentParameters.find('K0').text)
parameter_k1 = float(subcatchmentParameters.find('K1').text)
parameter_k2 = float(subcatchmentParameters.find('K2').text)
parameter_maxbas = float(subcatchmentParameters.find('MAXBAS').text)
parameter_cet = float(subcatchmentParameters.find('Cet').text)
parameter_pcalt = float(subcatchmentParameters.find('PCALT').text)
parameter_tcalt = float(subcatchmentParameters.find('TCALT').text)
parameter_pelev = float(subcatchmentParameters.find('Pelev').text)
parameter_telev = float(subcatchmentParameters.find('Telev').text)
parameter_part = float(subcatchmentParameters.find('PART').text)
parameter_delay = float(subcatchmentParameters.find('DELAY').text)

subcatchmentvegetationZoneParamteres = root.find('SubCatchment/SubCatchmentVegetationZone/SubCatchmentVegetationZoneParameters')
v_parameter_tt = int(subcatchmentvegetationZoneParamteres.find('TT').text)
v_parameter_cfmax = int(subcatchmentvegetationZoneParamteres.find('CFMAX').text)
v_parameter_sp= int(subcatchmentvegetationZoneParamteres.find('SP').text)
v_parameter_sfcf= int(subcatchmentvegetationZoneParamteres.find('SFCF').text)
v_parameter_cfr= float(subcatchmentvegetationZoneParamteres.find('CFR').text)
v_parameter_cwh= float(subcatchmentvegetationZoneParamteres.find('CWH').text)
v_parameter_cfglacier= int(subcatchmentvegetationZoneParamteres.find('CFGlacier').text)
v_parameter_cfslope= int(subcatchmentvegetationZoneParamteres.find('CFSlope').text)
v_parameter_fc= int(subcatchmentvegetationZoneParamteres.find('FC').text)
v_parameter_lp= int(subcatchmentvegetationZoneParamteres.find('LP').text)
v_parameter_beta= int(subcatchmentvegetationZoneParamteres.find('BETA').text)

# Preprocess the data
scaler = MinMaxScaler()
data_0_scaled = scaler.fit_transform(data_0)
# Convert to PyTorch tensors
data_0_tensor = torch.tensor(data_0_scaled, dtype=torch.float32)

# Create sequences
def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:i+seq_length, :2]  # Use the first two columns as features
        y = data[i+seq_length, 2]     # Use the third column as the target
        xs.append(x)
        ys.append(y)
    return torch.stack(xs), torch.stack(ys)

# The sequence length determines the number of rows used to predict the next row
seq_length = 10
X, y = create_sequences(data_0_tensor, seq_length)

# Function to split data into training and test sets
def train_test_split_tensor(X, y, test_size=0.2):
    # Calculate the number of test samples
    num_samples = X.size(0)
    num_test_samples = int(num_samples * test_size)
    
    # Shuffle the data
    indices = torch.randperm(num_samples)
    
    # Split the data
    test_indices = indices[:num_test_samples]
    train_indices = indices[num_test_samples:]
    
    X_train, y_train = X[train_indices], y[train_indices]
    X_test, y_test = X[test_indices], y[test_indices]
    
    return X_train, X_test, y_train, y_test

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split_tensor(X, y, test_size=0.2)

# Create DataLoaders
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Define the LSTM model
class LSTM_Model(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(LSTM_Model, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, self.hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(self.hidden_size, output_size)
    
    def forward(self, x):
        h_0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)
        c_0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h_0, c_0))
        out = self.fc(out[:, -1, :])
        return out

input_size = 2  # Number of features
hidden_size = 50
output_size = 1  # Predicting a single value

model = LSTM_Model(input_size, hidden_size, output_size)
'''
# Train the model
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


num_epochs = 500
for epoch in range(num_epochs):
    model.train()
    for batch_x, batch_y in train_dataloader:
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y.unsqueeze(1))  # Reshape batch_y to match output shape
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Evaluate the model on the test set
model.eval()
test_loss = 0
with torch.no_grad():
    for batch_x, batch_y in test_dataloader:
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y.unsqueeze(1))
        test_loss += loss.item()

test_loss /= len(test_dataloader)
print(f'Test Loss: {test_loss:.4f}')

# Save the model
torch.save(model.state_dict(), '/Users/SverreB/Github_Repo/sverrbey_project/model/save/LSTM_500.pth')

'''

# Load the saved model state dict
model.load_state_dict(torch.load('/Users/SverreB/Github_Repo/sverrbey_project/model/save/LSTM_500.pth'))

# Set the model to evaluation mode
model.eval()

def plot_predictions(model, dataloader, scaler, seq_length):
    model.eval()
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            outputs = model(batch_x)
            predictions.append(outputs.cpu().numpy())
            actuals.append(batch_y.cpu().numpy())
    
    # Flatten the lists
    predictions = np.concatenate(predictions).flatten()
    actuals = np.concatenate(actuals).flatten()
    
    # Inverse transform the predictions and actuals to original scale
    # Create a dummy array with the same shape as the original data
    dummy_array = np.zeros((len(predictions), 3))
    dummy_array[:, 2] = predictions  # Fill the third column with predictions
    predictions = scaler.inverse_transform(dummy_array)[:, 2]  # Inverse transform and extract the third column
    
    dummy_array[:, 2] = actuals  # Fill the third column with actuals
    actuals = scaler.inverse_transform(dummy_array)[:, 2]  # Inverse transform and extract the third column
    
    # Plot the predictions vs actuals
    plt.figure(figsize=(10, 6))
    plt.plot(actuals, label='Actual')
    plt.plot(predictions, label='Predicted')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.title('LSTM Model Predictions vs Actuals')
    plt.legend()
    plt.show()

# Example usage
plot_predictions(model, test_dataloader, scaler, seq_length)

# Save the model
torch.save(model.state_dict(), '/Users/SverreB/Github_Repo/sverrbey_project/model/save/LSTM.pth')