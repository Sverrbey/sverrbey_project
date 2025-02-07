import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt


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


def Main():
    """ Main function that set up, execute, and store results """
    
    ## ptq.txt 
    # Precipitation, Temperature, Runoff (10 years: 1981 01 01 - 1991 12 31)
    ptq_path = '/Users/SverreB/Github_Repo/sverrbey_project/src/data/HBV-land/Data/ptq.txt'

    # split data into training and test sets
    train_dataloader, test_dataloader, scaler = preprocessing_data(ptq_path)

    input_size = 2  # Number of features
    hidden_size = 10
    output_size = 1  # Predicting a single value

    model = LSTM_Model(input_size, hidden_size, output_size)

    # Train the model
    #model = train_model(model, train_dataloader, test_dataloader, '500_10')

    # Load the saved model state dict
    model.load_state_dict(torch.load('/Users/SverreB/Github_Repo/sverrbey_project/model/save/LSTM_500_10.pth'))

    # Set the model to evaluation mode
    model.eval()
    plot_predictions(model, test_dataloader, scaler)


def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:i+seq_length, :2]  # Use the first two columns as features
        y = data[i+seq_length, 2]     # Use the third column as the target
        xs.append(x)
        ys.append(y)
    return torch.stack(xs), torch.stack(ys)


def train_test_split_tensor(X, y, test_size=0.2):
    """Function to split data into training and test sets"""
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


def preprocessing_data(ptq_path):
    """Preprocessing the data and creating sequences"""

    ## ptq.txt 
    # Precipitation, Temperature, Runoff (10 years: 1981 01 01 - 1991 12 31)
    df = pd.read_csv(ptq_path, header=1, delimiter='\t')
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

    # Preprocess the data
    scaler = MinMaxScaler()
    data_0_scaled = scaler.fit_transform(data_0)
    # Convert to PyTorch tensors
    data_0_tensor = torch.tensor(data_0_scaled, dtype=torch.float32)

    # Create sequences
    # The sequence length determines the number of rows used to predict the next row
    seq_length = 10
    X, y = create_sequences(data_0_tensor, seq_length)

  
    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split_tensor(X, y, test_size=0.2)

    # Create DataLoaders
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    return train_dataloader, test_dataloader, scaler


def train_model(model,train_dataloader, test_dataloader, name):
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
    torch.save(model.state_dict(), '/Users/SverreB/Github_Repo/sverrbey_project/model/save/LSTM_' + name +'.pth')

    return model


def plot_predictions(model, dataloader, scaler):
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

Main()