import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt


class TransformerModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2, dropout=0.2, nhead=8):
        super(TransformerModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = nn.Linear(input_size, hidden_size)
        self.positional_encoding = nn.Parameter(torch.zeros(1, 1000, hidden_size))  # Assuming max sequence length of 1000
        self.transformer = nn.Transformer(hidden_size, nhead=nhead, num_encoder_layers=num_layers, num_decoder_layers=num_layers, dropout=dropout, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # Add positional encoding
        seq_length = x.size(1)
        x = self.embedding(x) + self.positional_encoding[:, :seq_length, :]
        
        # Transformer expects input of shape (seq_length, batch_size, hidden_size)
        x = x.permute(1, 0, 2)
        
        # Transformer forward pass
        transformer_output = self.transformer(x, x)
        
        # Take the output of the last time step
        out = transformer_output[-1, :, :]
        
        # Fully connected layers with ReLU activation
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        
        return out

def Main():
    ## ptq.txt 
    # Precipitation, Temperature, Runoff (10 years: 1981 01 01 - 1991 12 31)
    ptq_path = '/Users/SverreB/Github_Repo/sverrbey_project/src/data/HBV-land/Data/ptq.txt'

    # split data into training and test sets
    train_dataloader, test_dataloader, scaler = preprocessing_data(ptq_path)


    # Example usage
    input_size = 2  # Number of features
    hidden_size = 64
    output_size = 1  # Predicting a single value
    num_layers = 2  # Number of Transformer layers
    dropout = 0.2  # Dropout rate
    nhead = 8  # Number of attention heads

    model = TransformerModel(input_size, hidden_size, output_size, num_layers, dropout, nhead)

    # Train the model
    #model = train_model(model, train_dataloader, test_dataloader, 'attempt_1')
    
    # Load the saved model state dict
    model.load_state_dict(torch.load('/Users/SverreB/Github_Repo/sverrbey_project/model/save/Transformer_attempt_1.pth'))

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
    # Define the loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 100
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
    torch.save(model.state_dict(), '/Users/SverreB/Github_Repo/sverrbey_project/model/save/Transformer_' + name +'.pth')

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
    xs = np.linspace(0, len(actuals), len(actuals))
    plt.figure(figsize=(10, 6))
    plt.step(xs, actuals, label='Actual')
    plt.step(xs, predictions, label='Predicted')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.title('Transformer Model Predictions vs Actuals')
    plt.legend()
    plt.show()

Main()