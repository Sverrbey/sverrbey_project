"""
author: @Sverrbey

This file contains only the necessary code to run the application: Intelligent Streamflow Monitoring 
system. It is written in a general way so that it can be used for any reservoir one chooses to monitor.
Underneath instructions in the form of comments are provided to guide the user on how to use the code.

In order to use the application the following data is needed:
    - Precipitation data: (real-time, historical)
    - Temperature data (real-time, historical)
    - Altitude data (DEM)

**    
If multiple temporal scales are to be used, the code should include the steps where certain scales 
are chosen.
    - I think that if the script also includes a visualization that can be interpreted by the operator
    this specification would relate to what is shown by the program.

?- If the data includes some time series data that is updated with regular frequency this updating interval
needs to be specified in the code.
"""
# Import necessary models & libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

from model.HBV import HBV_Model
from model.LSTM import LSTM_Model, MaskedMSELoss
from model.CNN_LSTM import CNN_LSTM_Model
from src.spatial_data_distribution import reformat_data
import os
from src.topology import generate_contour_map

import rasterio
from rasterio.enums import Resampling

def main ():
    
    """ Parameters for the LSTM model """
    # Define the extent of the area and the spacing between grid points
    #x_limit     = (  55000,  100000) 
    #y_limit     = (6877000, 6905000) 
    x_limit     = (65387.0, 107637.0) # [UTM-33n]
    y_limit     = (6873697.0, 6902697.0)# [UTM-33n]

    extent      = (x_limit[0], x_limit[1], y_limit[0], y_limit[1])  # Left, right, bottom, top
    spacing     = 1000  # Spacing between grid points (e.g., 100 m)
    number_days = 5*365

    perc_points = { '1': (57485, 6893688),
                    '2': (94493, 6894359),
                    '3': (83571, 6889543),
                    '4': (65666, 6891201),
                    '11': (72542,6898457),
                    '12': (86026,6897547),
                    '13': (88077,6884381),
                    '14': (72435,6898436),
                    '10': (74031,6885089)}
    
    disch_points = {    '6': (94683,6894729),
                        '5': (74334,6893688)}
    
    """
    df = pd.read_csv('src/data/Stryn/DailyPrec_Stryn.txt', skiprows=7, encoding='latin1', delimiter='\t')
    df = df.drop([0, 1]).reset_index(drop=True)
    df_discharge = pd.read_csv('src/data/Stryn/DailyDisch_Stryn.txt', skiprows=3, encoding='latin1', delimiter='\t')
    df_discharge = df_discharge.drop([0, 1, 2, 3, 4, 5]).reset_index(drop=True)
    df_temp = pd.read_csv('src/data/Stryn/DailyTemp_Stryn.txt', skiprows=7, encoding='latin1', delimiter='\t')
    df_temp = df.drop([0, 1]).reset_index(drop=True)
    
    perc_values = { '1': df[['1']][0:number_days].to_numpy().flatten(),
                    '2': df[['2']][0:number_days].to_numpy().flatten(),
                    '3': df[['3']][0:number_days].to_numpy().flatten(),
                    '4': df[['4']][0:number_days].to_numpy().flatten(),
                    '11': df[['11']][0:number_days].to_numpy().flatten(),
                    '12': df[['12']][0:number_days].to_numpy().flatten(),
                    '13': df[['13']][0:number_days].to_numpy().flatten(),
                    '14': df[['14']][0:number_days].to_numpy().flatten(),
                    '10': df[['10']][0:number_days].to_numpy().flatten()}

    disch_values = {'6': df_discharge[['6']][0:number_days].to_numpy(dtype=float).flatten(),
                    '5': df_discharge[['5']][0:number_days].to_numpy(dtype=float).flatten()}
    

    temp_values = { '1': df_temp[['1']][0:number_days].to_numpy().flatten(),
                    '2': df_temp[['2']][0:number_days].to_numpy().flatten(),
                    '3': df_temp[['3']][0:number_days].to_numpy().flatten(),
                    '4': df_temp[['4']][0:number_days].to_numpy().flatten(),
                    '11': df_temp[['11']][0:number_days].to_numpy().flatten(),
                    '12': df_temp[['12']][0:number_days].to_numpy().flatten(),
                    '13': df_temp[['13']][0:number_days].to_numpy().flatten(),
                    '14': df_temp[['14']][0:number_days].to_numpy().flatten(),
                    '10': df_temp[['10']][0:number_days].to_numpy().flatten()}

    dates = df['Point ID'][0:number_days].to_numpy().flatten()
    
    perc_file_path = 'src/data/interpolated_spatial_data/perc_spatial.json'
    disch_file_path = 'src/data/interpolated_spatial_data/discharge.json'
    temp_file_path = 'src/data/interpolated_spatial_data/temp.json'
    elevation_tiff_path = 'src/data/Stryn/Elevation.tif'

    # Check if the files exist
    if not os.path.exists(perc_file_path) or not os.path.exists(disch_file_path)or not os.path.exists(temp_file_path):
        # Generate the grid using IDW
        reformat_data(perc_points, perc_values, temp_values, disch_points, disch_values, number_days, dates, perc_file_path, disch_file_path,temp_file_path, extent=extent, spacing=spacing, power=2)
    """

    perc_file_path = 'data/interpolated_spatial_data/perc_spatial.json'
    disch_file_path = 'data/interpolated_spatial_data/discharge.json'
    temp_file_path = 'data/interpolated_spatial_data/temp.json'
    elevation_tiff_path = 'data/Stryn/Elevation.tif'

    df_perc = pd.read_json(perc_file_path, orient='values')
    df_disch = pd.read_json(disch_file_path, orient='values')
    df_temp = pd.read_json(temp_file_path, orient='values')

    df_perc.columns = ["Date", "interpolated_perc"]
    df_disch.columns = ["Date", "discharge"]
    df_temp.columns = ["Date", "interpolated_temp"]
    
    """ Plotting the interpolated percipitation pattern """
    grid_list = df_perc['interpolated_perc'].to_list()
    day = 8 
    plot_perc(grid_list, day, perc_points, disch_points, extent)
    


    # Assuming x_data and y_data are your input and target data
    x_data = np.array(df_perc['interpolated_perc'].tolist())
    temp_data = np.array(df_temp['interpolated_temp'].tolist())
    y_data = np.array(df_disch['discharge'].tolist())
    x_data_shape = x_data.shape[1:3]   
    elev_data = reformat_elevation_map(elevation_tiff_path, x_data_shape)

    """
    x_data = x_data.reshape(x_data.shape[0], 46*29)
    # Preprocess the data
    seq_length = 1
    train_split = 0.8

    train_dataloader, test_dataloader, scaler  = preprocess_data(x_data, y_data, seq_length, train_split=train_split)

    # Initialize the model, criterion, and optimizer
    input_size = train_dataloader.dataset.tensors[0].shape[2]  # Number of features
    hidden_size = 200  # Increase the number of hidden units
    output_size = y_data.shape[1]  # Number of target features
    num_layers = 4  # Increase the number of LSTM layers
    dropout = 0.3  # Adjust the dropout rate
    model = LSTM_Model(input_size=input_size, hidden_size=hidden_size, output_size=output_size, num_layers=num_layers, dropout=dropout)
    """

    batch_size = 8*2
    seq_length = 5 # days
    input_channels = 2  # Number of input channels (precipitation, temperature, elevation)
    hidden_size = 128
    output_size = y_data.shape[1]  # Number of target features

    model = CNN_LSTM_Model(input_channels, hidden_size, output_size, num_layers=2, dropout=0.3)
    
    train_dataloader, test_dataloader, scaler_x, scaler_y  = preprocess_data(x_data, temp_data, y_data, seq_length, batch_size)

    # Train the model
    #model = train_model(model, train_dataloader, test_dataloader, 'CNN_LSTM_1', seq_length=seq_length)
    
    # Load the saved model state dict
    model.load_state_dict(torch.load('/Users/SverreB/Github_Repo/sverrbey_project/model/save/CNN_LSTM_1.pth'))

    # Plot predictions vs actuals
    plot_predictions_vs_actuals(model, test_dataloader, scaler_y)
    

 
def preprocess_data(x_data, temp_data, y_data, seq_length, batch_size, train_split=0.8):
    """
    Preprocess the data to include additional channels for temperature and elevation.

    Args:
        x_data: Precipitation data (shape: [num_samples, height, width]).
        temp_data: Temperature data (shape: [num_samples, height, width]).
        elev_data: Elevation data (shape: [height, width]).
        y_data: Target data (shape: [num_samples, num_targets]).
        seq_length: Sequence length for LSTM.
        batch_size: Batch size for DataLoader.
        train_split: Fraction of data to use for training.

    Returns:
        train_dataloader, test_dataloader, scaler_x, scaler_y
    """
    # Reshape x_data and temp_data to 2D for normalization
    num_samples, height, width = x_data.shape
    x_data_reshaped = x_data.reshape(num_samples, -1)  # Shape: (num_samples, height * width)
    temp_data_reshaped = temp_data.reshape(num_samples, -1)

    # Stack the features along the last axis
    combined_data = np.stack([x_data_reshaped, temp_data_reshaped], axis=-1)  # Shape: (num_samples, height * width, 2)
    combined_data = combined_data.reshape(num_samples, -1)  # Flatten spatial dimensions for normalization

    # Initialize scalers
    scaler_x = MinMaxScaler()
    # Fit the scaler only on valid rows
    scaler_y = MinMaxScaler()

    # Mask rows where any feature in y_data is equal to -99.0
    valid_indices = ~np.any(y_data == -99.0, axis=1)  # Keep rows where no feature is -99.0
    
    # Fit scalers on the training portion of the data
    split_index = int(train_split * num_samples)
    scaler_x.fit(combined_data[:split_index])
    scaler_y.fit(y_data[valid_indices][:split_index])  # Fit only on valid rows
    
    # Transform the data
    combined_data_normalized = scaler_x.transform(combined_data)  # Shape: (num_samples, height * width * 3)
    
    y_data_normalized = scaler_y.transform(y_data)               # Shape: (num_samples, num_targets)
    
    # Reshape combined_data back to 3D (spatial dimensions restored)
    combined_data_normalized = combined_data_normalized.reshape(num_samples, 2, height, width)  # 2 channels: precipitation, temperature

    # Create sequences for x_data and y_data
    x_sequences, y_sequences = [], []
    for i in range(len(combined_data_normalized) - seq_length + 1):
        x_seq = combined_data_normalized[i:i+seq_length]  # Sequence of length `seq_length`
        y_seq = y_data_normalized[i+seq_length-1]         # Target corresponds to the last time step
        x_sequences.append(x_seq)
        y_sequences.append(y_seq)

    # Convert the lists to NumPy arrays first
    x_sequences = np.array(x_sequences)  # Shape: (num_sequences, seq_length, 2, height, width)
    y_sequences = np.array(y_sequences)  # Shape: (num_sequences, num_targets)

    # Convert to PyTorch tensors
    x_sequences = torch.tensor(x_sequences, dtype=torch.float32)
    y_sequences = torch.tensor(y_sequences, dtype=torch.float32)

    # Split into training and test sets
    split_index = int(train_split * len(x_sequences))
    x_train, x_test = x_sequences[:split_index], x_sequences[split_index:]
    y_train, y_test = y_sequences[:split_index], y_sequences[split_index:]

    # Create DataLoaders
    train_dataset = TensorDataset(x_train, y_train)
    test_dataset = TensorDataset(x_test, y_test)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_dataloader, test_dataloader, scaler_x, scaler_y

def plot_predictions_vs_actuals(model, dataloader, scaler_y):
    model.eval()
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            outputs = model(batch_x)
            predictions.append(outputs.cpu().numpy())
            
            actuals.append(batch_y.cpu().numpy())
    
    # Concatenate predictions and actuals along the first axis
    predictions = np.concatenate(predictions, axis=0)  # Shape: (num_samples, output_size)
    actuals = np.concatenate(actuals, axis=0)          # Shape: (num_samples, output_size)
    
    # Ensure predictions have the same shape as the scaler's expected input
    if predictions.shape[1] != scaler_y.min_.shape[0]:
        raise ValueError(f"Predictions shape {predictions.shape} does not match scaler's expected shape {scaler_y.min_.shape}")

    # Inverse transform the predictions and actuals to the original scale
    predictions = scaler_y.inverse_transform(predictions)
    actuals = scaler_y.inverse_transform(actuals)
    
    # Plot the predictions vs actuals
    plt.figure(figsize=(10, 6))
    plt.plot(actuals[:, 0], label='Actual Data (Feature 1)')
    plt.plot(predictions[:, 0], label='Predicted Data (Feature 1)', linestyle='--')
    if actuals.shape[1] > 1:  # If there are multiple target features
        plt.plot(actuals[:, 1], label='Actual Data (Feature 2)')
        plt.plot(predictions[:, 1], label='Predicted Data (Feature 2)', linestyle='--')
    plt.xlabel('Time Step')
    plt.ylabel('Value')
    plt.title('Model Predictions vs Actual Data')
    plt.legend()
    plt.show()

def create_sequences(input_data, target_data, seq_length):
    xs, ys = [], []
    for i in range(len(input_data) - seq_length):
        x = input_data[i:i+seq_length]
        y = target_data[i+seq_length]
        ys.append(y)
        xs.append(x)
 
    return torch.stack(xs), torch.stack(ys)

def preprocess_data_v1(x_data, y_data, seq_length, train_split=0.8):
    # Generate sin(x) feature
    sin_x_data = np.sin(x_data)

    # Combine x, y, and sin(x) into a single input tensor
    combined_data = np.concatenate((x_data, y_data, sin_x_data), axis=1)

    # Initialize scalers
    scaler = MinMaxScaler()

    # Fit scaler on the training data
    split_index = int(train_split * len(combined_data))
    scaler.fit(combined_data[:split_index])

    # Transform the data
    combined_data = scaler.transform(combined_data)

    # Convert to PyTorch tensors
    combined_data_tensor = torch.tensor(combined_data, dtype=torch.float32)
    y_data_tensor = torch.tensor(y_data, dtype=torch.float32)

    X, y = create_sequences(combined_data_tensor, y_data_tensor, seq_length)
    
    # Ensure the sequences have matching lengths
    min_length = min(int(len(X)), len(y))
    X, y = X[:min_length], y[:min_length]

    # Creating the training and test sets
    split_index = int(train_split * len(X))
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]
    
    # Ensure the input tensor has the correct shape [batch_size, sequence_length, input_size]
    X_train_tensor = X_train.view(-1, seq_length, X_train.shape[-1])
    y_train_tensor = y_train.view(-1, y_train.shape[-1])
    X_test_tensor = X_test.view(-1, seq_length, X_test.shape[-1])
    y_test_tensor = y_test.view(-1, y_test.shape[-1])
    
    # Create DataLoaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    return train_dataloader, test_dataloader, scaler
    
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

def train_model(model, train_dataloader, test_dataloader, name, seq_length=10, num_epochs=100, learning_rate=0.001):
    """
    Train the CNN_LSTM model.

    Args:
        model: The CNN_LSTM model to train.
        train_dataloader: DataLoader for training data.
        test_dataloader: DataLoader for test data.
        name: Name to save the trained model.
        seq_length: Sequence length of the input data.
        num_epochs: Number of epochs to train the model.
        learning_rate: Learning rate for the optimizer.

    Returns:
        model: The trained model.
    """
    # Define the loss function and optimizer
    criterion = MaskedMSELoss(ignore_value=-0.7523467)  # Mean Squared Error Loss
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

    # Training loop
    best_loss = float('inf')
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        for batch_x, batch_y in train_dataloader:
            # Forward pass
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_dataloader)

        # Validation loop
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in test_dataloader:
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                test_loss += loss.item()

        test_loss /= len(test_dataloader)

        # Print epoch results
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")

        # Learning rate scheduling
        scheduler.step(test_loss)

        # Save the best model
        if test_loss < best_loss:
            best_loss = test_loss
            torch.save(model.state_dict(), f'/Users/SverreB/Github_Repo/sverrbey_project/model/save/{name}.pth')
            print(f"Model saved with Test Loss: {test_loss:.4f}")

    return model

def train_model_v1(model, train_dataloader, test_dataloader, name, seq_length=2):
    criterion = MaskedMSELoss(ignore_value=-99)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

    # Training loop with early stopping and learning rate scheduling
    num_epochs = 100

    for epoch in range(num_epochs):
        model.train()
        for batch_x, batch_y in train_dataloader:
            # Ensure batch_x has the correct shape [batch_size, sequence_length, input_size]
            batch_x = batch_x.view(batch_x.size(0), seq_length, -1)
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)  # Ensure the target shape matches the model's output

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Evaluate on the test set
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for batch_x, batch_y in test_dataloader:
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                test_loss += loss.item()

        test_loss /= len(test_dataloader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Test Loss: {test_loss:.4f}')

        # Learning rate scheduling
        scheduler.step(test_loss)

    # Save the model
    torch.save(model.state_dict(), '/Users/SverreB/Github_Repo/sverrbey_project/model/save/' + name +'.pth')

    return model

def plot_perc(grid_list, day, perc_points, disch_points, extent):
    # Plot the interpolated grid for the first time step
    plt.rcParams["figure.figsize"] = (20, 7)
    grid_plot = grid_list[day-1]
    plt.imshow(grid_plot, extent=(extent[0], extent[1], extent[2], extent[3]), origin='lower', cmap='viridis')
    plt.colorbar(label='Interpolated Value')
    plt.scatter(*zip(*perc_points.values()), color='red', label='Known Data Points')
    plt.scatter(*zip(*disch_points.values()), color='blue', label='Discharge Stations')
    # Annotate each scatter point with its corresponding number
    for key, (x, y) in perc_points.items():
        plt.text(x, y, key, fontsize=12, ha='right', color='white')
    plt.legend(loc='upper right')
    plt.title('2D Interpolated Data Representation using IDW : Day ' + str(day))
    plt.grid()
    plt.show()

def reformat_elevation_map(tif_path, target_shape):
    """
    Reformat a .tif elevation map to match the target shape.

    Args:
        tif_path (str): Path to the .tif elevation map.
        target_shape (tuple): Desired shape (height, width) to match x_data.

    Returns:
        np.ndarray: Resampled elevation map with the target shape.
    """
    # Open the .tif file
    with rasterio.open(tif_path) as src:
        # Read the elevation data
        elevation_data = src.read(1)  # Read the first band (assumes single-band elevation map)

        # Get the original shape of the elevation map
        original_shape = elevation_data.shape

        # Calculate the resampling scale factors
        scale_y = target_shape[0] / original_shape[0]
        scale_x = target_shape[1] / original_shape[1]

        # Resample the elevation data to the target shape
        elevation_resampled = src.read(
            1,
            out_shape=(int(target_shape[0]), int(target_shape[1])),
            resampling=Resampling.bilinear  # Use bilinear interpolation for resampling
        )

        # Ensure the output is a 2D NumPy array
        elevation_resampled = np.array(elevation_resampled)

    return elevation_resampled


if __name__ == "__main__":
    main()