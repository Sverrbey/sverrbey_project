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
from model.LSTM import LSTM
from model.CNN_LSTM import CNN_LSTM_Model
from model.CNN import CNN
from src.spatial_data_distribution import reformat_data
import os
from src.topology import generate_contour_map

import rasterio
from rasterio.enums import Resampling

def main ():
    
    """ Parameters for the LSTM model """
    # Define the extent of the area and the spacing between grid points
    x_limit     = (  55000,  100000) 
    y_limit     = (6877000, 6905000) 
    #x_limit     = (65387.0, 107637.0) # [UTM-33n]

    #y_limit     = (6873697.0, 6902697.0)# [UTM-33n]

    extent      = (x_limit[0], x_limit[1], y_limit[0], y_limit[1])  # Left, right, bottom, top
    spacing     = 4000  # Spacing between grid points (e.g., 100 m)
    number_days = 5*365

    points = {  '1': (57485, 6893688),
                '2': (94493, 6894359),
                '3': (83571, 6889543),
                '4': (65666, 6891201),
                '5': (74334, 6893688),
                '6': (94683, 6894729),
                '9': (87112, 6889962),
                '10':(74031, 6885089),
                '11':(72542, 6898457),
                '12':(86026, 6897547),
                '13':(88077, 6884381),
                '14':(72435, 6898436),
                '15':(78934, 6900441),
                '16':(89560, 6897378),
                '17':(77580, 6889281),
                '18':(88119, 6883302),
                '20':(84796, 6898457),
                '21':(67778, 6894393)
                }
    
    disch_points = {    '6': (94683,6894729),
                        '5': (74334,6893688)}
    
    
    df_perc = pd.read_csv('data/Stryn/DailyPrec_Stryn.txt', skiprows=7, encoding='latin1', delimiter='\t')
    df_perc = df_perc.drop([0, 1]).reset_index(drop=True)
   
    df_temp = pd.read_csv('data/Stryn/DailyTemp_Stryn.txt', skiprows=7, encoding='latin1', delimiter='\t')
    df_temp = df_temp.drop([0, 1]).reset_index(drop=True)
    
    df_evap = pd.read_csv('data/Stryn/DailyEvap_Stryn.txt', skiprows=7, encoding='latin1', delimiter='\t')
    df_evap = df_evap.drop([0, 1]).reset_index(drop=True)

    df_discharge = pd.read_csv('data/Stryn/DailyDisch_Stryn.txt', skiprows=3, encoding='latin1', delimiter='\t')
    df_discharge = df_discharge.drop([0, 1, 2, 3, 4, 5]).reset_index(drop=True)

    perc_values = { '1': df_perc[['1']][0:number_days].to_numpy().flatten(),
                    '2': df_perc[['2']][0:number_days].to_numpy().flatten(),
                    '3': df_perc[['3']][0:number_days].to_numpy().flatten(),
                    '4': df_perc[['4']][0:number_days].to_numpy().flatten(),
                    '10': df_perc[['10']][0:number_days].to_numpy().flatten(),
                    '11': df_perc[['11']][0:number_days].to_numpy().flatten(),
                    '12': df_perc[['12']][0:number_days].to_numpy().flatten(),
                    '13': df_perc[['13']][0:number_days].to_numpy().flatten(),
                    '14': df_perc[['14']][0:number_days].to_numpy().flatten()}
   
    temp_values = { '1': df_temp[['1']][0:number_days].to_numpy().flatten(),
                    '2': df_temp[['2']][0:number_days].to_numpy().flatten(),
                    '3': df_temp[['3']][0:number_days].to_numpy().flatten(),
                    '5': df_temp[['5']][0:number_days].to_numpy().flatten(),
                    '6': df_temp[['6']][0:number_days].to_numpy().flatten(),
                    '10': df_temp[['10']][0:number_days].to_numpy().flatten(),
                    '15': df_temp[['15']][0:number_days].to_numpy().flatten(),
                    '16': df_temp[['16']][0:number_days].to_numpy().flatten(),
                    '17': df_temp[['17']][0:number_days].to_numpy().flatten()}

    evap_values = { '1': df_evap[['1']][0:number_days].to_numpy().flatten(),
                    '2': df_evap[['2']][0:number_days].to_numpy().flatten(),
                    '5': df_evap[['5']][0:number_days].to_numpy().flatten(),
                    '9': df_evap[['9']][0:number_days].to_numpy().flatten(),
                    '10': df_evap[['10']][0:number_days].to_numpy().flatten(),
                    '18': df_evap[['18']][0:number_days].to_numpy().flatten(),
                    '20': df_evap[['20']][0:number_days].to_numpy().flatten(),
                    '21': df_evap[['21']][0:number_days].to_numpy().flatten()}

    disch_values = {'6': df_discharge[['6']][0:number_days].to_numpy(dtype=float).flatten(),
                    '5': df_discharge[['5']][0:number_days].to_numpy(dtype=float).flatten()}
    

    dates = df_perc['Point ID'][0:number_days].to_numpy().flatten()
    
    # X data
    perc_file_path = 'data/interpolated_spatial_data/perc_spatial.json'
    temp_file_path = 'data/interpolated_spatial_data/temp.json'
    evap_file_path = 'data/interpolated_spatial_data/evap.json'

    # Y data
    disch_file_path = 'data/interpolated_spatial_data/discharge.json'

    # Parameter data
    elevation_tiff_path = 'data/Stryn/Elevation.tif'

    # Check if the files exist
    if not os.path.exists(perc_file_path) or not os.path.exists(disch_file_path) or not os.path.exists(temp_file_path) or not os.path.exists(evap_file_path):
        # Generate the grid using IDW
        reformat_data(points, perc_values, temp_values, evap_values, disch_points, disch_values, number_days, dates, perc_file_path, disch_file_path,temp_file_path, evap_file_path, extent=extent, spacing=spacing, power=2)
    

    df_perc     = pd.read_json(perc_file_path, orient='values')
    df_disch    = pd.read_json(disch_file_path, orient='values')
    df_temp     = pd.read_json(temp_file_path, orient='values')
    df_evap     = pd.read_json(evap_file_path, orient='values')

    df_perc.columns     = ["Date", "interpolated_perc"]
    df_disch.columns    = ["Date", "discharge"]
    df_temp.columns     = ["Date", "interpolated_temp"]
    df_evap.columns     = ["Date", "interpolated_evap"]
    
    """ Plotting the interpolated percipitation pattern  
    grid_perc = df_perc['interpolated_perc'].to_list()
    grid_temp = df_temp['interpolated_temp'].to_list()
    grid_evap = df_evap['interpolated_evap'].to_list()

    day = 8 
    plot_interpolation(grid_perc, day, points, disch_points, extent, 'Precipitation')
    plot_interpolation(grid_temp, day, points, disch_points, extent, 'Temperature')
    plot_interpolation(grid_evap, day, points, disch_points, extent, 'Evaporation')
    """
    # Assuming x_data and y_data are your input and target data
    perc_data   = np.array(df_perc['interpolated_perc'].tolist())
    temp_data   = np.array(df_temp['interpolated_temp'].tolist())
    evap_data   = np.array(df_evap['interpolated_evap'].tolist())
    y_data      = np.array(df_disch['discharge'].tolist())
    
    # TO DO: Parse the data and extract the sections that have enough days to be used for training, based on the sequence numbers. 
    """
    I have data that is for example: data = [1,2,3,4,-99.0,6,7,8,-99.0,-99.0,-99.0,-99.0,12,13,14,15].

    And now I only have code that extracts the valid numbers so that I end up with: data = [1,2,3,4,6,7,8,12,13,14,15]

    And I want to parse the data and select the valid indices to use in my dataset based on the sequence length I've defined.
    So that I don't have sequences that span for more invalid data. I want each sequence to only include data from the same 
    time period. For example if my sequence length = 2 the data would be: data = [[1,2], [3,4], [6,7], [12,13], [14,15]]. 
    Here [8,12] wouldn't be valid because it spans over a time period with invalid data.
    
    """
   # Mask rows where any feature in y_data is equal to -99.0 and the sequence length isn't possible
    seq_length = 4
    valid_indices = filter_valid_indices(y_data, seq_length) 
    perc_data = perc_data[valid_indices]
    temp_data = temp_data[valid_indices]
    evap_data = evap_data[valid_indices]
    y_data = y_data[valid_indices]

    # Replace NaN values in y_data with 0
    #y_data = np.nan_to_num(y_data, nan=0.0001)
    
    # Extract the second column and keep it as 2D
    y_data = y_data[:, [1]]  # Shape: (num_samples, 1)   

    perc_data_shape = perc_data.shape[1:3]   
    elev_data = reformat_elevation_map(elevation_tiff_path, perc_data_shape)
    
    num_layers      = 4  # Increase the number of LSTM layers
    dropout         = 0.4  # Adjust the dropout rate
    batch_size      = 8
    seq_length      = 4 # days (# Because we're removing some days)
    input_channels  = 3  # Number of input channels (precipitation, temperature, evaporation)
    hidden_size     = 256
    output_size     = 1  # Number of target features
    input_size      = perc_data.shape[1:3]
    height          = input_size[0]
    width           = input_size[1]
    
    model_CNN_LSTM  = CNN_LSTM_Model(input_channels, hidden_size, output_size, input_size, num_layers=num_layers, dropout=dropout)
    model_CNN       = CNN(input_channels, output_size, hidden_size)
    model_LSTM      = LSTM(input_channels, height, width, hidden_size, output_size, num_layers, dropout)

    train_dataloader, val_dataloader, test_dataloader, scaler_x, scaler_y  = preprocess_data(perc_data, temp_data, evap_data, y_data, seq_length, batch_size)

    # Load the saved model state dict
    #model.load_state_dict(torch.load('/Users/SverreB/Github_Repo/sverrbey_project/model/save/CNN_LSTM_1.pth'))

    # Train the model
    model_CNN_LSTM  = train_model(model_CNN_LSTM, train_dataloader, val_dataloader, 'CNN_LSTM', scaler_y, scaler_x, seq_length=seq_length)
    model_CNN       = train_model(model_CNN, train_dataloader, val_dataloader, 'CNN', scaler_y, scaler_x, seq_length=seq_length)
    model_LSTM      = train_model(model_LSTM, train_dataloader, val_dataloader, 'LSTM', scaler_y, scaler_x, seq_length=seq_length)

    # Load the saved model state dict
    #model_CNN_LSTM.load_state_dict(torch.load('/Users/SverreB/Github_Repo/sverrbey_project/model/save/CNN_LSTM.pth'))
    #model_CNN.load_state_dict(torch.load('/Users/SverreB/Github_Repo/sverrbey_project/model/save/CNN.pth'))
    #model_LSTM.load_state_dict(torch.load('/Users/SverreB/Github_Repo/sverrbey_project/model/save/LSTM.pth'))

    # Plot predictions vs actuals
    plot_predictions_vs_actuals(model_CNN_LSTM, test_dataloader, scaler_y, 'CNN_LSTM')
    plot_predictions_vs_actuals(model_CNN, test_dataloader, scaler_y, 'CNN')
    plot_predictions_vs_actuals(model_LSTM, test_dataloader, scaler_y, 'LSTM')
    

def preprocess_data(perc_data, temp_data, evap_data, y_data, seq_length, batch_size,channels = 3, train_split=0.7, validation_split=0.15, test_split=0.15):
    """
    Preprocess the data to include additional channels for temperature and elevation.

    Args:
        perc_data: Precipitation data (shape: [num_samples, height, width]).
        temp_data: Temperature data (shape: [num_samples, height, width]).
        evap_data: Evaporation data (shape: [num_samples, height, width]).
        y_data: Target data, discharge (shape: [num_samples, num_targets]).
        seq_length: Sequence length for LSTM.
        batch_size: Batch size for DataLoader.
        train_split: Fraction of data to use for training.
        validation_split: Fraction of data to use for validation.
        test_split: Fraction of data to use for testing.

    Returns:
        train_dataloader, val_dataloader, test_dataloader, scaler_x, scaler_y
    """
    # Reshape perc_data and temp_data to 2D for normalization
    num_samples, height, width = perc_data.shape
    perc_data_reshaped = perc_data.reshape(num_samples, -1)  # Shape: (num_samples, height * width)
    temp_data_reshaped = temp_data.reshape(num_samples, -1)
    evap_data_reshaped = evap_data.reshape(num_samples, -1)

    # Stack the features along the last axis
    combined_data = np.stack([perc_data_reshaped, temp_data_reshaped, evap_data_reshaped], axis=-1)  # Shape: (num_samples, height * width, 3)
    combined_data = combined_data.reshape(num_samples, -1)  # Flatten spatial dimensions for normalization

    # Initialize scalers
    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()

    # Fit scalers on the training portion of the data
    train_end = int(train_split * num_samples)
    val_end = train_end + int(validation_split * num_samples)
    scaler_x.fit(combined_data[:train_end])
    scaler_y.fit(y_data[:train_end])  # Fit only on training data

    # Transform the data
    combined_data_normalized = scaler_x.transform(combined_data)  # Shape: (num_samples, height * width * 3)
    y_data_normalized = scaler_y.transform(y_data)               # Shape: (num_samples, num_targets)

    # Reshape combined_data back to 3D (spatial dimensions restored)
    combined_data_normalized = combined_data_normalized.reshape(num_samples, channels, height, width)  # 2 channels: precipitation, temperature

    # Create sequences for x_data and y_data
    x_sequences, y_sequences = [], []
    for i in range(len(combined_data_normalized) - seq_length + 1):
        x_seq = combined_data_normalized[i:i+seq_length]  # Sequence of length `seq_length`
        y_seq = y_data_normalized[i+seq_length-1]         # Target corresponds to the last time step
        x_sequences.append(x_seq)
        y_sequences.append(y_seq)

    # Convert the lists to NumPy arrays
    x_sequences = np.array(x_sequences)  # Shape: (num_sequences, seq_length, 2, height, width)
    y_sequences = np.array(y_sequences)  # Shape: (num_sequences, num_targets)

    # Split the data into train, validation, and test sets
    num_sequences = len(x_sequences)
    train_end = int(train_split * num_sequences)
    val_end = train_end + int(validation_split * num_sequences)

    x_train, x_val, x_test = x_sequences[:train_end], x_sequences[train_end:val_end], x_sequences[val_end:]
    y_train, y_val, y_test = y_sequences[:train_end], y_sequences[train_end:val_end], y_sequences[val_end:]

    # Convert to PyTorch tensors
    x_train = torch.tensor(x_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    x_val = torch.tensor(x_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.float32)
    x_test = torch.tensor(x_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    # Create DataLoaders
    train_dataset = TensorDataset(x_train, y_train)
    val_dataset = TensorDataset(x_val, y_val)
    test_dataset = TensorDataset(x_test, y_test)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_dataloader, val_dataloader, test_dataloader, scaler_x, scaler_y

def plot_predictions_vs_actuals(model, dataloader, scaler_y, name):
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

    plt.xlabel('Time Step [days]')
    plt.ylabel('Value')
    plt.title(name +': Model Predictions vs Actual Data')
    plt.legend()
    plt.show()
    
def train_model(model, train_dataloader, val_dataloader, name, scaler_y, scaler_x, seq_length=10, num_epochs=100, learning_rate=0.001):
    """
    Train the CNN_LSTM model.

    Args:
        model: The CNN_LSTM model to train.
        train_dataloader: DataLoader for training data.
        val_dataloader: DataLoader for validation data.
        name: Name to save the trained model.
        seq_length: Sequence length of the input data.
        num_epochs: Number of epochs to train the model.
        learning_rate: Learning rate for the optimizer.

    Returns:
        model: The trained model.
    """
    # Define the loss function and optimizer
    criterion = torch.nn.MSELoss()  # Mean Squared Error Loss
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Early stopping parameters
    patience = 10  # Number of epochs to wait for improvement
    best_val_loss = float('inf')
    early_stop_counter = 0

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_nse = 0.0

        for batch_x, batch_y in train_dataloader:
            # Forward pass
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            # Denormalize the predictions and true values
            y_true = scaler_y.inverse_transform(batch_y.detach().cpu().numpy())
            y_pred = scaler_y.inverse_transform(outputs.detach().cpu().numpy())

            # Calculate NSE in the original scale
            numerator = np.sum((y_true - y_pred) ** 2)
            denominator = np.sum((y_true - np.mean(y_true)) ** 2)
            batch_nse = 1 - (numerator / denominator if denominator != 0 else 0)
            train_nse += batch_nse

        train_loss /= len(train_dataloader)
        train_nse /= len(train_dataloader)

        # Validation loop
        model.eval()
        val_loss = 0.0
        val_nse = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_dataloader:
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()

                # Denormalize the predictions and true values
                y_true = scaler_y.inverse_transform(batch_y.detach().cpu().numpy())
                y_pred = scaler_y.inverse_transform(outputs.detach().cpu().numpy())

                # Calculate NSE in the original scale
                numerator = np.sum((y_true - y_pred) ** 2)
                denominator = np.sum((y_true - np.mean(y_true)) ** 2)
                batch_nse = 1 - (numerator / denominator if denominator != 0 else 0)
                val_nse += batch_nse

        val_loss /= len(val_dataloader)
        val_nse /= len(val_dataloader)

        # Print epoch results
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train NSE: {train_nse:.4f}, Val Loss: {val_loss:.4f}, Val NSE: {val_nse:.4f}")

        # Early stopping logic
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stop_counter = 0
            # Save the best model
            torch.save(model.state_dict(), '/Users/SverreB/Github_Repo/sverrbey_project/model/save/' + name + '_best.pth')
        else:
            early_stop_counter += 1
            print(f"Early stopping counter: {early_stop_counter}/{patience}")

        if early_stop_counter >= patience:
            print("Early stopping triggered. Stopping training.")
            break

    # Save the final model
    torch.save(model.state_dict(), '/Users/SverreB/Github_Repo/sverrbey_project/model/save/' + name + '.pth')
    return model

def plot_interpolation(grid_list, day, perc_points, disch_points, extent, title):

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
    plt.title(title + ': 2D Interpolated Data Representation using IDW : Day ' + str(day))
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

def filter_valid_indices(data, seq_length):
    """
    Filters out invalid data and creates sequences of a specified length.

    Args:
        data (list): Input data containing valid and invalid values.
        seq_length (int): Desired sequence length.

    Returns:
        set: set of valid indices.
    """
    result = np.full(len(data), False, dtype=bool)
    valid_set = set()
    for x in range(len(data) - seq_length):
        #Step 1: We create a sequence, and if the sequence contains -99.0 we skip it
        possible_seq = data[x:x+seq_length]
        if -99.0 in possible_seq:
            continue
        else:
            for i in range(x, x+seq_length):
                #Step 2: We add the valid indices to a set
                valid_set.add(i)
    for j in valid_set:
        result[j] = True
    

    return result

if __name__ == "__main__":
    main()