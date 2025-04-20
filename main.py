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
# Import necessary libraries & models
import torch
import os
import rasterio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing  import MinMaxScaler
from torch.utils.data       import DataLoader, TensorDataset
from model.HBV              import HBV_Model
from model.LSTM             import LSTM
from model.CNN_LSTM         import CNN_LSTM_Model
from model.CNN              import CNN
from src.spatial_data_distribution import reformat_data_stryn, reformat_data_gaula
from src.topology           import generate_contour_map
from rasterio.enums         import Resampling

from src.plotting           import plot_sensitivity, plot_predictions_vs_actuals, plot_interpolation

#train_model
#filter_valid_indices
#extract_info
#catchment_styrn, catchment_gaula

#preprocess_data 
#reformat_elevation_map

def main ():
    """
    Main function to run the application.
    It includes the following steps:
    1. Define the extent of the area and the spacing between grid points.
    2. Pick a catchment area (Stryn or Gaula).
    3. Check if the necessary files exist. If not, generate the grid using IDW.
    4. Read the interpolated data from JSON files.
    5. Define the model parameters and valid data.
    6. Preprocess the data to include additional channels for temperature and elevation.
    7. Train the model.
    8. Plot predictions vs actuals.
    
    TODO:
    - Fix the points
    The point ID don't overlap between values. Maybe its not that important.
    - Fix the limits
    
    """

   
    # Pick of catchment
    chosenCatchment = 'Stryn' # 'Stryn' or 'Gaula'
    if chosenCatchment == 'Stryn':
        # Area description 
        x_limit     = (  55000,  100000)
        x_degrees   = (6.45811333, 7.33417667) # (6°27'29.208", 7°20'3.036")
        y_limit     = (6877000, 6905000) 
        y_degrees   = (61.75932639, 62.06398778) #(61°45'33.575", 62°3'50.356")
        extent      = (x_limit[0], x_limit[1], y_limit[0], y_limit[1])  
        resolution  = 4000  # Spacing between grid points (e.g., 4000 m)
        spacing = 6*365 #  max 42 years (80-22)
        # Timeseries
        points, perc_values, temp_values, evap_values, disch_values, dates = catchment_styrn(number_days)

    elif chosenCatchment == 'Gaula':
        # Area description 
        x_limit     = (  55000,  100000)
        x_degrees   = (6.45811333, 7.33417667) # (6°27'29.208", 7°20'3.036")
        y_limit     = (6877000, 6905000) 
        y_degrees   = (61.75932639, 62.06398778) #(61°45'33.575", 62°3'50.356")
        extent      = (x_limit[0], x_limit[1], y_limit[0], y_limit[1])  
        spacing     = 4000  # Spacing between grid points (e.g., 4000 m)
        number_days = 6*365 # max 6 years (99-05)
        # Timeseries
        points_gaula, perc_values, temp_values, rad_values, hyd_values, relHum_values, wind_values, disch_values, dates = catchment_gaula(number_days)
    
    # X data
    perc_file_path  = 'data/interpolated_spatial_data/perc_spatial.json'
    temp_file_path  = 'data/interpolated_spatial_data/temp.json'
    evap_file_path  = 'data/interpolated_spatial_data/evap.json'
    rad_file_path   = 'data/interpolated_spatial_data/rad.json'   
    hyd_file_path   = 'data/interpolated_spatial_data/hyd.json'
    relHum_file_path = 'data/interpolated_spatial_data/relHum.json'
    wind_file_path  = 'data/interpolated_spatial_data/wind.json'

    # Y data
    disch_file_path = 'data/interpolated_spatial_data/discharge.json'

    # Catchment file paths
    stryn_file_paths = [perc_file_path, disch_file_path, temp_file_path, evap_file_path]
    gaula_file_paths = [perc_file_path, disch_file_path, temp_file_path, rad_file_path, hyd_file_path, relHum_file_path, wind_file_path]

    # Parameter data
    elevation_tiff_path = 'data/Stryn/Elevation.tif'

    # Check if the files exist or interpolate data
    if chosenCatchment == 'Stryn' and any(not os.path.exists(file) for file in stryn_file_paths):
        # Generate the grid using IDW
        reformat_data_stryn(points, perc_values, temp_values, evap_values, disch_values, number_days, dates, stryn_file_paths, extent=extent, spacing=spacing, power=2)
    elif chosenCatchment == 'Gaula' and any(not os.path.exists(file) for file in gaula_file_paths):
        # Generate the grid using IDW
        reformat_data_gaula(points_gaula, perc_values, temp_values, rad_values, hyd_values, relHum_values, wind_values, disch_values, number_days, dates, gaula_file_paths, extent=extent, spacing=spacing, power=2)
    
    # Reading the interpolated data
    if chosenCatchment == 'Stryn':
        df_perc     = pd.read_json(perc_file_path, orient='values')
        df_disch    = pd.read_json(disch_file_path, orient='values')
        df_temp     = pd.read_json(temp_file_path, orient='values')
        df_evap     = pd.read_json(evap_file_path, orient='values')

        df_perc.columns     = ["Date", "interpolated_perc"]
        df_disch.columns    = ["Date", "discharge"]
        df_temp.columns     = ["Date", "interpolated_temp"]
        df_evap.columns     = ["Date", "interpolated_evap"]
    elif chosenCatchment == 'Gaula':
        df_perc     = pd.read_json(perc_file_path, orient='values')
        df_disch    = pd.read_json(disch_file_path, orient='values')
        df_temp     = pd.read_json(temp_file_path, orient='values')
        df_rad      = pd.read_json(rad_file_path, orient='values')
        df_hyd      = pd.read_json(hyd_file_path, orient='values')
        df_relHum   = pd.read_json(relHum_file_path, orient='values')
        df_wind     = pd.read_json(wind_file_path, orient='values')

        df_perc.columns     = ["Date", "interpolated_perc"]
        df_disch.columns    = ["Date", "discharge"]
        df_temp.columns     = ["Date", "interpolated_temp"]
        df_rad.columns      = ["Date", "interpolated_rad"]
        df_hyd.columns      = ["Date", "interpolated_hyd"]
        df_relHum.columns   = ["Date", "interpolated_relHum"]
        df_wind.columns     = ["Date", "interpolated_wind"]
    
    # Defining the model parameters and valid data
    if chosenCatchment == 'Stryn':
        # Assuming x_data and y_data are your input and target data
        perc_data   = np.array(df_perc['interpolated_perc'].tolist())
        temp_data   = np.array(df_temp['interpolated_temp'].tolist())
        evap_data   = np.array(df_evap['interpolated_evap'].tolist())
        y_data      = np.array(df_disch['discharge'].tolist())

        # Mask rows where any feature in y_data is equal to -99.0 and the sequence length isn't possible
        seq_length = 7
        valid_indices = filter_valid_indices(y_data, seq_length) 
        perc_data = perc_data[valid_indices]
        temp_data = temp_data[valid_indices]
        evap_data = evap_data[valid_indices]
        y_data = y_data[valid_indices]

        # Replace NaN values in y_data with 0
        #y_data = np.nan_to_num(y_data, nan=0.0001)
        
        # Extract the second column and keep it as 2D
        y_data = y_data[:, [0,1]]  # Shape: (num_samples, 1)   

        perc_data_shape = perc_data.shape[1:3]   
        #elev_data = reformat_elevation_map(elevation_tiff_path, perc_data_shape)
        
        num_layers      = 4  # Increase the number of LSTM layers
        dropout         = 0.4  # Adjust the dropout rate
        batch_size      = 2**6
        seq_length      = seq_length # days (# Because we're removing some days)
        input_channels  = 3  # Number of input channels (precipitation, temperature, evaporation)
        hidden_size     = 256
        output_size     = 2  # Number of target features
        input_size      = perc_data.shape[1:3]
        height          = input_size[0]
        width           = input_size[1]


        """ Plotting the interpolated percipitation pattern  

        graph_patch_perc = 'data/graphs/perc'
        graph_patch_temp = 'data/graphs/temp'
        graph_patch_evap = 'data/graphs/evap'

        #for day in range(1, len(perc_data)+1):
            #plot_interpolation(perc_data, day, points, graph_patch_perc, extent, 'Precipitation')
            #plot_interpolation(temp_data, day, points, graph_patch_temp, extent, 'Temperature')
            #plot_interpolation(evap_data, day, points, extent,graph_patch_evap, 'Evaporation')
        """
    elif chosenCatchment == 'Gaula':
        # Assuming x_data and y_data are your input and target data
        perc_data   = np.array(df_perc['interpolated_perc'].tolist())
        temp_data   = np.array(df_temp['interpolated_temp'].tolist())
        rad_data    = np.array(df_rad['interpolated_rad'].tolist())
        hyd_data    = np.array(df_hyd['interpolated_hyd'].tolist())
        relHum_data = np.array(df_relHum['interpolated_relHum'].tolist())
        wind_data   = np.array(df_wind['interpolated_wind'].tolist())
        y_data      = np.array(df_disch['discharge'].tolist())
        # Mask rows where any feature in y_data is equal to -99.0 and the sequence length isn't possible
        seq_length = 7
        valid_indices = filter_valid_indices(y_data, seq_length) 
        perc_data = perc_data[valid_indices]
        temp_data = temp_data[valid_indices]
        rad_data = rad_data[valid_indices]
        hyd_data = hyd_data[valid_indices]
        relHum_data = relHum_data[valid_indices]
        wind_data = wind_data[valid_indices]
        y_data = y_data[valid_indices]

        # Replace NaN values in y_data with 0
        #y_data = np.nan_to_num(y_data, nan=0.0001)
        
        # Extract the second column and keep it as 2D
        y_data = y_data[:, [0,1,3,4]]  # Shape: (num_samples, 1)   
        perc_data_shape = perc_data.shape[1:3]   
        #elev_data = reformat_elevation_map(elevation_tiff_path, perc_data_shape)
        
        num_layers      = 4  # Increase the number of LSTM layers
        dropout         = 0.4  # Adjust the dropout rate
        batch_size      = 2**6 # 64
        seq_length      = seq_length # days (# Because we're removing some days)
        input_channels  = 6  # Number of input channels (precipitation, temperature, evaporation)
        hidden_size     = 2**8
        output_size     = 4  # Number of target features
        input_size      = perc_data.shape[1:3]
        height          = input_size[0]
        width           = input_size[1]

    #model_CNN_LSTM  = CNN_LSTM_Model(input_channels, height, width, hidden_size, output_size, num_layers=num_layers, dropout=dropout)
    #model_CNN       = CNN(input_channels, output_size, hidden_size)
    model_LSTM       = LSTM(input_channels, height, width, hidden_size, output_size, num_layers=num_layers, dropout = dropout)

    # Combining the data for training
    if chosenCatchment == 'Stryn':
        # Reshape perc_data and temp_data to 2D for normalization
        num_samples, height, width = perc_data.shape
        perc_data_reshaped = perc_data.reshape(num_samples, -1)  # Shape: (num_samples, height * width)
        temp_data_reshaped = temp_data.reshape(num_samples, -1)
        evap_data_reshaped = evap_data.reshape(num_samples, -1)

        # Stack the features along the last axis
        combined_data = np.stack([perc_data_reshaped, temp_data_reshaped, evap_data_reshaped], axis=-1)  # Shape: (num_samples, height * width, 3)
        combined_data = combined_data.reshape(num_samples, -1)  # Flatten spatial dimensions for normalization
    elif chosenCatchment == 'Gaula':
        # Reshape perc_data and temp_data to 2D for normalization
        num_samples, height, width = perc_data.shape
        perc_data_reshaped = perc_data.reshape(num_samples, -1)  # Shape: (num_samples, height * width)
        temp_data_reshaped = temp_data.reshape(num_samples, -1)
        rad_data_reshaped = rad_data.reshape(num_samples, -1)
        hyd_data_reshaped = hyd_data.reshape(num_samples, -1)
        relHum_data_reshaped = relHum_data.reshape(num_samples, -1)
        wind_data_reshaped = wind_data.reshape(num_samples, -1)

        # Stack the features along the last axis
        combined_data = np.stack([perc_data_reshaped, temp_data_reshaped, rad_data_reshaped, hyd_data_reshaped, relHum_data_reshaped, wind_data_reshaped], axis=-1)  # Shape: (num_samples, height * width, input_channels)
        combined_data = combined_data.reshape(num_samples, -1)  # Flatten spatial dimensions for normalization

    train_dataloader, val_dataloader, test_dataloader, scaler_x, scaler_y  = preprocess_data(combined_data, y_data, perc_data.shape, seq_length, batch_size, channels = input_channels)

    # Load the saved model state dict
    #model.load_state_dict(torch.load('/Users/SverreB/Github_Repo/sverrbey_project/model/save/CNN_LSTM_1.pth'))

    # Train the model
    #model_CNN_LSTM  = train_model(model_CNN_LSTM, train_dataloader, val_dataloader, 'CNN_LSTM_gaula_opt', scaler_y)
    #model_CNN       = train_model(model_CNN, train_dataloader, val_dataloader, 'CNN_gaula', scaler_y)
    #model_LSTM      = train_model(model_LSTM, train_dataloader, val_dataloader, 'LSTM_stryn_opt', scaler_y)

    # Load the saved model state dict
    #model_CNN_LSTM.load_state_dict(torch.load('/Users/SverreB/Github_Repo/sverrbey_project/model/save/CNN_LSTM.pth'))
    #model_CNN.load_state_dict(torch.load('/Users/SverreB/Github_Repo/sverrbey_project/model/save/CNN.pth'))
    #model_LSTM.load_state_dict(torch.load('/Users/SverreB/Github_Repo/sverrbey_project/model/save/LSTM.pth'))

    # Plot predictions vs actuals
    #plot_predictions_vs_actuals(model_CNN_LSTM, test_dataloader, scaler_y, 'CNN_LSTM_gaula')
    #plot_predictions_vs_actuals(model_CNN, test_dataloader, scaler_y, 'CNN_gaula')
    #plot_predictions_vs_actuals(model_LSTM, test_dataloader, scaler_y, 'LSTM_stryn')
    

def preprocess_data(combined_data, y_data, input_size, seq_length, batch_size,channels=3, train_split=0.7, validation_split=0.15, test_split=0.15):
    """
    * Process the data for training, validation, and testing.
    * This function normalizes the input data, creates sequences, and splits the data into training,
    validation, and testing sets.

    Args:
        combinde_data: Combined input data (shape: [num_samples, height * width * channels]).
        y_data: Target data, discharge (shape: [num_samples, num_targets]).
        seq_length: Sequence length for LSTM.
        batch_size: Batch size for DataLoader.
        train_split: Fraction of data to use for training.
        validation_split: Fraction of data to use for validation.
        test_split: Fraction of data to use for testing.

    Returns:
        train_dataloader, val_dataloader, test_dataloader, scaler_x, scaler_y
    """
    num_samples, height, width = input_size
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

def train_model(model, train_dataloader, val_dataloader, name, scaler_y, num_epochs=500, learning_rate=0.001):
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
    patience = 15  # Number of epochs to wait for improvement
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

def extract_info(df, identifier):
    catchment_info = pd.DataFrame(df[:][:9].transpose())
    column_labels = catchment_info.iloc[0]
    catchment_info = catchment_info.drop(index=0)
    catchment_info.columns = column_labels
    index_labels = catchment_info[identifier]
    catchment_info = catchment_info.drop(columns=identifier)
    catchment_info.index = index_labels
    index_labels = ['Date'] + list(index_labels)
    df = df.iloc[10:].reset_index(drop=True)
    df.columns = index_labels

    return catchment_info, df

def catchment_styrn(number_days):
    #Stryn
    df_perc = pd.read_csv('data/Stryn/DailyPrec_Stryn.txt', header=None, encoding='latin1', delimiter='\t')
    df_temp = pd.read_csv('data/Stryn/DailyTemp_Stryn.txt', header=None, encoding='latin1', delimiter='\t')
    df_evap = pd.read_csv('data/Stryn/DailyEvap_Stryn.txt', header=None, encoding='latin1', delimiter='\t')
    df_discharge = pd.read_csv('data/Stryn/DailyDisch_Stryn.txt', header=None, encoding='latin1', delimiter='\t')

    perc_info, df_perc   = extract_info(df_perc, 'Point ID')
    temp_info, df_temp   = extract_info(df_temp, 'point id')
    evap_info, df_evap   = extract_info(df_evap, 'point id')
    disch_info, df_discharge  = extract_info(df_discharge, 'point id')
    
    catchment_info = [perc_info, temp_info, evap_info, disch_info]
    points = dict()
    for input_value in catchment_info:
        for pointID in input_value.index:
            points[pointID] =  (float(input_value['Xcoord'][pointID]), float(input_value['Ycoord'][pointID]))

    perc_values = dict()
    for c in df_perc.columns[1:]:
        perc_values[c] = df_perc[c][0:number_days].to_numpy(dtype=float).flatten()

    temp_values = dict()
    for c in df_temp.columns[1:]:
        temp_values[c] = df_temp[c][0:number_days].to_numpy(dtype=float).flatten()

    evap_values = dict()
    for c in df_evap.columns[1:]:
        evap_values[c] = df_evap[c][0:number_days].to_numpy(dtype=float).flatten()

    disch_values = dict()
    for c in df_discharge.columns[1:]:
        disch_values[c] = df_discharge[c][0:number_days].to_numpy(dtype=float).flatten()
    
    dates = df_perc['Date'][0:number_days].to_numpy().flatten()
    return points, perc_values, temp_values, evap_values, disch_values, dates

def catchment_gaula(number_days):
    # Gaula
    df_perc         = pd.read_csv('data/Gaula/DailyPrecip_Gaula.txt', header=None, encoding='latin1', delimiter='\t')
    df_temp         = pd.read_csv('data/Gaula/DailyTemp_Gaula.txt', header=None, encoding='latin1', delimiter='\t')
    df_rad          = pd.read_csv('data/Gaula/DailyGlobRad_Gaula.txt', header=None, encoding='latin1', delimiter='\t')
    df_hydMetOBs    = pd.read_csv('data/Gaula/DailyHydMetObs.txt', header=None, encoding='latin1', delimiter='\t')
    df_relHum       = pd.read_csv('data/Gaula/DailyRelHum_Gaula.txt', header=None, encoding='latin1', delimiter='\t')
    df_wind         = pd.read_csv('data/Gaula/DailyWind_Gaula.txt', header=None, encoding='latin1', delimiter='\t')
    df_discharge    = pd.read_csv('data/Gaula/DailyDischarge_Gaula.txt', header=None, encoding='latin1', delimiter='\t')
    
    perc_info, df_perc   = extract_info(df_perc, 'Point ID')
    temp_info, df_temp   = extract_info(df_temp, 'point id')
    rad_info, df_rad     = extract_info(df_rad, 'Point ID')
    hyd_info, df_hydMetOBs = extract_info(df_hydMetOBs, 'point id')
    relHum_info, df_relHum = extract_info(df_relHum, 'Point ID')
    wind_info, df_wind   = extract_info(df_wind, 'Point ID')
    disch_info, df_discharge  = extract_info(df_discharge, 'point id')

    gaula_info = [perc_info,temp_info, rad_info, hyd_info, relHum_info, wind_info, disch_info]
    points_gaula = dict()
    for input_value in gaula_info:
        for pointID in input_value.index:
            points_gaula[pointID] = (float(input_value['Xcoord'][pointID]), float(input_value['Ycoord'][pointID])) 

    perc_values = dict()
    for c in df_perc.columns[1:]:
        perc_values[c] = df_perc[c][0:number_days].to_numpy(dtype=float).flatten()
    
    temp_values = dict()
    for c in df_temp.columns[1:]:
        temp_values[c] = df_temp[c][0:number_days].to_numpy(dtype=float).flatten()

    rad_values = dict()
    for c in df_rad.columns[1:]:
        rad_values[c] = df_rad[c][0:number_days].to_numpy(dtype=float).flatten()
    hyd_values = dict()
    for c in df_hydMetOBs.columns[1:]:
        hyd_values[c] = df_hydMetOBs[c][0:number_days].to_numpy(dtype=float).flatten()

    relHum_values = dict()
    for c in df_relHum.columns[1:]:
        relHum_values[c] = df_relHum[c][0:number_days].to_numpy(dtype=float).flatten()

    wind_values = dict()
    for c in df_wind.columns[1:]:
        wind_values[c] = df_wind[c][0:number_days].to_numpy(dtype=float).flatten()

    disch_values = dict()
    for c in df_discharge.columns[1:]:
        disch_values[c] = df_discharge[c][0:number_days].to_numpy(dtype=float).flatten()
    
    dates = df_perc['Date'][0:number_days].to_numpy().flatten()

    return points_gaula, perc_values, temp_values, rad_values, hyd_values, relHum_values, wind_values, disch_values, dates

if __name__ == "__main__":
    main()