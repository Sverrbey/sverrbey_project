import torch
import os
import rasterio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pyproj
import geopandas as gpd

from hmmlearn.hmm           import GaussianHMM
from sklearn.preprocessing  import MinMaxScaler
from torch.utils.data       import DataLoader, TensorDataset
from rasterio.enums         import Resampling
from hmmlearn.hmm           import GaussianHMM
from src.plotting           import *
from src.evaluation_metrics import *
from src.topology           import *
from src.spatial_data_distribution import *

def plot_study_area():
    geo_jsons = [
        'https://geodata.ucdavis.edu/gadm/gadm4.1/json/gadm41_NOR_0.json'
    ]
    geo_df_total = pd.DataFrame()
    for geo_json in geo_jsons:
        geo_df = gpd.read_file(geo_json)
        geo_df_total = gpd.GeoDataFrame(pd.concat([geo_df_total, geo_df], ignore_index=True))
    
    geo_axes = geo_df_total.plot(facecolor="#6decf0", edgecolor='#fff', linewidth=.2, figsize=(5, 8), zorder=0)
    
    stryn_long = int( (7.545843 + 6.758247)/2 )
    stryn_lat =  int( (61.793368 + 62.009898)/2)
    gaula_long = int((9.779947 + 11.873204)/2)
    gaula_lat =  int((62.633520 + 63.100231)/2)
   
    geo_axes.scatter([stryn_long], [stryn_lat], color="#088819", marker='o', label='Stryn', zorder=3)
    geo_axes.scatter([gaula_long], [gaula_lat], color="#1F58D4", marker='o', label='Gaula', zorder=3)
    plt.gcf().axes[0].axis('off')
    plt.legend(loc='upper left', fontsize=12, markerscale=1.5)
    plt.savefig('catchment_locations.jpg')


def preprocess_data(combined_data, y_data, input_size, seq_length, batch_size, num_samples, channels=3, train_split=0.7, validation_split=0.15, test_split=0.15):
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
    _, height, width = input_size
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

    return train_dataloader, val_dataloader, test_dataloader, scaler_y

def preprocess_data_no_sequence(combined_data, y_data, input_size, batch_size, channels=3, train_split=0.7, validation_split=0.15, test_split=0.15):
    """
    * Process the data for training, validation, and testing.
    * This function normalizes the input data, creates sequences, and splits the data into training,
    validation, and testing sets.

    Args:
        combinde_data: Combined input data (shape: [num_samples, height * width * channels]).
        y_data: Target data, discharge (shape: [num_samples, num_targets]).
        batch_size: Batch size for DataLoader.
        train_split: Fraction of data to use for training.
        validation_split: Fraction of data to use for validation.
        test_split: Fraction of data to use for testing.

    Returns:
        train_dataloader, val_dataloader, test_dataloader, scaler_x, scaler_y
    """
    # Ensure input_size is a tuple of (num_samples, height, width)
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
    combined_data_normalized = scaler_x.transform(combined_data)  # Shape: (num_samples,channels * height * width )
    y_data_normalized = scaler_y.transform(y_data)               # Shape: (num_samples, num_targets)

    # Reshape combined_data back to 3D (spatial dimensions restored)
    combined_data_normalized = combined_data_normalized.reshape(num_samples, channels, height, width) 

    x_train, x_val, x_test = combined_data_normalized[:train_end], combined_data_normalized[train_end:val_end], combined_data_normalized[val_end:]
    y_train, y_val, y_test = y_data_normalized[:train_end], y_data_normalized[train_end:val_end], y_data_normalized[val_end:]

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

def train_model(model, train_dataloader, val_dataloader, name, scaler_y, num_epochs=100, learning_rate=0.003):
    """
    Train the ML model.

    Args:
        model: The ML model to train.
        train_dataloader: DataLoader for training data.
        val_dataloader: DataLoader for validation data.
        name: Name to save the trained model.
        seq_length: Sequence length of the input data.
        num_epochs: Number of epochs to train the model.
        learning_rate: Learning rate for the optimizer.

    Returns:
        model: The trained model.
    """
    #model = model.to(device)  # Move model to the specified device (GPU or CPU)
    # Define the loss function and optimizer
    criterion = torch.nn.MSELoss()   # Mean Squared Error Loss
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

            train_nse   += NSE_formula(y_true, y_pred)
    
        train_loss /= len(train_dataloader)
        train_nse /= len(train_dataloader)

        # Validation loop
        model.eval()
        val_loss = 0.0
        val_nse  = 0.0

        with torch.no_grad():
            for batch_x, batch_y in val_dataloader:
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()

                # Denormalize the predictions and true values
                y_true = scaler_y.inverse_transform(batch_y.detach().cpu().numpy())
                y_pred = scaler_y.inverse_transform(outputs.detach().cpu().numpy())

                val_nse   += NSE_formula(y_true, y_pred)

        val_loss /= len(val_dataloader)
        val_nse /= len(val_dataloader)


        # Print epoch results
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train NSE: {train_nse:.4f}, Val Loss: {val_loss:.4f}, Val NSE: {val_nse:.4f}")
    
        # Early stopping logic
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stop_counter = 0
            # Save the best model
            torch.save(model.state_dict(), 'model/save/' + name + '_best.pth')
        else:
            early_stop_counter += 1
            print(f"Early stopping counter: {early_stop_counter}/{patience}")

        if early_stop_counter >= patience:
            print("Early stopping triggered. Stopping training.")
            break
    # Print final results
    print("Training complete.")
    # Save the final model
    torch.save(model.state_dict(), 'model/save/' + name + '.pth')
    return model

def train_model_multi(model, train_loader, val_loader, name, scaler_y, num_epochs=100, lr=1e-3):
    """
    Train a (multi-scale) model with two input sequences per sample.
    """


    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)


    # Early stopping parameters
    patience = 10  # Number of epochs to wait for improvement
    best_val_loss = float('inf')
    early_stop_counter = 0

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_nse = 0.0
        train_rmse = 0.0
        train_mae = 0.0
        train_mape = 0.0
        train_r_squared = 0.0
        train_kge = 0.0
        train_pearson = 0.0

        for *x_inputs, batch_y in train_loader:
            # x_inputs: [x_short, x_long, ...]
            optimizer.zero_grad()
            outputs = model(*x_inputs)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * batch_y.size(0)

            # Denormalize the predictions and true values
            y_true = scaler_y.inverse_transform(batch_y.detach().cpu().numpy())
            y_pred = scaler_y.inverse_transform(outputs.detach().cpu().numpy())

            # Calculate evaluation metrics
            train_nse   += NSE_formula(y_true, y_pred)
            train_rmse  += RMSE_formula(y_true, y_pred)
            train_mae   += MAE_formula(y_true, y_pred)
            train_mape  += MAPE_formula(y_true, y_pred)
            train_r_squared += R_squared_formula(y_true, y_pred)
            train_kge   += KGE_formula(y_true, y_pred)
            train_pearson   += Pearson_formula(y_true, y_pred)
        train_loss /= len(train_loader.dataset)
        train_nse /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        val_nse  = 0.0
        val_rmse = 0.0
        val_mae  = 0.0
        val_mape = 0.0
        val_r_squared = 0.0
        val_kge  = 0.0
        val_pearson   = 0.0
        with torch.no_grad():
            for *x_inputs, batch_y in val_loader:
                outputs = model(*x_inputs)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item() * batch_y.size(0)

                # Denormalize the predictions and true values
                y_true = scaler_y.inverse_transform(batch_y.detach().cpu().numpy())
                y_pred = scaler_y.inverse_transform(outputs.detach().cpu().numpy())
                # Calculate evaluation metrics
                val_nse   += NSE_formula(y_true, y_pred)
                val_rmse  += RMSE_formula(y_true, y_pred)
                val_mae   += MAE_formula(y_true, y_pred)
                val_mape  += MAPE_formula(y_true, y_pred)
                val_r_squared += R_squared_formula(y_true, y_pred)
                val_kge   += KGE_formula(y_true, y_pred)
                val_pearson   += Pearson_formula(y_true, y_pred)
        val_loss /= len(val_loader.dataset)
        val_nse /= len(val_loader)

        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train NSE: {train_nse:.4f}, Val Loss: {val_loss:.4f}, Val NSE: {val_nse:.4f}")

        # Early stopping logic
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stop_counter = 0
            # Save the best model
            torch.save(model.state_dict(), 'model/save_' + name + '_best.pth')
        else:
            early_stop_counter += 1
            print(f"Early stopping counter: {early_stop_counter}/{patience}")

        if early_stop_counter >= patience:
            print("Early stopping triggered. Stopping training.")
            break
    # Print final results
    print("Training complete.")
    print("###### Final Results ######")
    print(f"Val NSE: {val_nse:.4f}, Val RMSE: {val_rmse:.4f}, Val MAE: {val_mae:.4f}, Val MAPE: {val_mape:.4f}, Val R^2: {val_r_squared:.4f}, Val KGE: {val_kge:.4f}, Val Pearson: {val_pearson:.4f}")
    # Save the final model
    torch.save(model.state_dict(), 'model/save' + name + '.pth')
    return model

def filter_valid_indices(data, seq_length):
    """
    Filters out invalid data and creates sequences of a specified length.

    Args:
        data (list): Input data containing valid and invalid values.
        seq_length (int): Desired sequence length.

    Returns:
        set: set of valid indices.
    """
    valid_indices = []
    for i in range(len(data) - seq_length + 1):
        window = data[i:i+seq_length]
        if np.all(~np.isnan(window)):
            valid_indices.append(i)
    return np.array(valid_indices)

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
    df = df.set_index('Date')

    return catchment_info, df

def catchment_styrn(number_days):
    #Stryn
    df_prec = pd.read_csv('data\Stryn\HBV snow\DailyPrec.txt', header=None, encoding='latin1', delimiter='\t', na_values=[-99.0, -99], skip_blank_lines=True)
    df_temp = pd.read_csv('data\Stryn\HBV snow\Dailytemp.txt', header=None, encoding='latin1', delimiter='\t', na_values=[-99.0, -99], skip_blank_lines=True)
    df_evap = pd.read_csv('data\Stryn\HBV snow\DailyEvap.txt', header=None, encoding='latin1', delimiter='\t', na_values=[-99.0, -99], skip_blank_lines=True)
    df_rad  = pd.read_csv('data\Stryn\HBV snow\DailyGlobRad.txt', header=None, encoding='latin1', delimiter='\t', na_values=[-99.0, -99], skip_blank_lines=True)
    df_relHum = pd.read_csv('data\Stryn\HBV snow\DailyRelHum.txt', header=None, encoding='latin1', delimiter='\t', na_values=[-99.0, -99], skip_blank_lines=True)
    df_wind = pd.read_csv('data\Stryn\HBV snow\DailyWind.txt', header=None, encoding='latin1', delimiter='\t', na_values=[-99.0, -99], skip_blank_lines=True)
    df_discharge = pd.read_csv('data\Stryn\HBV snow\DailyDisch.txt', header=None, encoding='latin1', delimiter='\t', na_values=[-99.0, -99, "#VALUE!"], skip_blank_lines=True)
    
    df_discharge    = df_discharge.dropna(how='all')
    df_prec         = df_prec.dropna(how='all')
    df_temp         = df_temp.dropna(how='all')
    df_rad          = df_rad.dropna(how='all')
    df_relHum       = df_relHum.dropna(how='all')
    df_wind         = df_wind.dropna(how='all')
    df_evap         = df_evap.dropna(how='all')

    # Remove rows where all elements are empty strings
    df_discharge = df_discharge[~(df_discharge.apply(lambda row: row.map(lambda x: str(x).strip() == '').all(), axis=1))]
    df_prec      = df_prec[~(df_prec.apply(lambda row: row.map(lambda x: str(x).strip() == '').all(), axis=1))]
    df_temp      = df_temp[~(df_temp.apply(lambda row: row.map(lambda x: str(x).strip() == '').all(), axis=1))]
    df_rad       = df_rad[~(df_rad.apply(lambda row: row.map(lambda x: str(x).strip() == '').all(), axis=1))]
    df_relHum    = df_relHum[~(df_relHum.apply(lambda row: row.map(lambda x: str(x).strip() == '').all(), axis=1))]
    df_wind      = df_wind[~(df_wind.apply(lambda row: row.map(lambda x: str(x).strip() == '').all(), axis=1))]
    df_evap      = df_evap[~(df_evap.apply(lambda row: row.map(lambda x: str(x).strip() == '').all(), axis=1))]

    prec_info, df_prec   = extract_info(df_prec, 'Point ID')                                                        
    temp_info, df_temp   = extract_info(df_temp, 'point id')
    evap_info, df_evap   = extract_info(df_evap, 'point id')
    rad_info, df_rad     = extract_info(df_rad, 'Point ID')
    relHum_info, df_relHum = extract_info(df_relHum, 'Point ID')
    wind_info, df_wind   = extract_info(df_wind, 'Point ID')
    disch_info, df_discharge  = extract_info(df_discharge, 'point id')
    
    catchment_info = [prec_info, temp_info, evap_info,  disch_info, rad_info, relHum_info, wind_info]
    points = dict()
    for input_value in catchment_info:
        for pointID in input_value.index:
            points[pointID] =  (float(input_value['Xcoord'][pointID]), float(input_value['Ycoord'][pointID]))
    
    perc_values = dict()
    for c in df_prec.columns:
        perc_values[c] = df_prec[c][0:number_days].to_numpy(dtype=float).flatten()

    temp_values = dict()
    for c in df_temp.columns:
        temp_values[c] = df_temp[c][0:number_days].to_numpy(dtype=float).flatten()
    
    evap_values = dict()
    for c in df_evap.columns:
        evap_values[c] = df_evap[c][0:number_days].to_numpy(dtype=float).flatten()

    disch_values = dict()
    for c in df_discharge.columns:
        disch_values[c] = df_discharge[c][0:number_days].to_numpy(dtype=float).flatten()

    rad_values = dict()
    for c in df_rad.columns:
        rad_values[c] = df_rad[c][0:number_days].to_numpy(dtype=float).flatten()
    
    relHum_values = dict()
    for c in df_relHum.columns:
        relHum_values[c] = df_relHum[c][0:number_days].to_numpy(dtype=float).flatten()
    
    wind_values = dict()
    for c in df_wind.columns:
        wind_values[c] = df_wind[c][0:number_days].to_numpy(dtype=float).flatten()
    
    return points, perc_values, temp_values, disch_values, evap_values, rad_values, relHum_values, wind_values

def catchment_gaula(number_days):
    # Gaula
    df_prec         = pd.read_csv('data/Gaula/DailyPrecip_Gaula.txt', header=None, encoding='latin1', delimiter='\t', na_values=[-99.0, -99], skip_blank_lines=True)
    df_temp         = pd.read_csv('data/Gaula/DailyTemp_Gaula.txt', header=None, encoding='latin1', delimiter='\t', na_values=[-99.0, -99], skip_blank_lines=True)
    df_rad          = pd.read_csv('data/Gaula/DailyGlobRad_Gaula.txt', header=None, encoding='latin1', delimiter='\t', na_values=[-99.0, -99], skip_blank_lines=True)
    df_relHum       = pd.read_csv('data/Gaula/DailyRelHum_Gaula.txt', header=None, encoding='latin1', delimiter='\t', na_values=[-99.0, -99], skip_blank_lines=True)
    df_wind         = pd.read_csv('data/Gaula/DailyWind_Gaula.txt', header=None, encoding='latin1', delimiter='\t', na_values=[-99.0, -99], skip_blank_lines=True)
    df_discharge    = pd.read_csv('data/Gaula/DailyDischarge_Gaula.txt', header=None, encoding='latin1', delimiter='\t', na_values=[-99.0, -99], skip_blank_lines=True)
    
    df_discharge    = df_discharge.dropna(how='all')
    df_prec         = df_prec.dropna(how='all')
    df_temp         = df_temp.dropna(how='all')
    df_rad          = df_rad.dropna(how='all')
    df_relHum       = df_relHum.dropna(how='all')
    df_wind         = df_wind.dropna(how='all')

    # Remove rows where all elements are empty strings
    df_discharge = df_discharge[~(df_discharge.apply(lambda row: row.map(lambda x: str(x).strip() == '').all(), axis=1))]
    df_prec      = df_prec[~(df_prec.apply(lambda row: row.map(lambda x: str(x).strip() == '').all(), axis=1))]
    df_temp      = df_temp[~(df_temp.apply(lambda row: row.map(lambda x: str(x).strip() == '').all(), axis=1))]
    df_rad       = df_rad[~(df_rad.apply(lambda row: row.map(lambda x: str(x).strip() == '').all(), axis=1))]
    df_relHum    = df_relHum[~(df_relHum.apply(lambda row: row.map(lambda x: str(x).strip() == '').all(), axis=1))]
    df_wind      = df_wind[~(df_wind.apply(lambda row: row.map(lambda x: str(x).strip() == '').all(), axis=1))]

    df_discharge.drop(columns=[3,5], inplace=True)

    prec_info, df_prec   = extract_info(df_prec, 'Point ID')
    temp_info, df_temp   = extract_info(df_temp, 'point id')
    rad_info, df_rad     = extract_info(df_rad, 'Point ID')
    wind_info, df_wind   = extract_info(df_wind, 'Point ID')
    relHum_info, df_relHum = extract_info(df_relHum, 'Point ID')
    disch_info, df_discharge  = extract_info(df_discharge, 'point id')

    gaula_info = [prec_info,temp_info, rad_info, relHum_info, wind_info, disch_info]
    points = dict()
    for input_value in gaula_info:
        for pointID in input_value.index:
            points[pointID] = (float(input_value['Xcoord'][pointID]), float(input_value['Ycoord'][pointID])) 
    
    prec_values = dict()
    for c in df_prec.columns:
        prec_values[c] = df_prec[c][0:number_days].to_numpy(dtype=float).flatten()
    
    temp_values = dict()
    for c in df_temp.columns:
        temp_values[c] = df_temp[c][0:number_days].to_numpy(dtype=float).flatten()

    rad_values = dict()
    for c in df_rad.columns:
        rad_values[c] = df_rad[c][0:number_days].to_numpy(dtype=float).flatten()

    relHum_values = dict()
    for c in df_relHum.columns:
        relHum_values[c] = df_relHum[c][0:number_days].to_numpy(dtype=float).flatten()

    wind_values = dict()
    for c in df_wind.columns:
        wind_values[c] = df_wind[c][0:number_days].to_numpy(dtype=float).flatten()

    disch_values = dict()
    for c in df_discharge.columns:
        disch_values[c] = df_discharge[c][0:number_days].to_numpy(dtype=float).flatten()

    return points, prec_values, temp_values, disch_values, rad_values, relHum_values, wind_values

def utm_to_latlon(easting, northing, zone_number=33, northern_hemisphere=True):
    """
    Convert UTM coordinates to geographical coordinates (latitude and longitude).
    :param easting: UTM easting coordinate.
    :param northing: UTM northing coordinate.
    :param zone_number: UTM zone number.
    :param northern_hemisphere: Boolean indicating if the coordinates are in the northern hemisphere.
    :return: Tuple of (latitude, longitude).
    """
    # Define the UTM projection
    utm_proj = pyproj.Proj(proj='utm', zone=zone_number, ellps='WGS84', south=not northern_hemisphere)
    
    # Define the WGS84 projection
    wgs84_proj = pyproj.Proj(proj='latlong', datum='WGS84')
    
    # Perform the transformation
    lon, lat = pyproj.transform(utm_proj, wgs84_proj, easting, northing)
    
    return lon, lat

def create_grid(longitudes, latitudes, spacing):
    """
    Create a list of longitudes and latitudes for the catchment area. To be used in the UKMO download script.
    """
    space = spacing/111.32 # km per degree of latitude
    long_list = np.arange(min(longitudes), max(longitudes), space)
    lat_list = np.arange(latitudes[0], latitudes[1], space)

    height = len(lat_list)
    width = len(long_list)
    # make a list of all combinations of longitudes and latitudes
    #long_list = np.repeat(long_list, len(lat_list))
    #lat_list = np.tile(lat_list, len(long_list) // len(lat_list))
    
    return long_list, lat_list, width, height

def weighted_mse_loss(outputs, targets, threshold=0.8, high_weight=2.0):
    # threshold: value above which events are considered "large"
    weights = (targets > threshold).float() * high_weight + (targets <= threshold).float()
    return (weights * (outputs - targets) ** 2).mean()

def HMMR_3(observation, forecast, n_states=2):
        """
        observation: Historical streamflow: (sequence, output_size)
        forecast: Forecasted streamflow: (sequence, output_size)

        State 0: Normal Operation
        State 1: Drought Management 10th Percentile
        State 2: Flood Management 90th Percentile
        """ 
        hmm_1= GaussianHMM(n_components=n_states, covariance_type="diag", n_iter=100)
        hmm_2= GaussianHMM(n_components=n_states, covariance_type="diag", n_iter=100)
        hmm_3= GaussianHMM(n_components=n_states, covariance_type="diag", n_iter=100)
       

        batch_size, output_size = forecast.shape
        # Flatten the forecasts to fit the HMM input requirements
        forecast_1      = forecast[:, 0].reshape(-1, 1)  # Reshape to 2D array for HMM
        observation_1   = observation[:, 0].reshape(-1, 1)
        forecast_2      = forecast[:, 1].reshape(-1, 1)  # Reshape to 2D array for HMM
        observation_2   = observation[:, 1].reshape(-1, 1)
        forecast_3      = forecast[:, 2].reshape(-1, 1)  # Reshape to 2D array for HMM
        observation_3   = observation[:, 2].reshape(-1, 1)
        

        
        uncertainty = np.zeros((output_size, n_states))  # Initialize uncertainty array
        residual_1 = forecast_1 - observation_1 # Forecast and observation has shape (batch_size, 1)
        residual_2 = forecast_2 - observation_2 # Forecast and observation has shape (batch_size, 1)
        residual_3 = forecast_3 - observation_3 # Forecast and observation has shape (batch_size, 1)
        

        # Estimate model parameters for the HMMs
        hmm_1.fit(observation_1)
        hmm_2.fit(observation_2)
        hmm_3.fit(observation_3)
       

        # Find the most likely state sequence corresponding to the 
        h1 = hmm_1.predict(observation_1)
        h2 = hmm_2.predict(observation_2)
        h3 = hmm_3.predict(observation_3)
        

        #Plotting the variance of the hidden states might be more useful than the mean residuals

        # Calculate the standard deviation of the residuals for each hidden state
        std_deviation = np.zeros((output_size, n_states))
        for state in range(n_states):
            indices_1 = np.where(h1 == state)[0]
            indices_2 = np.where(h2 == state)[0]
            indices_3 = np.where(h3 == state)[0]
           
            if len(indices_1) > 0:
                std_deviation[0, state] = np.std(residual_1[indices_1])

            if len(indices_2) > 0:
                std_deviation[1, state] = np.std(residual_2[indices_2])

            if len(indices_3) > 0:
                std_deviation[2, state] = np.std(residual_3[indices_3])
            
            

        return std_deviation, h1, h2, h3 # Shape: (batch_size, output_size)

def HMMR_2(observation, forecast, n_states=2):
        """
        observation: Historical streamflow: (sequence, output_size)
        forecast: Forecasted streamflow: (sequence, output_size)

        State 0: Normal Operation
        State 1: Drought Management 10th Percentile
        State 2: Flood Management 90th Percentile
        """ 
        hmm_1= GaussianHMM(n_components=n_states, covariance_type="diag", n_iter=100)
        hmm_2= GaussianHMM(n_components=n_states, covariance_type="diag", n_iter=100)
        #hmm_3= GaussianHMM(n_components=n_states, covariance_type="diag", n_iter=100)
        #hmm_4= GaussianHMM(n_components=n_states, covariance_type="diag", n_iter=100)

        batch_size, output_size = forecast.shape
        # Flatten the forecasts to fit the HMM input requirements
        forecast_1      = forecast[:, 0].reshape(-1, 1)  # Reshape to 2D array for HMM
        observation_1   = observation[:, 0].reshape(-1, 1)
        forecast_2      = forecast[:, 1].reshape(-1, 1)  # Reshape to 2D array for HMM
        observation_2   = observation[:, 1].reshape(-1, 1)

        uncertainty = np.zeros((output_size, n_states))  # Initialize uncertainty array
        residual_1 = forecast_1 - observation_1 # Forecast and observation has shape (batch_size, 1)
        residual_2 = forecast_2 - observation_2 # Forecast and observation has shape (batch_size, 1)
        #residual_3 = forecast_3 - observation_3 # Forecast and observation has shape (batch_size, 1)
        #residual_4 = forecast_4 - observation_4 # Forecast and observation has shape (batch_size, 1)

        # Estimate model parameters for the HMMs
        hmm_1.fit(observation_1)
        hmm_2.fit(observation_2)
        #hmm_3.fit(observation_3)
        #hmm_4.fit(observation_4)

        # Find the most likely state sequence corresponding to the 
        h1 = hmm_1.predict(observation_1)
        h2 = hmm_2.predict(observation_2)
        #h3 = hmm_3.predict(observation_3)
        #h4 = hmm_4.predict(observation_4)

        #Plotting the variance of the hidden states might be more useful than the mean residuals

        # Calculate the standard deviation of the residuals for each hidden state
        std_deviation = np.zeros((output_size, n_states))
        for state in range(n_states):
            indices_1 = np.where(h1 == state)[0]
            indices_2 = np.where(h2 == state)[0]
            #indices_3 = np.where(h3 == state)[0]
            #indices_4 = np.where(h4 == state)[0]
            if len(indices_1) > 0:
                std_deviation[0, state] = np.std(residual_1[indices_1])

            if len(indices_2) > 0:
                std_deviation[1, state] = np.std(residual_2[indices_2])

            #if len(indices_3) > 0:
            #    std_deviation[2, state] = np.std(residual_3[indices_3])
            #
            #if len(indices_4) > 0:
            #    std_deviation[3, state] = np.std(residual_4[indices_4])

        return std_deviation, h1, h2#, h3, h4 # Shape: (batch_size, output_size)

def integrated_gradients(model, 
                         input_tensor,
                         channels=1, 
                         baseline=None, 
                         target_index=None, 
                         steps=50):
    """
    Compute Integrated Gradients for a given model and input tensor.

    Args:
        model: PyTorch model (e.g., LSTM) with a forward() method.
        input_tensor: Input tensor for which to compute attributions (shape: [batch, seq_len, ...]).
        baseline: Baseline tensor (same shape as input_tensor). If None, uses zeros.
        target_index: Index of the output to compute gradients for (for multi-output models).
        steps: Number of steps for the Riemann approximation of the integral.

    Returns:
        attributions: Integrated gradients attributions (same shape as input_tensor).
    """
    
    batch_size, seq_length, channels, height, width = input_tensor.size()
    if baseline is None:
        baseline = torch.zeros_like(input_tensor)
    else:
        baseline = baseline.clone().detach()

    attributions = torch.zeros_like(input_tensor)
    for channel in range(channels):
        # Only interpolate the current channel, keep others at baseline
        channel_attr = torch.zeros_like(input_tensor)
        for alpha in torch.linspace(0, 1, steps+1):
            # Interpolate only the current channel
            interp = baseline.clone()
            interp[..., channel,...] = baseline[..., channel,...] + alpha * (input_tensor[..., channel,...] - baseline[..., channel,...])
            interp.requires_grad_(True)
            output = model(interp)
            if target_index is not None:
                output = output[..., target_index]
            output = output.sum()
            grad = torch.autograd.grad(output, interp, retain_graph=True)[0]
            channel_attr += grad
        # Average gradients and scale by input difference
        avg_grad = channel_attr / (steps + 1)
        attributions[..., channel,...] = (input_tensor[..., channel,...] - baseline[..., channel,...]) * avg_grad[..., channel,...]
    
    return attributions
    
def pair_wise_integrated_gradient(model, input_tensor, target_class, baseline_tensor, steps=50):
    """
    Compute pair-wise Integrated Gradients for a given model and input tensor.
    
    Args:
        model: The model to explain.
        input_tensor: Input tensor for which to compute the gradients.
        target_class: The target class for which to compute the gradients.
        baseline_tensor: Baseline tensor for Integrated Gradients.
        steps: Number of steps for approximation.
    
    Returns:
        Integrated gradients for each pair of inputs.
    """
    # Ensure input_tensor is a 2D tensor
    if input_tensor.dim() == 1:
        input_tensor = input_tensor.unsqueeze(0)
    
    # Initialize integrated gradients
    integrated_grads = torch.zeros_like(input_tensor)

    # Compute gradients for each pair of inputs
    for i in range(input_tensor.size(0)):
        integrated_grads[i] = integrated_gradient(model, input_tensor[i].unsqueeze(0), target_class, baseline_tensor[i].unsqueeze(0), steps)

    return integrated_grads



