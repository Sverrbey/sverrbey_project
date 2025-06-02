import torch
import os
import rasterio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pyproj

from sklearn.preprocessing  import MinMaxScaler
from torch.utils.data       import DataLoader, TensorDataset
from rasterio.enums         import Resampling

from model.HBV              import HBV_Model
from model.LSTM             import LSTM
from model.HBV_LSTM         import HBV_LSTM
from model.CNN_LSTM         import CNN_LSTM_Model
from model.CNN              import CNN


from src.plotting           import *
from src.evaluation_metrics import *
from src.topology           import *
from src.spatial_data_distribution import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

def plot_study_area():
    
    country_shape_file = r"data\Norway_shapefile\no_100km.shp"
    number_days = 6*365 #  max 42 years (80-22)
    # Timeseries
    points, _, _, _, _,  _ = catchment_styrn(number_days)

    met_stations = {}
    hydro_stations = {}
    for key, value in points.items():
        if key == "5" or key == "6": # Hydro stations
            hydro_stations[key] = (utm_to_latlon(value[0], value[1], 33, northern_hemisphere=True))
        met_stations[key] = (utm_to_latlon(value[0], value[1], 33, northern_hemisphere=True)) # Convert UTM to lat/lon
    
    stryn_catchment_file = "data\Stryn\Wtshed.tif"

    stryn_catchment_gdf, stryn_met_gdf, stryn_hydro_gdf, country_gdf = create_catchment_map("Stryn",stryn_catchment_file, met_stations, hydro_stations, country_shape_file)
    
   # Gaula
    number_days = 6*365 # max 6 years (99-05)
    # Timeseries
    points_gaula,  _, _, _, _, _, _, _, _ = catchment_gaula(number_days)

    met_stations = {}
    hydro_stations = {}
    for key, value in points_gaula.items():
        if key in ["4", "1", "2", "3", "10"]: # Hydro stations
            hydro_stations[key] = (utm_to_latlon(value[0], value[1], 32, northern_hemisphere=True))
        met_stations[key] = (utm_to_latlon(value[0], value[1], 32, northern_hemisphere=True)) # Convert UTM to lat/lon
    
    gaula_catchment_file = "data\Gaula\catchments.tiff"

    gaula_gdf, gaula_met_gdf, gaula_hydro_gdf, _ = create_catchment_map("Gaula", gaula_catchment_file, met_stations, hydro_stations, country_shape_file)
    
    plot_country_and_catchment_zoom(country_gdf, stryn_catchment_gdf, stryn_met_gdf, stryn_hydro_gdf, gaula_gdf, gaula_met_gdf, gaula_hydro_gdf)

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

def train_model(model, train_dataloader, val_dataloader, name, scaler_y, num_epochs=100, learning_rate=0.001):
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
        train_rmse = 0.0
        train_mae = 0.0
        train_mape = 0.0
        train_r_squared = 0.0
        train_kge = 0.0
        train_pearson = 0.0
       

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
            train_rmse  += RMSE_formula(y_true, y_pred)
            train_mae   += MAE_formula(y_true, y_pred)
            train_mape  += MAPE_formula(y_true, y_pred)
            train_r_squared += R_squared_formula(y_true, y_pred)
            train_kge   += KGE_formula(y_true, y_pred)
            train_pearson   += Pearson_formula(y_true, y_pred)


        train_loss /= len(train_dataloader)
        train_nse /= len(train_dataloader)

        # Validation loop
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
            for batch_x, batch_y in val_dataloader:
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()

                # Denormalize the predictions and true values
                y_true = scaler_y.inverse_transform(batch_y.detach().cpu().numpy())
                y_pred = scaler_y.inverse_transform(outputs.detach().cpu().numpy())

                val_nse   += NSE_formula(y_true, y_pred)
                val_rmse  += RMSE_formula(y_true, y_pred)
                val_mae   += MAE_formula(y_true, y_pred)
                val_mape  += MAPE_formula(y_true, y_pred)
                val_r_squared += R_squared_formula(y_true, y_pred)
                val_kge   += KGE_formula(y_true, y_pred)
                val_pearson   += Pearson_formula(y_true, y_pred)

        val_loss /= len(val_dataloader)
        val_nse /= len(val_dataloader)

        # Print epoch results
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train NSE: {train_nse:.4f}, Val Loss: {val_loss:.4f}, Val NSE: {val_nse:.4f}")
        #print(f"Train RMSE: {train_rmse:.4f}, Train MAE: {train_mae:.4f}, Train MAPE: {train_mape:.4f}, Train R^2: {train_r_squared:.4f}, Train KGE: {train_kge:.4f}, Train Pearson: {train_pearson:.4f}")
        #print(f"Val RMSE: {val_rmse:.4f}, Val MAE: {val_mae:.4f}, Val MAPE: {val_mape:.4f}, Val R^2: {val_r_squared:.4f}, Val KGE: {val_kge:.4f}, Val Pearson: {val_pearson:.4f}")
        # Early stopping logic
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stop_counter = 0
            # Save the best model
            torch.save(model.state_dict(), 'model/save' + name + '_best.pth')
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
    result = np.full(len(data), False, dtype=bool)
    valid_set = set()
    for x in range(len(data) - seq_length):
        #Step 1: We create a sequence, and if the sequence contains -99.0 we skip it
        possible_seq = data[x:x+seq_length]
        if -99.0 in possible_seq or np.isnan(possible_seq).any():
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
    long_list = np.repeat(long_list, len(lat_list))
    lat_list = np.tile(lat_list, len(long_list) // len(lat_list))
    
    return long_list, lat_list, width, height
