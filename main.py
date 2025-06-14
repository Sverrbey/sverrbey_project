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

HydAPI key: bq5Ny6WGYkK5ySwwkqjCBQ==

"""
# Import necessary libraries & models
import torch
import time
import os
import rasterio
import cv2
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pyproj
from scipy.ndimage import gaussian_filter1d

from model.LSTM             import LSTM
from model.Bi_LSTM          import BiLSTM
from model.LSTM_flat        import LSTM_flat
from model.Bi_LSTM_flat     import BiLSTM_flat  
from model.CNN_LSTM         import CNN_LSTM
from model.CNN_MLP          import CNN_MLP
from model.Decoupled_model  import Decoupled_LSTM
from model.double_model     import Double_LSTM

from src.plotting           import *
from src.evaluation_metrics import *
from src.topology           import *
from src.spatial_data_distribution import *
from src.functions          import *
from src.HydAPI_download    import *

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
    """

    stryn_file_paths = [
        'data/interpolated_spatial_data/stryn_prec.json', 
        'data/interpolated_spatial_data/stryn_temp.json', 
        'data/interpolated_spatial_data/stryn_discharge.json',
        'data/interpolated_spatial_data/stryn_rad.json',
        'data/interpolated_spatial_data/stryn_relHum.json',
        'data/interpolated_spatial_data/stryn_wind.json',
        'data/interpolated_spatial_data/stryn_evap.json'
    ]

    stryn_flat_file_paths = [
        'data/flat_timeseries/stryn_prec.json',
        'data/flat_timeseries/stryn_temp.json',
        'data/flat_timeseries/stryn_discharge.json',
        'data/flat_timeseries/stryn_rad.json',
        'data/flat_timeseries/stryn_relHum.json',
        'data/flat_timeseries/stryn_wind.json',
        'data/flat_timeseries/stryn_evap.json'
    ]
    
    gaula_file_paths = [
        'data/interpolated_spatial_data/gaula_prec.json',
        'data/interpolated_spatial_data/gaula_temp.json',
        'data/interpolated_spatial_data/gaula_disch.json',
        'data/interpolated_spatial_data/gaula_rad.json',
        'data/interpolated_spatial_data/gaula_relHum.json',
        'data/interpolated_spatial_data/gaula_wind.json',
        'data/interpolated_spatial_data/gaula_hyd.json',
    ]

 

    Stryn_HBV_result_1 = pd.read_csv("data/SimulatedDischarge_Grasdøla.txt", sep="\t", skiprows=12, names=["Datetime", "Q","Qsim"], na_values=[-99.0, -99, "#VALUE!"])
    Stryn_HBV_result_2 = pd.read_csv("data/SimulatedDischarge_Strynsvatn.txt", sep="\t", skiprows=12, names=["Datetime", "Q","Qsim"], na_values=[-99.0, -99, "#VALUE!"])
    Stryn_dates        = pd.to_datetime(Stryn_HBV_result_1["Datetime"], format="%d.%m.%Y %H:%M")
    Stryn_HBV_actuals = pd.concat([Stryn_HBV_result_1["Q"], Stryn_HBV_result_2["Q"]], axis=1)
    Stryn_HBV_actuals.columns = ["Station_1", "Station_2"]
    Stryn_HBV = pd.concat([Stryn_HBV_result_1["Qsim"], Stryn_HBV_result_2["Qsim"]], axis=1)
    Stryn_HBV.columns = ["Station_1", "Station_2"]

    Gaula_RAVEN_result_1 = pd.read_csv("data\GaulaResult_Gaulfoss local.txt", sep="\t", skiprows=12, names=["Datetime", "Q","Qsim"], na_values=[-99.0, -99, "#VALUE!"])
    Gaula_RAVEN_result_2 = pd.read_csv("data\GaulaResult_Lillebudal Bru  .txt", sep="\t", skiprows=12, names=["Datetime", "Q","Qsim"], na_values=[-99.0, -99, "#VALUE!"])
    Gaula_RAVEN_result_3 = pd.read_csv("data/GaulaResult_Hugdal Bru  .txt", sep="\t", skiprows=12, names=["Datetime", "Q","Qsim"], na_values=[-99.0, -99, "#VALUE!"])
    Gaula_RAVEN_result_4 =  pd.read_csv("data/GaulaResult_Eggafoss.txt", sep="\t", skiprows=12, names=["Datetime", "Q","Qsim"], na_values=[-99.0, -99, "#VALUE!"])
    Gaula_dates = pd.to_datetime(Gaula_RAVEN_result_1["Datetime"], format="%d.%m.%Y %H:%M")
    Gaula_actuals = pd.concat([Gaula_RAVEN_result_1["Q"], Gaula_RAVEN_result_2["Q"], Gaula_RAVEN_result_4["Q"]], axis=1)
    Gaula_actuals.columns = ["Station_1", "Station_2", "Station_3"]
   

    Gaula_RAVEN = pd.concat([Gaula_RAVEN_result_1["Qsim"], Gaula_RAVEN_result_2["Qsim"], Gaula_RAVEN_result_4["Qsim"]], axis=1)
    Gaula_RAVEN.columns = ["Station_1", "Station_2", "Station_4"]

    # Gaula catchment
    #spacing     = 8000  # Spacing between grid points 
    #number_days = 2191 # max 6 years (99-05)
    #Gaula_RAVEN = Gaula_RAVEN[:number_days]  # Limit to 6 years of data
    #Gaula_RAVEN = Gaula_RAVEN.to_numpy()  # Convert to numpy array for further processing
    #Gaula_actuals = Gaula_actuals[:number_days]  # Limit to 6 years of data
    #Gaula_actuals = Gaula_actuals.to_numpy()  # Convert to numpy array for further processing
    #Gaula_dates = Gaula_dates[:number_days]  # Limit to 6 years of data
    #x_limit = (540000, 645000)
    #y_limit = (6945000,7000000)
    #x_degrees   = ( 9.779947,  11.873204) 
    #y_degrees   = (62.633520, 63.100231) 
    #extent      = (x_limit[0], x_limit[1], y_limit[0], y_limit[1])  
    #
    ## Timeseries
    #points, prec_values, temp_values, disch_values, rad_values, hyd_values, relHum_values, wind_values = catchment_gaula(number_days)    
    

    # Stryn catchment
    spacing  = 4000         # Spacing between grid points 
    number_days = 15377     #  (01-01-1980 - 06-02-2022)
    Stryn_HBV = Stryn_HBV[:number_days]  # Limit to 42 years of data
    Stryn_HBV = Stryn_HBV.to_numpy()  # Convert to numpy array for further processing
    Stryn_HBV_actuals = Stryn_HBV_actuals[:number_days]  # Limit to 42 years of data
    Stryn_HBV_actuals = Stryn_HBV_actuals.to_numpy()  # Convert to numpy array for further processing
    Stryn_dates = Stryn_dates[:number_days]  # Limit to 42 years of data
    
    x_limit = (  69295, 107545)
    y_limit = (6873705, 6902705)
    x_degrees   = ( 6.758247, 7.545843)       # (6°42'21",  7°40'03")
    y_degrees   = (61.793368, 62.009898)    # (61°44'11", 61°59'56")
    extent  = (x_limit[0], x_limit[1], y_limit[0], y_limit[1])  
    # Timeseries
    points, prec_values, temp_values, disch_values, evap_values, rad_values, relHum_values, wind_values = catchment_styrn(number_days)

    ##########################################################################
    
    
    ################################## CASE 1 ########################################
    interpolated_data_folder = "data/interpolated_spatial_data"
    has_interpolated_data = any(
        file.startswith("stryn_") and file.endswith(".json")
        for file in os.listdir(interpolated_data_folder)
    )
##
    if not has_interpolated_data:
        # Generate the grid using IDW
        reformat_data(points, prec_values, temp_values, disch_values, number_days=number_days, file_paths=stryn_file_paths,evap_values=evap_values, rad_values=None, hyd_values=None, relHum_values=None, wind_values=None, extent=extent, spacing=spacing)
    df_prec    = pd.read_json(stryn_file_paths[0], orient='index')
    df_temp    = pd.read_json(stryn_file_paths[1], orient='index')
    df_disch   = pd.read_json(stryn_file_paths[2], orient='index')
    #df_rad     = pd.read_json(stryn_file_paths[3], orient='index')
    #df_relHum  = pd.read_json(stryn_file_paths[4], orient='index')
    #df_wind    = pd.read_json(stryn_file_paths[5], orient='index')
    df_evap    = pd.read_json(stryn_file_paths[6], orient='index')

    
   #gaula_has_interpolated_data = any(
   #    file.startswith("gaula_") and file.endswith(".json")
   #    for file in os.listdir(interpolated_data_folder)
   #)
   #if not gaula_has_interpolated_data:
   #    # Generate the grid using IDW for Gaula catchment
   #    reformat_data(points, prec_values, temp_values, disch_values, number_days=number_days, file_paths=gaula_file_paths,evap_values=None, rad_values=rad_values, hyd_values=hyd_values, relHum_values=relHum_values, wind_values=wind_values, extent=extent, spacing=spacing)
#
   #df_prec     = pd.read_json(gaula_file_paths[0], orient='index')
   #df_temp     = pd.read_json(gaula_file_paths[1], orient='index')
   #df_disch    = pd.read_json(gaula_file_paths[2], orient='index')
   #df_rad     = pd.read_json(gaula_file_paths[3], orient='index')
   #df_relHum  = pd.read_json(gaula_file_paths[4], orient='index')
   #df_wind    = pd.read_json(gaula_file_paths[5], orient='index')
   #df_hyd     = pd.read_json(gaula_file_paths[6], orient='index')

    
    ################################## CASE 2 ########################################

    # Reading the insitu data
    #df_disch_insitu = pd.read_json(hyd_station["filepath"], orient='index')
    #df_disch_insitu.columns = ["discharge1", "discharge2"]
#
    ###Satellite data
    #df_sat_prec     = pd.read_json(resized_file_paths[0], orient='index')
    #df_sat_temp     = pd.read_json(resized_file_paths[1], orient='index')
    #df_sat_snow     = pd.read_json(sat_file_paths[2], orient='index')


    # Defining the model parameters and valid data
    
    ################################## CASE 1 ########################################

    #Flat data

    #prec_data   = np.array(df_prec['flat_prec'].tolist())
    #temp_data   = np.array(df_temp['flat_temp'].tolist())
    ## Add one column of zeros to evap_data
    #evap_data   = np.array(df_evap['flat_evap'].tolist())
    #if evap_data.ndim == 2:
    #    evap_data = np.hstack([evap_data, np.zeros((evap_data.shape[0], 1))])
#
    #prec_data   = prec_data.reshape(-1, 1, 9)  # Reshape to (num_samples, height, width)
    #temp_data   = temp_data.reshape(-1, 1, 9)  # Reshape to (num_samples, height, width)
    #evap_data   = evap_data.reshape(-1, 1, 9)  # Reshape to (num_samples, height, width)
#
#
    #y_data      = np.array(df_disch['discharge'].tolist(), dtype=float)

    ## Assuming x_data and y_data are your input and target data
    ## In-situ data   
    prec_data   = np.array(df_prec['interpolated_prec'].tolist(), dtype=float)
    temp_data   = np.array(df_temp['interpolated_temp'].tolist(), dtype=float)
    y_data      = np.array(df_disch['discharge'].tolist(), dtype=float)

    # Switching
    evap_data   = np.array(df_evap['interpolated_evap'].tolist(), dtype=float)    
    #hyd_data   = np.array(df_hyd['interpolated_hyd'].tolist(), dtype=float)
    #rad_data    = np.array(df_rad['interpolated_rad'].tolist(), dtype=float)
    #relHum_data = np.array(df_relHum['interpolated_relHum'].tolist(), dtype=float)
    #wind_data   = np.array(df_wind['interpolated_wind'].tolist(), dtype=float)
    
    # How many nan values are in the data?
    #print(f"Number of NaN values in precipitation data: {np.isnan(prec_data).sum()}")
    #print(f"Number of NaN values in temperature data: {np.isnan(temp_data).sum()}")
    #print(f"Number of NaN values in evaporation data: {np.isnan(evap_data).sum()}")
    #print(f"Number of NaN values in hydrological data: {np.isnan(hyd_data).sum()}")
    #print(f"Number of NaN values in radiation data: {np.isnan(rad_data).sum()}")
    #print(f"Number of NaN values in wind data: {np.isnan(wind_data).sum()}")
    #print(f"Number of NaN values in relative humidity data: {np.isnan(relHum_data).sum()}")
    #print(f"Number of NaN values in discharge data: {np.isnan(y_data).sum()}")



    ## Mask rows where any feature in y_data is equal to -99.0 and the sequence length isn't possible
    # In-situ data

    seq_length = 14
    valid_indices = filter_valid_indices(y_data, seq_length) 
    prec_data = prec_data[valid_indices]
    temp_data = temp_data[valid_indices]
    y_data = y_data[valid_indices]

    
    # Switching
    evap_data = evap_data[valid_indices]
    Stryn_HBV = Stryn_HBV[valid_indices]
    Stryn_HBV_actuals = Stryn_HBV_actuals[valid_indices]
    Stryn_dates = Stryn_dates[valid_indices]
    
    #rad_data        = rad_data[valid_indices]
    #hyd_data        = hyd_data[valid_indices]
    #wind_data       = wind_data[valid_indices]
    #relHum_data     = relHum_data[valid_indices]
    #Gaula_RAVEN     = Gaula_RAVEN[valid_indices]
    #Gaula_actuals   = Gaula_actuals[valid_indices]
    #Gaula_dates     = Gaula_dates[valid_indices]



    # Combining the data for training
    num_samples, height, width = prec_data.shape
    prec_data_reshaped = prec_data.reshape(num_samples, -1)  # Shape: (num_samples, height * width)
    temp_data_reshaped = temp_data.reshape(num_samples, -1)
    
    #Switching
    evap_data_reshaped = evap_data.reshape(num_samples, -1)
    
    #hyd_data_reshaped = hyd_data.reshape(num_samples, -1)
    #rad_data_reshaped = rad_data.reshape(num_samples, -1)
    #wind_data_reshaped = wind_data.reshape(num_samples, -1)
    #relHum_data_reshaped = relHum_data.reshape(num_samples, -1)

    # Remove rows with NaN values in the reshaped data
    prec_data_reshaped = np.nan_to_num(prec_data_reshaped, nan=0.0)  # Replace NaN values with 0
    temp_data_reshaped = np.nan_to_num(temp_data_reshaped, nan=0.0)
    
    # Swtching
    evap_data_reshaped = np.nan_to_num(evap_data_reshaped, nan=0.0)

    #hyd_data_reshaped = np.nan_to_num(hyd_data_reshaped, nan=0.0)
    #rad_data_reshaped = np.nan_to_num(rad_data_reshaped, nan=0.0)
    #wind_data_reshaped = np.nan_to_num(wind_data_reshaped, nan=0.0)
    #relHum_data_reshaped = np.nan_to_num(relHum_data_reshaped, nan=0.0)
    
    # Print how many rows are zero in each of the reshaped data)

    # Stack the features along the last axis
    combined_data = np.stack([prec_data_reshaped, temp_data_reshaped,evap_data_reshaped], axis=-1)  
    combined_data = combined_data.reshape(num_samples, -1)  # Flatten spatial dimensions for normalization

    ################################## CASE 2 ########################################
    # Satellite data
    #sat_prec_data   = np.array(df_sat_prec.values, dtype=float).reshape(-1, 5, 16)  # Reshape to (num_samples, height, width)
    #sat_temp_data   = np.array(df_sat_temp.values, dtype=float).reshape(-1, 5, 16)  # Reshape to (num_samples, height, width)
    #sat_snow_data   = np.array(df_sat_snow.values, dtype=float).reshape(-1, 5, 16)  # Reshape to (num_samples, height, width)
#
    #resized_prec = []
    #resized_temp = []
    #resized_snow = []
    #
    #for i in range(len(sat_prec_data)):
    #    # Reshape the values in the row into the image
    #    sat_image_prec = sat_prec_data[i]
    #    sat_image_temp = sat_temp_data[i]
    #    sat_image_snow = sat_snow_data[i]
    #    # Resize the image to the desired size
    #    new_width, new_height = 5, 5
    #    arr_prec = cv2.resize(sat_image_prec, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    #    arr_temp = cv2.resize(sat_image_temp, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    #    arr_snow = cv2.resize(sat_image_snow, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    #    resized_prec.append(arr_prec.flatten())
    #    resized_temp.append(arr_temp.flatten())
    #    resized_snow.append(arr_snow.flatten())
    #
    #sat_prec_data = np.array(resized_prec).reshape(-1, 5, 5)  # Reshape to (num_samples, height, width)
    #sat_temp_data = np.array(resized_temp).reshape(-1, 5, 5)  # Reshape to (num_samples, height, width)
    #sat_snow_data = np.array(resized_snow).reshape(-1, 5, 5)  # Reshape to (num_samples, height, width)
#
    #height = 5
    #width  = 5
#
    #y_data_sat = np.column_stack([df_disch_insitu["discharge1"].tolist(), df_disch_insitu["discharge2"].tolist()])
    #                                                                 
    #sat_seq_length = 36
    #valid_indices  = filter_valid_indices(y_data_sat, sat_seq_length) 
    #sat_prec_data  = sat_prec_data[valid_indices]
    #sat_temp_data  = sat_temp_data[valid_indices]
    #sat_snow_data  = sat_snow_data[valid_indices]
    #y_data_sat     = y_data_sat[valid_indices]
##
    ### Flatten spatial dimensions (height, width) into a single feature vector
    #num_samples, height, width = sat_prec_data.shape
    #sat_prec_data_reshaped = sat_prec_data.reshape(num_samples, -1)  # Shape: (num_samples, height * width)
    #sat_temp_data_reshaped = sat_temp_data.reshape(num_samples, -1)
    #sat_snow_data_reshaped = sat_snow_data.reshape(num_samples, -1)
#
    ### Remove nan values from the reshaped data
    #sat_prec_data_reshaped = np.nan_to_num(sat_prec_data_reshaped, nan=0.0) 
    #sat_temp_data_reshaped = np.nan_to_num(sat_temp_data_reshaped, nan=0.0)
    #sat_snow_data_reshaped = np.nan_to_num(sat_snow_data_reshaped, nan=0.0)
    #
    ## Stack the features along the last axis
    #sat_combined_data = np.stack([sat_prec_data_reshaped, sat_temp_data_reshaped, sat_snow_data_reshaped], axis=-1)  
    #sat_combined_data = sat_combined_data.reshape(num_samples, -1)  # Flatten spatial dimensions for normalization

    # Define the model parameters
    num_layers      = 4         # Increase the number of LSTM layers
    dropout         = 0.4       # Adjust the dropout rate
    batch_size      = 2**6      # 32
    input_channels  = 3         # Number of input channels 
    hidden_size     = 2**8      # 254
    output_size     = 2         # Number of target features
    height          = height
    width           = width
    ################################## CASE 1 ########################################
    seq_length      = seq_length                    # days (# Because we're removing some days)
    num_samples     = combined_data.shape[0]        # Number of samples
    input_size      = (num_samples, height, width)  # Flatten spatial dimensions for normalization
    ################################## CASE 2 ########################################
    #seq_length      = sat_seq_length # hours
    #num_samples     = sat_combined_data.shape[0]  # Number of samples
    #input_size      = (num_samples, height, width)  # Flatten spatial dimensions for normalization

    # Define the model
    model_LSTM         = LSTM(input_channels, height, width, hidden_size, output_size, num_layers=num_layers, dropout = dropout)
    model_BiLSTM       = BiLSTM(input_channels, height, width, hidden_size, output_size, num_layers=num_layers, dropout = dropout)

    ################################## CASE 1 ########################################
    train_dataloader, val_dataloader, test_dataloader, scaler_y  = preprocess_data(combined_data, y_data, input_size, seq_length, batch_size, num_samples, channels = input_channels)
    
    h_dataloader, h_val_dataloader, h_test_dataloader, scaler_y_hybrid = preprocess_data(combined_data, y_data, input_size, seq_length, batch_size, num_samples, channels = input_channels)
    
    
    ################################## CASE 2 ########################################
    #Satellite data
    #train_dataloader, val_dataloader, test_dataloader, scaler_y = preprocess_data(sat_combined_data, y_data_sat, input_size, sat_seq_length, batch_size, num_samples, channels = input_channels)
    #
    #hybrid_dataloader, hybrid_val_dataloader, hybrid_test_dataloader, scaler_y_hybrid = preprocess_data(sat_combined_data, y_data_sat, input_size, sat_seq_length, batch_size, num_samples, channels = input_channels)
    
    
    ################################## CASE 1 ########################################
    # Train the model
    #model_LSTM     = train_model(model_LSTM, train_dataloader, val_dataloader, 'LSTM_stryn_historical', scaler_y)
    #model_BiLSTM   = train_model(model_BiLSTM, h_dataloader, h_val_dataloader, 'Hybrid_stryn_historical', scaler_y_hybrid)

    #model_LSTM    = train_model(model_LSTM, train_dataloader, val_dataloader, 'LSTM_flat_stryn_historical', scaler_y)
    #model_BiLSTM  = train_model(model_BiLSTM, h_dataloader, h_val_dataloader, 'BiLSTM_flat_stryn_historical', scaler_y_hybrid)

    #model_LSTM     = train_model(model_LSTM, train_dataloader, val_dataloader, 'LSTM_gaula_historical', scaler_y)
    #mmodel_BiLSTM   = train_model(model_BiLSTM, h_dataloader, h_val_dataloader, 'BiLSTM_gaula_historical', scaler_y_hybrid)
    
    # Load the saved model
    model_LSTM.load_state_dict(torch.load('model/save/LSTM_stryn_historical.pth'))
    model_BiLSTM.load_state_dict(torch.load('model/save/Hybrid_stryn_historical.pth'))

    #model_LSTM.load_state_dict(torch.load('model/save/LSTM_flat_stryn_historical.pth'))
    #model_BiLSTM.load_state_dict(torch.load('model/save/BiLSTM_flat_stryn_historical.pth'))


    #model_LSTM.load_state_dict(torch.load('model/save/LSTM_gaula_historical.pth'))
    #model_BiLSTM.load_state_dict(torch.load('model/save/BiLSTM_gaula_historical.pth'))

    len_train = len(train_dataloader.dataset)
    len_val   = len(val_dataloader.dataset)
    len_test  = len(test_dataloader.dataset)

    #Gaula_RAVEN_test = Gaula_RAVEN[int(len_train + len_val):]       # Limit to the last 15% of the data for testing
    #Gaula_actuals_test = Gaula_actuals[int(len_train + len_val):]   # Limit to the last 15% of the data for testing 
    #Gaula_dates_test = Gaula_dates[int(len_train + len_val):]       # Adjust dates to match the test data length

    Stryn_HBV_test = Stryn_HBV[int(len_train + len_val):]       # Limit to the last 15% of the data for testing
    Stryn_HBV_actuals_test = Stryn_HBV_actuals[int(len_train + len_val):]   # Limit to the last 15% of the data for testing
    Stryn_dates_test = Stryn_dates[int(len_train + len_val):]       # Adjust dates to match the test data length
    # Plot predictions vs actuals
    #plot_historical(model_LSTM, model_BiLSTM, test_dataloader, h_test_dataloader, Gaula_RAVEN_test, scaler_y, scaler_y_hybrid, Gaula_dates_test)
    #plot_historical_minifigures(model_LSTM, model_BiLSTM, test_dataloader, h_test_dataloader, Gaula_RAVEN, scaler_y, scaler_y_hybrid, Gaula_dates)
    #plot_historical(model_LSTM, model_BiLSTM, test_dataloader, h_test_dataloader, Stryn_HBV_test, scaler_y, scaler_y_hybrid, Stryn_dates_test)
    #plot_historical_minifigures(model_LSTM, model_BiLSTM, test_dataloader, h_test_dataloader, Stryn_HBV_test, scaler_y, scaler_y_hybrid, Stryn_dates_test)
    ################################## CASE 2 ########################################
    #model_LSTM      = train_model(model_LSTM, train_dataloader, val_dataloader, 'LSTM_stryn_satellite', scaler_y)
    #model_Hybrid    = train_model(model_Hybrid, hybrid_dataloader, hybrid_val_dataloader, 'Hybrid_stryn_satellite', scaler_y_hybrid)

    # Load the saved model
    # model_LSTM.load_state_dict(torch.load('model/save/LSTM_stryn_satellite.pth'))
    # model_Hybrid.load_state_dict(torch.load('model/save/Hybrid_stryn_satellite.pth'))

    # Plot predictions vs actuals
    #plot_satellite(model_LSTM, model_Hybrid, test_dataloader, hybrid_test_dataloader, scaler_y, scaler_y_hybrid)

    #exploratory(height, width)
    #print_test_results(model_BiLSTM, h_test_dataloader, scaler_y_hybrid, name="BiLSTM")
    #print_test_results(model_LSTM, test_dataloader, scaler_y, name="LSTM")
    #evaluation_HBV(Gaula_RAVEN, Gaula_actuals, name="RAVEN")
    #evaluation_HBV(Stryn_HBV, Stryn_HBV_actuals, name="HBV")
    #plot_study_area_v2()

    plot_climate_sensitivity(model_LSTM, model_BiLSTM, test_dataloader, h_test_dataloader, Stryn_HBV_test, Stryn_HBV_actuals_test, scaler_y, scaler_y_hybrid)
    #plot_climate_sensitivity(model_LSTM, model_BiLSTM, test_dataloader, h_test_dataloader, Gaula_RAVEN_test, Gaula_actuals_test, scaler_y, scaler_y_hybrid)
    
def plot_training_data(model,train_dataloader, scaler_y, name):
    """
    Plots the training data from the dataloader.
    
    Parameters:
    - train_dataloader: DataLoader containing the training data.
    - scaler_y: Scaler used to inverse transform the target values.
    - name: Name of the plot for title purposes.
    """
    model.eval()
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for batch_x, batch_y in train_dataloader:
            outputs = model(batch_x)
            predictions.append(outputs.cpu().numpy())
            actuals.append(batch_y.cpu().numpy())
    
    # Concatenate predictions and actuals along the first axis
    predictions = np.concatenate(predictions, axis=0)  # Shape: (num_samples, output_size)
    actuals = np.concatenate(actuals, axis=0)          # Shape: (num_samples, output_size)
    
    # Inverse transform the predictions and actuals to the original scale
    predictions = scaler_y.inverse_transform(predictions)
    actuals = scaler_y.inverse_transform(actuals)
    
    # Plot the predictions vs actuals
    plt.figure(figsize=(10, 6))
    
    if actuals.shape[1] > 1:  # If there are multiple target features
        for i in range(min(6, actuals.shape[1])):  # Plot up to 6 features
            plt.plot(actuals[:, i], label=f'Actual Data (Feature {i + 1})')
            plt.plot(predictions[:, i], label=f'Predicted Data (Feature {i + 1})', linestyle='--')
    else:
        plt.plot(actuals[:, 0], label='Actual Data (Feature 1)')
        plt.plot(predictions[:, 0], label='Predicted Data (Feature 1)', linestyle='--')

    plt.xlabel('Time Step [hours]')
    plt.ylabel('Value')
    plt.title(name + ': Training Data Predictions vs Actual Data')
    plt.legend()
    plt.show()

def plot_historical(model, hybrid, dataloader, h_dataloader, HBV, scaler_y, scaler_y_h, dates):
    dates = pd.to_datetime(dates, format="%Y-%m-%d")  # Convert dates to datetime objects
    model.eval()
    hybrid.eval()
    predictions = []
    predictions_h = []
    actuals = []
    
    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            outputs = model(batch_x)
            predictions.append(outputs.cpu().numpy())
            actuals.append(batch_y.cpu().numpy())
    
    # Concatenate predictions and actuals along the first axis
    predictions = np.concatenate(predictions, axis=0)  # Shape: (num_samples, output_size)
    actuals = np.concatenate(actuals, axis=0)          # Shape: (num_samples, output_size)
    
    with torch.no_grad():
        for batch_x, batch_y in h_dataloader:
            outputs_h = hybrid(batch_x)
            predictions_h.append(outputs_h.cpu().numpy())
    
    predictions_h = np.concatenate(predictions_h, axis=0)  # Shape: (num_samples, output_size)

    # Ensure predictions have the same shape as the scaler's expected input
    if predictions.shape[1] != scaler_y.min_.shape[0]:
        raise ValueError(f"Predictions shape {predictions.shape} does not match scaler's expected shape {scaler_y.min_.shape}")

    # Inverse transform the predictions and actuals to the original scale
    predictions = scaler_y.inverse_transform(predictions)
    actuals = scaler_y.inverse_transform(actuals)
    
    predictions_h = scaler_y_h.inverse_transform(predictions_h)

    min_len = min(len(dates), actuals.shape[0], predictions.shape[0], predictions_h.shape[0], HBV.shape[0])
    dates = dates[-min_len:]
    HBV = HBV[-min_len:]

    # 4. Plot (for feature 0 as example)
    plt.figure(figsize=(10, 6))
    plt.plot(dates, actuals[:, 0], label='Streamflow', color='blue')
    plt.plot(dates, predictions[:, 0], label='LSTM', color="green", linestyle='dotted')
    plt.plot(dates, predictions_h[:, 0], label='BiLSTM', color='purple', linestyle='dashed')
    plt.plot(dates, HBV[:, 0], label='RAVEN', color='orange', linestyle='dashdot')

    plt.plot(dates, actuals[:, 1], color='blue')
    plt.plot(dates, predictions[:, 1], color="green", linestyle='dotted')
    plt.plot(dates, predictions_h[:, 1], color='purple', linestyle='dashed')
    plt.plot(dates, HBV[:, 1], color='orange', linestyle='dashdot')

    #plt.plot(dates, actuals[:, 2], color='blue')
    #plt.plot(dates, predictions[:, 2], color="green", linestyle='dotted')
    #plt.plot(dates, predictions_h[:, 2], color='purple', linestyle='dashed')
    #plt.plot(dates, HBV[:, 2], color='orange', linestyle='dashdot')

    
    #plt.figure(figsize=(10, 6))
    #plt.plot(actuals[:, 1], label=f'Streamflow', color='blue')
    #plt.plot(predictions[:, 1], label=f'LSTM', color="green", linestyle='dotted')
    #plt.plot(predictions_h[:, 1], label=f'BiLSTM', color='purple', linestyle='dashed')  # Plot Hybrid model predictions
    #plt.plot(HBV[:, 1], label=f'HBV', color='orange', linestyle='dashdot')  # Plot HBV model predictions
    
    #std_deviation, h1, h2, h3 = HMMR_3(actuals, predictions)

    std_deviation, h1, h2 = HMMR_2(actuals, predictions_h)
    # Plot uncertainty (shaded area)
    plt.fill_between(dates,
                    actuals[:,0] - std_deviation[0, h1],
                    actuals[:,0] + std_deviation[0, h1],
                    color='blue', alpha=0.2, label='Standard Deviation')
    plt.fill_between(dates,
                actuals[:,1] - std_deviation[1, h2],
                actuals[:,1] + std_deviation[1, h2],
                color='blue', alpha=0.2)
    #plt.fill_between(dates,
    #            actuals[:,2] - std_deviation[2, h3],
    #            actuals[:,2] + std_deviation[2, h3],
    #            color='blue', alpha=0.2)
    
    plt.xlabel('Time Step [Days]')
    plt.ylabel('Streamflow [m³/s]')
    plt.xticks(rotation=45)
    plt.title('Gaula: Predictions with Uncertainty and RAVEN Comparison')
    plt.legend()
    plt.show()
    return

def plot_historical_minifigures(model, hybrid, dataloader, h_dataloader, HBV, scaler_y, scaler_y_h, dates):
    model.eval()
    hybrid.eval()
    predictions = []
    predictions_h = []
    actuals = []
    
    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            outputs = model(batch_x)
            predictions.append(outputs.cpu().numpy())
            actuals.append(batch_y.cpu().numpy())
    
    # Concatenate predictions and actuals along the first axis
    predictions = np.concatenate(predictions, axis=0)  # Shape: (num_samples, output_size)
    actuals = np.concatenate(actuals, axis=0)          # Shape: (num_samples, output_size)
    
    with torch.no_grad():
        for batch_x, batch_y in h_dataloader:
            outputs_h = hybrid(batch_x)
            predictions_h.append(outputs_h.cpu().numpy())
    
    predictions_h = np.concatenate(predictions_h, axis=0)  # Shape: (num_samples, output_size)

    # Ensure predictions have the same shape as the scaler's expected input
    if predictions.shape[1] != scaler_y.min_.shape[0]:
        raise ValueError(f"Predictions shape {predictions.shape} does not match scaler's expected shape {scaler_y.min_.shape}")

    # Inverse transform the predictions and actuals to the original scale
    predictions = scaler_y.inverse_transform(predictions)
    actuals = scaler_y.inverse_transform(actuals)
    
    predictions_h = scaler_y_h.inverse_transform(predictions_h)
    min_len = min(len(dates), actuals.shape[0], predictions.shape[0], predictions_h.shape[0], HBV.shape[0])
    dates = dates[-min_len:]
    HBV = HBV[-min_len:]
    # Select years of interest
    years = [2018, 2019, 2020, 2021, 2022]
    #years = [2004, 2005]
    n_years = len(years)
    fig, axes = plt.subplots(n_years, 1, figsize=(12, 3 * n_years), sharex=False)
    years_arr = dates.dt.year if hasattr(dates, 'dt') else dates.year
    
    for i, year in enumerate(years):
        ax = axes[i]
        mask = years_arr == year
        if not np.any(mask):
            continue  # Skip if no data for this year

        ax.plot(dates[mask], actuals[mask, 0], color='blue')
        ax.plot(dates[mask], predictions[mask, 0], color="green", linestyle='dotted')
        ax.plot(dates[mask], predictions_h[mask, 0], color='purple', linestyle='dashed')
        ax.plot(dates[mask], HBV[mask, 0], color='orange', linestyle='dashdot')

        ax.plot(dates[mask], actuals[mask, 1], color='blue')
        ax.plot(dates[mask], predictions[mask, 1], color="green", linestyle='dotted')
        ax.plot(dates[mask], predictions_h[mask, 1], color='purple', linestyle='dashed')
        ax.plot(dates[mask], HBV[mask, 1], color='orange', linestyle='dashdot')
        
        # Optional: plot uncertainty if available
        std_deviation, h1, h2 = HMMR_2(actuals[mask], predictions_h[mask])
        ax.fill_between(
            dates[mask],
            actuals[mask, 1] - std_deviation[1, h2],
            actuals[mask, 1] + std_deviation[1, h2],
            color='blue', alpha=0.2
        )

        ax.set_ylabel('Streamflow [m³/s]')
        ax.set_title(f'Gaula: {year}')
        ax.tick_params(axis='x')

    plt.xlabel('Date')
    plt.tight_layout(pad=2.0)
    plt.subplots_adjust(hspace=1.0) 
    plt.show()
    return

def plot_climate_sensitivity(model, hybrid, dataloader, h_dataloader, HBV, HBV_actual, scaler_y, scaler_y_h):
    """ 
        In this function we will be plotting and calculating the models accuracy at 
        predicting the lower and upper 10 th percentile of the streamflow data.
    """
    print("Plotting climate sensitivity...")
    model.eval()
    hybrid.eval()
    predictions_h = []
    predictions = []
    actuals = []
    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            output = model(batch_x)
            # Store the means and variances
            predictions.append(output.cpu().numpy())
            actuals.append(batch_y.cpu().numpy())

    with torch.no_grad():
        for batch_x, batch_y in h_dataloader:
            outputs_h = hybrid(batch_x)
            predictions_h.append(outputs_h.cpu().numpy())
    
    # Concatenate predictions and actuals along the first axis
    predictions     = np.concatenate(predictions, axis=0)     # Shape: (num_samples, output_size)
    predictions_h   = np.concatenate(predictions_h, axis=0)   # Shape: (num_samples, output_size)
    actuals         = np.concatenate(actuals, axis=0)         # Shape: (num_samples, output_size)

    # Inverse transform the predictions and actuals to the original scale
    predictions     = scaler_y.inverse_transform(predictions)
    predictions_h   = scaler_y_h.inverse_transform(predictions_h)
    actuals         = scaler_y.inverse_transform(actuals)    

    min_len = min(actuals.shape[0], predictions.shape[0], predictions_h.shape[0], HBV.shape[0])
    HBV = HBV[-min_len:]
    HBV_actual = HBV_actual[-min_len:]

    # Calculate the lower and upper 10th percentiles of the actuals (as scalars)
    lower_10th_percentile_0 = np.percentile(actuals[:, 0], 10)
    lower_10th_percentile_1 = np.percentile(actuals[:, 1], 10)
    #lower_10th_percentile_2 = np.percentile(actuals[:, 2], 10)
    upper_10th_percentile_0 = np.percentile(actuals[:, 0], 90)
    upper_10th_percentile_1 = np.percentile(actuals[:, 1], 90)
    #upper_10th_percentile_2 = np.percentile(actuals[:, 2], 90)

    # Filter predictions and actuals based on the percentiles
    lower_10th_actuals = np.full_like(actuals, np.nan)
    mask_0 = actuals[:, 0] <= lower_10th_percentile_0
    mask_1 = actuals[:, 1] <= lower_10th_percentile_1
    #mask_2 = actuals[:, 2] <= lower_10th_percentile_2
    lower_10th_actuals[mask_0, 0] = actuals[mask_0, 0]
    lower_10th_actuals[mask_1, 1] = actuals[mask_1, 1]
    #lower_10th_actuals[mask_2, 2] = actuals[mask_2, 2]

    upper_10th_actuals = np.full_like(actuals, np.nan)
    mask_0 = actuals[:, 0] >= upper_10th_percentile_0
    mask_1 = actuals[:, 1] >= upper_10th_percentile_1
    #mask_2 = actuals[:, 2] >= upper_10th_percentile_2
    upper_10th_actuals[mask_0, 0] = actuals[mask_0, 0]
    upper_10th_actuals[mask_1, 1] = actuals[mask_1, 1]
    #upper_10th_actuals[mask_2, 2] = actuals[mask_2, 2]

    lower_10th_predictions = np.full_like(actuals, np.nan)
    mask_0 = actuals[:, 0] <= lower_10th_percentile_0
    mask_1 = actuals[:, 1] <= lower_10th_percentile_1
    #mask_2 = actuals[:, 2] <= lower_10th_percentile_2
    lower_10th_predictions[mask_0, 0] = predictions[mask_0, 0]
    lower_10th_predictions[mask_1, 1] = predictions[mask_1, 1]
    #lower_10th_predictions[mask_2, 2] = predictions[mask_2, 2]

    upper_10th_predictions = np.full_like(actuals, np.nan)
    mask_0 = actuals[:, 0] >= upper_10th_percentile_0
    mask_1 = actuals[:, 1] >= upper_10th_percentile_1
    #mask_2 = actuals[:, 2] >= upper_10th_percentile_2
    upper_10th_predictions[mask_0, 0] = predictions[mask_0, 0]
    upper_10th_predictions[mask_1, 1] = predictions[mask_1, 1]
    #upper_10th_predictions[mask_2, 2] = predictions[mask_2, 2]

    lower_10th_predictions_h = np.full_like(actuals, np.nan)
    mask_0 = actuals[:, 0] <= lower_10th_percentile_0
    mask_1 = actuals[:, 1] <= lower_10th_percentile_1
    #mask_2 = actuals[:, 2] <= lower_10th_percentile_2
    lower_10th_predictions_h[mask_0, 0] = predictions_h[mask_0, 0]
    lower_10th_predictions_h[mask_1, 1] = predictions_h[mask_1, 1]
    #lower_10th_predictions_h[mask_2, 2] = predictions_h[mask_2, 2]
    

    upper_10th_predictions_h = np.full_like(actuals, np.nan)
    mask_0 = actuals[:, 0] >= upper_10th_percentile_0
    mask_1 = actuals[:, 1] >= upper_10th_percentile_1
    #mask_2 = actuals[:, 2] >= upper_10th_percentile_2
    upper_10th_predictions_h[mask_0, 0] = predictions_h[mask_0, 0]
    upper_10th_predictions_h[mask_1, 1] = predictions_h[mask_1, 1]
    #upper_10th_predictions_h[mask_2, 2] = predictions_h[mask_2, 2]

    # HBV model predictions
    lower_10th_HBV = np.full_like(actuals, np.nan)
    mask_0 = actuals[:, 0] <= lower_10th_percentile_0
    mask_1 = actuals[:, 1] <= lower_10th_percentile_1
    #mask_2 = actuals[:, 2] <= lower_10th_percentile_2
    lower_10th_HBV[mask_0, 0] = HBV[mask_0, 0]
    lower_10th_HBV[mask_1, 1] = HBV[mask_1, 1]
    #lower_10th_HBV[mask_2, 2] = HBV[mask_2, 2]

    upper_10th_HBV = np.full_like(actuals, np.nan)
    mask_0 = actuals[:, 0] >= upper_10th_percentile_0
    mask_1 = actuals[:, 1] >= upper_10th_percentile_1
    #mask_2 = actuals[:, 2] >= upper_10th_percentile_2
    upper_10th_HBV[mask_0, 0] = HBV[mask_0, 0]
    upper_10th_HBV[mask_1, 1] = HBV[mask_1, 1]
    #upper_10th_HBV[mask_2, 2] = HBV[mask_2, 2]

    lower_10th_HBV_actuals = np.full_like(actuals, np.nan)
    mask_0 = actuals[:, 0] <= lower_10th_percentile_0
    mask_1 = actuals[:, 1] <= lower_10th_percentile_1
    #mask_2 = actuals[:, 2] <= lower_10th_percentile_2
    lower_10th_HBV_actuals[mask_0, 0] = HBV_actual[mask_0, 0]
    lower_10th_HBV_actuals[mask_1, 1] = HBV_actual[mask_1, 1]
    #lower_10th_HBV_actuals[mask_2, 2] = HBV_actual[mask_2, 2]

    upper_10th_HBV_actuals = np.full_like(actuals, np.nan)
    mask_0 = actuals[:, 0] >= upper_10th_percentile_0
    mask_1 = actuals[:, 1] >= upper_10th_percentile_1
    #mask_2 = actuals[:, 2] >= upper_10th_percentile_2
    upper_10th_HBV_actuals[mask_0, 0] = HBV_actual[mask_0, 0]
    upper_10th_HBV_actuals[mask_1, 1] = HBV_actual[mask_1, 1]
    #upper_10th_HBV_actuals[mask_2, 2] = HBV_actual[mask_2, 2]

    evaluation_HBV(lower_10th_predictions, lower_10th_actuals, name="Lower 10th Percentile (LSTM)")
    evaluation_HBV(upper_10th_predictions, upper_10th_actuals, name="Upper 10th Percentile (LSTM)")

    evaluation_HBV(lower_10th_predictions_h, lower_10th_actuals, name="Lower 10th Percentile (BiLSTM)")
    evaluation_HBV(upper_10th_predictions_h, upper_10th_actuals, name="Upper 10th Percentile (BiLSTM)")

    evaluation_HBV(lower_10th_HBV, lower_10th_actuals, name="Lower 10th Percentile (HBV)")
    evaluation_HBV(upper_10th_HBV, upper_10th_actuals, name="Upper 10th Percentile (HBV)")
    return

def plot_satellite(model, hybrid, dataloader, h_dataloader, scaler_y, scaler_y_h):

    model.eval()
    hybrid.eval()
    predictions_h = []
    predictions = []
    actuals = []
    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            output = model(batch_x)
            # Store the means and variances
            predictions.append(output.cpu().numpy())
            actuals.append(batch_y.cpu().numpy())
    # Concatenate predictions along the first axis
    predictions = np.concatenate(predictions, axis=0)  # (7, 7, output_size) 
    predictions = scaler_y.inverse_transform(predictions)
    actuals = np.concatenate(actuals, axis=0)          # Shape: (num_samples, output_size)
    actuals = scaler_y.inverse_transform(actuals)

    with torch.no_grad():
        for batch_x, batch_y in h_dataloader:
            outputs_h = hybrid(batch_x)
            predictions_h.append(outputs_h.cpu().numpy())
    
    predictions_h = np.concatenate(predictions_h, axis=0)  # Shape: (num_samples, output_size)
    predictions_h = scaler_y_h.inverse_transform(predictions_h)
    
    plt.figure(figsize=(10, 6))
    plt.plot(actuals[:, 1], label=f'Streamflow', color='blue')
    plt.plot(predictions[:, 1], label=f'LSTM', color="green", linestyle='dotted')
    plt.plot(predictions_h[:, 1], label=f'Hybrid', color='purple', linestyle='dashed')  # Plot Hybrid model predictions
    
    std_deviation, h1, h2 = HMMR_2(actuals, predictions)
    # Plot uncertainty (shaded area)
    x = np.arange(len(predictions))  # Assuming predictions is a 1D array for simplicity
    plt.fill_between(x,
                     actuals[:, 1] - std_deviation[1, h2],
                     actuals[:, 1] + std_deviation[1, h2],
                     color='blue', alpha=0.2, label='Standard Deviation')
    
    plt.xlabel('Time Step [hours]')
    plt.ylabel('Streamflow [m³/s]')
    plt.title('Stryn - LSTM: Predictions with Uncertainty (Satellite Data)')
    plt.legend()
    plt.show()

def flood_hydrographs(rain, snow, q_data_1, q_data_2):

    x = np.arange(len(rain))  # Assuming x is the time steps
    fig, ax1 = plt.subplots(figsize=(12, 6))
    plt.title('High Precipitation and Flood Hydrographs')
    # Plot precipitation as a bar chart
    ax1.set_xlabel('Time Step [days]')
    ax1.set_ylabel('Precipitation', color='blue')
    ax1.hlines(0, xmin=x[0], xmax=x[-1], color='black', linewidth=0.5, linestyle='--')
    ax1.plot(x, rain, color='blue', label='Rain')
    ax1.fill_between(x, rain, color='blue', alpha=0.3)
    ax1.plot(x, snow, color='green', label='Snow')
    ax1.fill_between(x, snow, color='green', alpha=0.3)
    ax1.tick_params(axis='y', labelcolor='blue')

    ax2 = ax1.twinx()
    ax2.set_ylabel('Discharge', color='red')
    ax2.plot(x, q_data_1, color='red', label='Discharge 1')
    ax2.plot(x, q_data_2, color='red', label='Discharge 2')
    ax2.tick_params(axis='y', labelcolor='red')

    fig.tight_layout()
    plt.show()

def exploratory(height, width):
    met_files = sorted(glob.glob(os.path.join("data/satellite_data/", "MET_*.json")))

    temp_list = []
    prec_list = []
    timestamps = None

    for file_path in met_files:
        df = pd.read_json(file_path, orient='values')
        if timestamps is None:
            timestamps = df[0]
        temp_list.append(df[1].rename(file_path))
        prec_list.append(df[2].rename(file_path))

    # Combine all columns into a single DataFrame for each variable
    satellite_temp = pd.concat(temp_list, axis=1)
    satellite_prec = pd.concat(prec_list, axis=1)
    satellite_temp.index = pd.to_datetime(timestamps, unit='ms', utc=True)
    satellite_prec.index = pd.to_datetime(timestamps, unit='ms', utc=True)


    DWD_files = sorted(glob.glob(os.path.join("data/satellite_data/", "DWD_*.json")))
    snow_list = []
    timestamps = None
    for file_path in DWD_files:
        df = pd.read_json(file_path, orient='values')
        if timestamps is None:
            timestamps = df[0]
        snow_list.append(df[1].rename(file_path))

    # Combine all columns into a single DataFrame for each variable
    satellite_snow = pd.concat(snow_list, axis=1)
    satellite_snow.index = pd.to_datetime(timestamps, unit='ms', utc=True)    
    
    df_disch_insitu = pd.read_json("data\hourly_data\insitu_discharge.json", orient='index')
    #df_disch_insitu["insitu_discharge"] = df_disch_insitu.apply(lambda row: np.array(row.values), axis=1)
    #df_disch_insitu = df_disch_insitu.drop(df_disch_insitu.columns[[0, 1]], axis=1)
    df_disch_insitu.index = pd.to_datetime(df_disch_insitu.index)

    ### Inputs
    ## satellite_prec
    ## satellite_temp
#
    ## Outputs
    ## discharge
    intervall = int(0.7 * len(satellite_prec))  # 15% of the data length
    val_length = intervall + int(0.15 * len(satellite_prec))
    satellite_prec  = satellite_prec[intervall:val_length]
    satellite_temp  = satellite_temp[intervall:val_length]
    satellite_snow  = satellite_snow[intervall:val_length] 
    df_disch_insitu = df_disch_insitu[intervall:val_length]

    #Remove negative precipitation values
    satellite_prec[satellite_prec < 0] = 0

    #satellite_temp  = satellite_temp["02-02-2024 22:00:00":]
    #satellite_dew   = satellite_dew["02-02-2024 22:00:00":]  
    #df_disch_insitu = df_disch_insitu["02-02-2024 22:00:00":] 

    fig, ax1 = plt.subplots(figsize=(12, 6))
    # Plot temperature (as a line)
    ax1.plot(satellite_temp.mean(axis=1), label='Temperature', color='red')
    ax1.set_ylabel('Temperature (°C)', color='red')
    ax1.tick_params(axis='y', labelcolor='red')

   
    # Plot discharge (as a line)
    for col in df_disch_insitu.columns:
        ax1.plot(df_disch_insitu.index, df_disch_insitu[col], label=f'Discharge {col}', linewidth=1.5)
    ax1.set_ylabel('Temperature (°C) / Discharge (m³/s)')

    # Create a second y-axis for precipitation
    ax2 = ax1.twinx()
    ax2.fill_between(satellite_prec.index, satellite_prec.mean(axis=1), color='blue', alpha=0.3, label='Precipitation')
    ax2.set_ylabel('Precipitation (mm)', color='blue')
    ax2.tick_params(axis='y', labelcolor='blue')
    ax2.invert_yaxis()  # Invert precipitation axis

    from matplotlib.ticker import MaxNLocator
    ax3 = ax1.twinx()
    ax3.spines['right'].set_position(('outward', 60))  # Offset the third axis
    ax3.fill_between(satellite_snow.index, satellite_snow.mean(axis=1), color='green', alpha=0.3, label='Snow Depth')
    ax3.set_ylabel('Snow Depth (m)', color='green')
    ax3.tick_params(axis='y', labelcolor='green')
    ax3.invert_yaxis()
    ax3.yaxis.set_major_locator(MaxNLocator(integer=True))

    # Add legends
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    lines_3, labels_3 = ax3.get_legend_handles_labels()
    ax3.legend(lines_1 + lines_2 + lines_3, labels_1 + labels_2 + labels_3, loc='best', facecolor='white')
    
    plt.title('Precipitation, Temperature, Snow Depth and Discharge')
    plt.tight_layout()
    plt.show()

def print_test_results(model, test_dataloader, scaler_y, name="Model"):
    print(f"\n")
    print(f"Test Results for {name}:")
    print(f"Station\tNSE\tRMSE\tMAE\tMAPE\tR^2\tPearson\tKGE")
    feature_idx = [0, 1]
    #feature_idx = [0, 1, 2]
    model.eval()
    all_y_true = []
    all_y_pred = []
    with torch.no_grad():
        for batch_x, batch_y in test_dataloader:
            outputs = model(batch_x)
            y_true = scaler_y.inverse_transform(batch_y.detach().cpu().numpy())
            y_pred = scaler_y.inverse_transform(outputs.detach().cpu().numpy())
            all_y_true.append(y_true)
            all_y_pred.append(y_pred)
    all_y_true = np.concatenate(all_y_true, axis=0)
    all_y_pred = np.concatenate(all_y_pred, axis=0)
    for f in feature_idx:
        test_nse = NSE_formula(all_y_true[:, f], all_y_pred[:, f])
        test_rmse = RMSE_formula(all_y_true[:, f], all_y_pred[:, f])
        test_mae = MAE_formula(all_y_true[:, f], all_y_pred[:, f])
        test_mape = MAPE_formula(all_y_true[:, f], all_y_pred[:, f])
        test_r_squared = R_squared_formula(all_y_true[:, f], all_y_pred[:, f])
        test_kge = KGE_formula(all_y_true[:, f], all_y_pred[:, f])
        test_pearson = Pearson_formula(all_y_true[:, f], all_y_pred[:, f])
        print(f"{f}\t{test_nse:.4f}\t{test_rmse:.4f}\t{test_mae:.4f}\t{test_mape:.4f}\t{test_r_squared:.4f}\t{test_pearson:.4f}\t{test_kge:.4f}")
    print(f"\n")
    return

def evaluation_HBV(HBV, actual, name="Model"):
    """
    Evaluates the HBV model against actual data.
    
    Parameters:
    - HBV: The HBV model predictions.
    - actual: The actual observed data.
    
    Returns:
    - A dictionary containing evaluation metrics.
    """
    print(f"\n")
    print(f"Test Results for {name}:")
    print(f"Station\tNSE\tRMSE\tMAE\tMAPE\tR^2\tPearson\tKGE")
    feature_idx = [0,1] # for the second feature
    #feature_idx = [0, 1, 2]  # Adjusted to include all features for evaluation  
    for f in feature_idx:
        y_true = actual[:, f]
        y_pred = HBV[:, f]
        # Filter out nan/inf
        y_true, y_pred = nan_filter(y_true, y_pred)
        if len(y_true) == 0 or len(y_pred) == 0:
            print(f"{f}\tNo valid data for evaluation.")
            continue
        nse = NSE_formula(y_true, y_pred)
        rmse = RMSE_formula(y_true, y_pred)
        mae = MAE_formula(y_true, y_pred)
        mape = MAPE_formula(y_true, y_pred)
        r_squared = R_squared_formula(y_true, y_pred)
        kge = KGE_formula(y_true, y_pred)
        pearson = Pearson_formula(y_true, y_pred)

        print(f"{f}\t{nse:.4f}\t{rmse:.4f}\t{mae:.4f}\t{mape:.4f}\t{r_squared:.4f}\t{pearson:.4f}\t{kge:.4f}")
    print(f"\n")
    return

if __name__ == "__main__":
    main()