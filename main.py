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
from src.functions          import *
from src.UKMO_download      import *
from src.HydAPI_download    import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

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
    
    #Note: The level of accuracy the model is able to achieve defines how good it is
    able to predict the water flow as it is right now. In addition to capturing the
    current water flow, the model should then deliever the water flow for the next 
    hours and days with a displayed level of uncertainty. This uncertainty curve can
    be generated based on the historical accuracy of the model to predict the future trends. 
    
    """
    
    #Create a list of longitudes and latitudes for the catchment area

    # Pick of catchment
    chosenCatchment = 'Stryn' # 'Stryn' or 'Gaula'

    if chosenCatchment == 'Stryn':
        spacing  = 4000  # Spacing between grid points (e.g., 4000 m)
        number_days = 6*365 #  max 42 years (80-22)
        # Timeseries
        points, perc_values, temp_values, evap_values, disch_values, dates = catchment_styrn(number_days)

        # Area description 
        x_limit = (min(points.values(), key=lambda p: p[0])[0], max(points.values(), key=lambda p: p[0])[0])
        y_limit = (min(points.values(), key=lambda p: p[1])[1], max(points.values(), key=lambda p: p[1])[1])
        x_degrees   = ( 6.45811333,  7.33417667) # (6°27'29.208", 7°20'3.036")
        y_degrees   = (61.75932639, 62.06398778) #(61°45'33.575", 62°3'50.356")
        extent      = (x_limit[0], x_limit[1], y_limit[0], y_limit[1])  
        

        # Hydrological and Meterological data
        hyd_station = {
            "STNR": "88.15.0,88.11.0",
            "parameter": "1001",
            "filepath": "data/hourly_data/insitu_discharge.json"
        }

        prec_station = {
            "STNR": "98.4.0",
            "parameter": "9160",
            "filepath": "data/hourly_data/insitu_prec.json"
        }

        temp_station = {
            "STNR": "88.50.7,88.11.0,88.48.5,88.35.0,88.33.0,88.51.6,88.24.0,98.4.0,88.23.0,88.3.0,88.24.0",
            "parameter": "17",
            "filepath": "data/hourly_data/insitu_temp.json"
        }

        api_key = "bq5Ny6WGYkK5ySwwkqjCBQ=="

        disc_argv = [
            "-a", api_key,
            "-s", hyd_station["STNR"],
            "-p", hyd_station["parameter"],
            "-r", 60,
            "-t", "2023-05-17/2025-05-31"
        ]

        data_folder_path = "data/hourly_data"
        has_data = any(
            file.startswith("insitu_") and file.endswith(".json")
            for file in os.listdir(data_folder_path)
        )

        if not has_data:
            get_observations(disc_argv,hyd_station["STNR"],filepath=hyd_station["filepath"])
       
        # Satelite Forecast
        long_list, lat_list, width, height = create_grid(x_degrees, y_degrees, 4) # Create a grid of points

        #Create interpolated satellite data
        start_date = "2023-05-17"
        end_date   = "2025-05-31"
        data_folder_path = "data/interpolated_spatial_data"
        sat_file_paths = [
            'data/interpolated_spatial_data/sat_perc_spatial.json',
            'data/interpolated_spatial_data/sat_temp.json'
        ]
        
        
        # Check if the data folder contains any .json files for the given period
        has_data = any(
            file.startswith("sat_") and file.endswith(".json")
            for file in os.listdir(data_folder_path)
        )

        if not has_data:
            create_satellite_data(long_list, lat_list, start_date, end_date, width, height, extent, sat_file_paths)
        

    elif chosenCatchment == 'Gaula':
        spacing     = 8000  # Spacing between grid points (e.g., 4000 m)
        number_days = 6*365 # max 6 years (99-05)
        # Timeseries
        points_gaula, perc_values, temp_values, rad_values, hyd_values, relHum_values, wind_values, disch_values, dates = catchment_gaula(number_days)
        # Area description 
        x_limit = (min(points_gaula.values(), key=lambda p: p[0])[0], max(points_gaula.values(), key=lambda p: p[0])[0])
        y_limit = (min(points_gaula.values(), key=lambda p: p[1])[1], max(points_gaula.values(), key=lambda p: p[1])[1])
        x_degrees   = (0, 0) # (6°27'29.208", 7°20'3.036")
        y_degrees   = (0, 0) #(61°45'33.575", 62°3'50.356")
        extent      = (x_limit[0], x_limit[1], y_limit[0], y_limit[1])  
        

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


    # Check if the files exist or interpolate data
    #if chosenCatchment == 'Stryn' and any(not os.path.exists(file) for file in stryn_file_paths):
    #    # Generate the grid using IDW
    #    reformat_data_stryn(points, perc_values, temp_values, evap_values, disch_values, number_days, dates, stryn_file_paths, extent=extent, spacing=spacing, power=2)
    #elif chosenCatchment == 'Gaula' and any(not os.path.exists(file) for file in gaula_file_paths):
    #    # Generate the grid using IDW
    #    reformat_data_gaula(points_gaula, perc_values, temp_values, rad_values, hyd_values, relHum_values, wind_values, disch_values, number_days, dates, gaula_file_paths, extent=extent, spacing=spacing, power=2)
    

    # Reading the interpolated data
    if chosenCatchment == 'Stryn':
        #df_perc     = pd.read_json(perc_file_path, orient='values')
        #df_disch    = pd.read_json(disch_file_path, orient='values')
        #df_temp     = pd.read_json(temp_file_path, orient='values')
        #df_evap     = pd.read_json(evap_file_path, orient='values')
#
        #df_perc.columns     = ["Date", "interpolated_perc"]
        #df_disch.columns    = ["Date", "discharge"]
        #df_temp.columns     = ["Date", "interpolated_temp"]
        #df_evap.columns     = ["Date", "interpolated_evap"]   

        # Reading the insitu data
        df_disch_insitu = pd.read_json(hyd_station["filepath"], orient='index')
        # Convert all values to a numpy array (each row as an array)
        df_disch_insitu["insitu_discharge"] = df_disch_insitu.apply(lambda row: np.array(row.values), axis=1)
        df_disch_insitu = df_disch_insitu.drop(df_disch_insitu.columns[[0, 1]], axis=1)

        #Satellite data
        df_sat_prec     = pd.read_json(sat_file_paths[0], orient='index')
        df_sat_temp     = pd.read_json(sat_file_paths[1], orient='index')
        df_sat_prec.columns     = ["sat_precipitation"]
        df_sat_temp.columns     = ["sat_temperature"]  

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
        #perc_data   = np.array(df_perc['interpolated_perc'].tolist())
        #temp_data   = np.array(df_temp['interpolated_temp'].tolist())
        #evap_data   = np.array(df_evap['interpolated_evap'].tolist())
        #y_data      = np.array(df_disch['discharge'].tolist())
#
        ## Mask rows where any feature in y_data is equal to -99.0 and the sequence length isn't possible
        #seq_length = 7
        #valid_indices = filter_valid_indices(y_data, seq_length) 
        #perc_data = perc_data[valid_indices]
        #temp_data = temp_data[valid_indices]
        #evap_data = evap_data[valid_indices]
        #y_data = y_data[valid_indices]
#
        ## Extract the second column and keep it as 2D
        #y_data = y_data[:, [0,1]]  # Shape: (num_samples, 1)   


        #Satellite data
        sat_seq_length = 24 #h

        sat_prec_data   = np.array(df_sat_prec["sat_precipitation"].tolist())
        sat_prec_data   = sat_prec_data[:-23]
        sat_temp_data   = np.array(df_sat_temp["sat_temperature"].tolist())
        sat_temp_data   = sat_temp_data[:-23]

        y_data_sat      = np.array(df_disch_insitu["insitu_discharge"].tolist())
        #y_data_sat      = y_data_sat[:-1]
        
        valid_indices = filter_valid_indices(y_data_sat, sat_seq_length) 
        sat_prec_data = sat_prec_data[valid_indices]
        sat_temp_data = sat_temp_data[valid_indices]
        y_data_sat = y_data_sat[valid_indices]
        
        y_data_sat = y_data_sat[:, [0,1]]  # Shape: (num_samples, 1)

        num_layers      = 12  # Increase the number of LSTM layers
        dropout         = 0.4  # Adjust the dropout rate
        batch_size      = 2**6 # 64
        #seq_length      = seq_length # days/hours
        input_channels  = 2  # Number of input channels (precipitation, temperature, evaporation)
        hidden_size     = 256
        output_size     = 2  # Number of target features
        input_size      = (height,width)
        height          = height
        width           = width
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
        seq_length = 4
        valid_indices = filter_valid_indices(y_data, seq_length) 
        perc_data = perc_data[valid_indices]
        temp_data = temp_data[valid_indices]
        rad_data = rad_data[valid_indices]
        hyd_data = hyd_data[valid_indices]
        relHum_data = relHum_data[valid_indices]
        wind_data = wind_data[valid_indices]
        y_data = y_data[valid_indices]

        # Adjusting the output_size
        y_data = y_data[:, [0,1,2,3,4]]  # Shape: (num_samples, output_size)   

        #elev_data = reformat_elevation_map(elevation_tiff_path, perc_data_shape)
        num_layers      = 4  # Increase the number of LSTM layers
        dropout         = 0.4  # Adjust the dropout rate
        batch_size      = 2**5 # 64
        seq_length      = seq_length # days (# Because we're removing some days)
        input_channels  = 6  # Number of input channels (precipitation, temperature, evaporation)
        hidden_size     = 2**8
        output_size     = y_data.shape[1]  # Number of target features
        input_size      = perc_data.shape[1:3]
        height          = input_size[0]
        width           = input_size[1]

    # Define the model
    #model_LSTM       = LSTM(input_channels, height, width, hidden_size, output_size, num_layers=num_layers, dropout = dropout)

    # Combining the data for training
    if chosenCatchment == 'Stryn':
        ## Reshape perc_data and temp_data to 2D for normalization
        #num_samples, height, width = perc_data.shape
        #perc_data_reshaped = perc_data.reshape(num_samples, -1)  # Shape: (num_samples, height * width)
        #temp_data_reshaped = temp_data.reshape(num_samples, -1)
        #evap_data_reshaped = evap_data.reshape(num_samples, -1)
#
        ## Stack the features along the last axis
        #combined_data = np.stack([perc_data_reshaped, temp_data_reshaped, evap_data_reshaped], axis=-1)  # Shape: (num_samples, height * width, 3)
        #combined_data = combined_data.reshape(num_samples, -1)  # Flatten spatial dimensions for normalization
        #print("combined_data shape:", combined_data.shape)
        #Satellite data
        num_samples, height, width = sat_prec_data.shape
        sat_perc_data_reshaped = sat_prec_data.reshape(num_samples, -1)  # Shape: (num_samples, height * width)
        sat_temp_data_reshaped = sat_temp_data.reshape(num_samples, -1)

        # Stack the features along the last axis
        sat_combined_data = np.stack([sat_perc_data_reshaped, sat_temp_data_reshaped], axis=-1)  # Shape: (num_samples, height * width, 3)
        sat_combined_data = sat_combined_data.reshape(num_samples, -1)  # Flatten spatial dimensions for normalization

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
        #combined_data = np.stack([perc_data_reshaped, temp_data_reshaped], axis=-1)  # Shape: (num_samples, height * width, input_channels)
        combined_data = combined_data.reshape(num_samples, -1)  # Flatten spatial dimensions for normalization

    #train_dataloader, val_dataloader, test_dataloader, scaler_x, scaler_y  = preprocess_data(combined_data, y_data, perc_data.shape, seq_length, batch_size, channels = input_channels)
    #train_dataloader, val_dataloader, test_dataloader, scaler_x, scaler_y  = preprocess_data(sat_combined_data, y_data_sat, sat_prec_data.shape, sat_seq_length, batch_size, channels = input_channels)
    
    # Load the saved model state dict
    #model.load_state_dict(torch.load('/Users/SverreB/Github_Repo/sverrbey_project/model/save/CNN_LSTM_1.pth'))


    # Train the model
    #model_LSTM      = train_model(model_LSTM, train_dataloader, val_dataloader, 'LSTM_gaula_res_sat', scaler_y)

    # Load the saved model state dict
    #model_LSTM.load_state_dict(torch.load('/Users/SverreB/Github_Repo/sverrbey_project/model/save/LSTM_gaula_res_8000.pth'))

    # Plot predictions vs actuals
    #plot_predictions_vs_actuals(model_LSTM, test_dataloader, scaler_y, 'LSTM_Satellite')
    
if __name__ == "__main__":
    main()