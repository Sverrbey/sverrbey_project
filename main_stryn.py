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
import matplotlib.patches as mpatches
import matplotlib.dates as mdates
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
 
    Stryn_HBV_result_1 = pd.read_csv("data/SimulatedDischarge_Grasdøla.txt", sep="\t", skiprows=12, names=["Datetime", "Q","Qsim"], na_values=[-99.0, -99, "#VALUE!"])
    Stryn_HBV_result_2 = pd.read_csv("data/SimulatedDischarge_Strynsvatn.txt", sep="\t", skiprows=12, names=["Datetime", "Q","Qsim"], na_values=[-99.0, -99, "#VALUE!"])
    Stryn_dates        = pd.to_datetime(Stryn_HBV_result_1["Datetime"], format="%d.%m.%Y %H:%M")
    Stryn_HBV_actuals = pd.concat([Stryn_HBV_result_1["Q"], Stryn_HBV_result_2["Q"]], axis=1)
    Stryn_HBV_actuals.columns = ["Station_1", "Station_2"]
    Stryn_HBV = pd.concat([Stryn_HBV_result_1["Qsim"], Stryn_HBV_result_2["Qsim"]], axis=1)
    Stryn_HBV.columns = ["Station_1", "Station_2"]


    
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
###
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
    

    #plot_time_series(prec_data, temp_data, evap_data, df_disch, Stryn_dates)
    
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
    
    
    # Combining the data for training
    num_samples, height, width = prec_data.shape
    prec_data_reshaped = prec_data.reshape(num_samples, -1)  # Shape: (num_samples, height * width)
    temp_data_reshaped = temp_data.reshape(num_samples, -1)
    
    #Switching
    evap_data_reshaped = evap_data.reshape(num_samples, -1)
    
    # Remove rows with NaN values in the reshaped data
    prec_data_reshaped = np.nan_to_num(prec_data_reshaped, nan=0.0)  # Replace NaN values with 0
    temp_data_reshaped = np.nan_to_num(temp_data_reshaped, nan=0.0)
    
    # Swtching
    evap_data_reshaped = np.nan_to_num(evap_data_reshaped, nan=0.0)

    # Stack the features along the last axis
    combined_data = np.stack([prec_data_reshaped, temp_data_reshaped, evap_data_reshaped], axis=-1)  
    combined_data = combined_data.reshape(num_samples, -1)  # Flatten spatial dimensions for normalization

    
    # Define the model parameters
    num_layers      = 4         # Increase the number of LSTM layers
    dropout         = 0.4       # Adjust the dropout rate
    batch_size      = 2**6      # 64
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
   
    # Define the model
    model_LSTM         = LSTM(input_channels, height, width, hidden_size, output_size, num_layers=num_layers, dropout = dropout)
    model_BiLSTM       = BiLSTM(input_channels, height, width, hidden_size, output_size, num_layers=num_layers, dropout = dropout)

    ################################## CASE 1 ########################################
    train_dataloader, val_dataloader, test_dataloader, scaler_y  = preprocess_data(combined_data, y_data, input_size, seq_length, batch_size, num_samples, channels = input_channels)
    
    h_dataloader, h_val_dataloader, h_test_dataloader, scaler_y_hybrid = preprocess_data(combined_data, y_data, input_size, seq_length, batch_size, num_samples, channels = input_channels)
    
        
    ################################## CASE 1 ########################################
    # Train the model
    #model_LSTM     = train_model(model_LSTM, train_dataloader, val_dataloader, 'LSTM_stryn_historical', scaler_y)
    #model_BiLSTM   = train_model(model_BiLSTM, h_dataloader, h_val_dataloader, 'Hybrid_stryn_historical', scaler_y_hybrid)

    #model_LSTM    = train_model(model_LSTM, train_dataloader, val_dataloader, 'LSTM_flat_stryn_historical', scaler_y)
    #model_BiLSTM  = train_model(model_BiLSTM, h_dataloader, h_val_dataloader, 'BiLSTM_flat_stryn_historical', scaler_y_hybrid)

  
    
    # Load the saved model
    model_LSTM.load_state_dict(torch.load('model/save/LSTM_stryn_historical.pth'))
    model_BiLSTM.load_state_dict(torch.load('model/save/Hybrid_stryn_historical.pth'))

    len_train = len(train_dataloader.dataset)
    len_val   = len(val_dataloader.dataset)
    len_test  = len(test_dataloader.dataset)
   
    Stryn_HBV_test = Stryn_HBV[int(len_train + len_val):]       # Limit to the last 15% of the data for testing
    Stryn_HBV_actuals_test = Stryn_HBV_actuals[int(len_train + len_val):]   # Limit to the last 15% of the data for testing
    Stryn_dates_test = Stryn_dates[int(len_train + len_val):]       # Adjust dates to match the test data length
    # Plot predictions vs actuals
    #plot_historical(model_LSTM, model_BiLSTM, test_dataloader, h_test_dataloader, Stryn_HBV_test, scaler_y, scaler_y_hybrid, Stryn_dates_test)
    #plot_historical_peak(model_LSTM, model_BiLSTM, test_dataloader, h_test_dataloader, Stryn_HBV_test, scaler_y, scaler_y_hybrid, Stryn_dates_test)
    #plot_historical_minifigures(model_LSTM, model_BiLSTM, test_dataloader, h_test_dataloader, Stryn_HBV_test, scaler_y, scaler_y_hybrid, Stryn_dates_test)
    #plot_historical_abnormal(model_LSTM, model_BiLSTM, test_dataloader, h_test_dataloader, Stryn_HBV_test, scaler_y, scaler_y_hybrid, Stryn_dates_test)

    #exploratory(height, width)
    #print_test_results(model_BiLSTM, h_test_dataloader, scaler_y_hybrid, name="BiLSTM")
    #print_test_results(model_LSTM, test_dataloader, scaler_y, name="LSTM")
    #evaluation_HBV(Stryn_HBV, Stryn_HBV_actuals, name="HBV")
    #plot_study_area_v2()

    #plot_climate_sensitivity(model_LSTM, model_BiLSTM, test_dataloader, h_test_dataloader, Stryn_HBV_test, Stryn_HBV_actuals_test, scaler_y, scaler_y_hybrid)   
    
    #interpretability_results(model_LSTM, test_dataloader, "LSTM", channels = input_channels)
    #interpretability_results(model_BiLSTM, test_dataloader, "BiLSTM", channels = input_channels)

    #interpret_file = ["Stryn_attributions_LSTM_station_0.csv", "Stryn_attributions_LSTM_station_1.csv"]
    #plot_IG(model_LSTM, interpret_file, Stryn_dates_test, test_dataloader, scaler_y, name="LSTM")

    #interpret_file = ["Stryn_attributions_BiLSTM_station_0.csv", "Stryn_attributions_BiLSTM_station_1.csv"]
    #plot_IG(model_BiLSTM, interpret_file, Stryn_dates_test, h_test_dataloader, scaler_y, name="BiLSTM")

    plot_uncertainty_values(Stryn_HBV_test, Stryn_HBV_actuals_test)

def plot_catchment_data(points, catchment_extent):

    # Stryn
    catchments_file = r"data\Stryn\HBV snow\Elevation.rst"

     # Read catchment raster
    with rasterio.open(catchments_file) as src:
        catchment = src.read(1)
        catchment_extent = rasterio.plot.plotting_extent(src)

    met_stations = {}
    hydro_stations = {}
    for key, value in points.items():
        if key == "5" or key == "6": # Hydro stations
            hydro_stations[key] = (value[1], value[0])
        met_stations[key] = (value[1], value[0]) # Convert UTM to lat/lon
    
      # Plot
    fig, ax = plt.subplots(figsize=(10, 10))
    # Plot catchment
    show(catchment, extent=catchment_extent, ax=ax, cmap='mako', alpha=1, zorder=1, label='Catchment Area')
   

    # Set axis limits to raster extent
    ax.set_xlim(catchment_extent[0], catchment_extent[1])
    ax.set_ylim(catchment_extent[2], catchment_extent[3])
    # Plot meteorological stations
    for i, (lat, lon) in enumerate(met_stations.values()):
        label = "Meteorological Station" if i == 0 else None
        ax.plot(lon, lat, 'ro', color="#15FF00" ,markersize=8, zorder=3, label=label)
    # Plot hydrological stations
    for i, (lat, lon) in enumerate(hydro_stations.values()):
        label = "Hydrological Station" if i == 0 else None
        ax.plot(lon, lat, 'bo', color="#FF0000", markersize=8, zorder=3, label=label)

    ax.set_title("Stryn: Runoff Area with Insitu Stations", fontsize=16)
    ax.set_xlabel("Easting (UTM 33N)")
    ax.set_ylabel("Northing (UTM 33N)")
    ax.legend(loc='upper right', fontsize=10 )
    plt.tight_layout(pad=4.0)
    plt.show()

def plot_time_series(precipitation, temperature, evaporation, discharge, dates):

    """
    Plot precipitation and show the seasonal distribution of the data (rainfall vs. snowfall)
    Seasonal variation in evaporation and temperature
    """
    
    dates = pd.to_datetime(dates, format="%Y-%m-%d")  # Convert dates to datetime objects

    # I have a grid og temperature and precipitation data, I want to iterate through
    # the grid of temperatures and collect the amount of rainfall and snowfall for each day.
    snowfall = []
    rainfall = []
    evap = []
    for i in range(temperature.shape[0]):
        # temp and prec is in a shape of (8, 10) 
        temp = temperature[i, :, :]  # Get the temperature for the day
        prec = precipitation[i, :, :]  # Get the precipitation for the day
        # Create a mask for snowfall (assuming snowfall is when temperature is below 0)
        snowfall_mask = temp < 0
        snowfall.append(prec[snowfall_mask].sum())  # Sum the snowfall for the day
        rainfall.append(prec[~snowfall_mask].sum())  # Sum the rainfall for the day
        evap.append(evaporation[i, :, :].sum())
    snowfall = np.array(snowfall)
    rainfall = np.array(rainfall)
    evap = np.array(evap)

    # Flatten temperature spatially if needed
    if temperature.ndim > 1:
        temp_flat = temperature.mean(axis=tuple(range(1, temperature.ndim)))
    else:
        temp_flat = temperature

    df = pd.DataFrame({
        "date": pd.to_datetime(dates),
        "temperature": temp_flat,
        "rainfall": rainfall,
        "snowfall": snowfall,
        "evaporation": evap
    })
    df["year"] = df["date"].dt.year

    yearly = df.groupby("year").mean(numeric_only=True)

    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.plot(yearly.index, yearly["temperature"], color="red", marker="o", label="Avg Temp (°C)")
    ax1.set_ylabel("Temperature (°C)", color="red")
    ax1.tick_params(axis='y', labelcolor='red')
    
    ax4 = ax1.twinx()
    ax4.spines['left'].set_position(('axes', -0.1))  # Move the evaporation axis inward on the left
    ax4.plot(yearly.index, yearly["evaporation"], color="purple", marker="x", label="Avg Evaporation (mm)")
    ax4.set_ylabel("Evaporation (mm)", color="purple")
    ax4.tick_params(axis='y', labelcolor='purple')
    ax4.yaxis.set_label_position("left")  # Ensure the label is on the left side
    ax4.spines['left'].set_visible(True)  # Make the left spine visible
    ax4.yaxis.set_ticks_position('left')  # Ensure ticks are on the left side

    ax2 = ax1.twinx()
    ax2.bar(yearly.index-0.15, yearly["rainfall"], width=0.3, color="blue", alpha=0.5, label="Avg Rainfall (mm)")
    ax2.set_ylabel("Rain (mm)", color="blue")
    ax2.tick_params(axis='y', labelcolor='blue')

    ax3 = ax1.twinx()
    ax3.spines['right'].set_position(('outward', 60))  # Offset the third axis
    ax3.bar(yearly.index+0.15, yearly["snowfall"], width=0.3, color="green", alpha=0.5, label="Avg Snowfall (mm)")
    ax3.set_ylabel("Snow (mm)", color="green")
    ax3.tick_params(axis='y', labelcolor='green')
    ax3.invert_yaxis()

    ax1.set_xlabel("Year")
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    lines3, labels3 = ax3.get_legend_handles_labels()
    lines4, labels4 = ax4.get_legend_handles_labels()
    ax4.legend(
        lines + lines2 + lines3 + lines4,
        labels + labels2 + labels3 + labels4,
        loc='upper center',
        bbox_to_anchor=(0.5, -0.18),  # Centered below the axes
        ncol=2,                       # Number of columns in the legend
        frameon=False
    )

    plt.title("Stryn: Yearly Average Temperature, Rainfall, and Snowfall")
    plt.tight_layout()
    plt.show()

def plot_time_series_gaula(precipitation, temperature, radiation, relhumidity, windstats, discharge, dates):

    """
    Plot precipitation and show the seasonal distribution of the data (rainfall vs. snowfall)
    Seasonal variation in evaporation and temperature
    """
    
    dates = pd.to_datetime(dates, format="%Y-%m-%d")  # Convert dates to datetime objects

    # I have a grid og temperature and precipitation data, I want to iterate through
    # the grid of temperatures and collect the amount of rainfall and snowfall for each day.
    temps = []
    snowfall = []
    rainfall = []
    rad = []
    relhum = []
    wind = []
    for i in range(temperature.shape[0]):
        # temp and prec is in a shape of (8, 10) 
        temp = temperature[i, :, :]  # Get the temperature for the day
        prec = precipitation[i, :, :]  # Get the precipitation for the day
        # Create a mask for snowfall (assuming snowfall is when temperature is below 0)
        snowfall_mask = temp < 0
        temps.append(temperature[i, :, :].mean())  # Average temperature for the day
        snowfall.append(prec[snowfall_mask].sum())  # Sum the snowfall for the day
        rainfall.append(prec[~snowfall_mask].sum())  # Sum the rainfall for the day
        rad.append(radiation[i, :, :].mean())
        relhum.append(relhumidity[i, :, :].mean())
        wind.append(windstats[i, :, :].mean())
    snowfall = np.array(snowfall)
    rainfall = np.array(rainfall)
    rad = np.array(rad)
    relhum = np.array(relhum)
    wind = np.array(wind)

    df = pd.DataFrame({
        "date": pd.to_datetime(dates),
        "temperature": temps,
        "rainfall": rainfall,
        "snowfall": snowfall,
        "radiation": rad,
        "relhumidity": relhum,
        "wind": wind
    })
    df["year"] = df["date"].dt.year

    yearly = df.groupby("year").mean(numeric_only=True)

    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.plot(yearly.index, yearly["temperature"], color="red", marker="o", label="Avg Temp (°C)")
    ax1.set_ylabel("Temperature (°C)", color="red")
    ax1.tick_params(axis='y', labelcolor='red')
    
    ax4 = ax1.twinx()
    ax4.spines['left'].set_position(('axes', -0.1))  # Move the evaporation axis inward on the left
    ax4.plot(yearly.index, yearly["radiation"], color="purple", marker="x", label="Avg Global Radiation (W/m²)")
    ax4.set_ylabel("Global Radiation (W/m²)", color="purple")
    ax4.tick_params(axis='y', labelcolor='purple')
    ax4.yaxis.set_label_position("left")  # Ensure the label is on the left side
    ax4.spines['left'].set_visible(True)  # Make the left spine visible
    ax4.yaxis.set_ticks_position('left')  # Ensure ticks are on the left side

    ax5 = ax1.twinx()
    ax5.spines['left'].set_position(('axes', -0.2))  # Move the evaporation axis inward on the left
    ax5.plot(yearly.index, yearly["relhumidity"], color="orange", marker="x", label="Avg Relative Humidity (%)")
    ax5.set_ylabel("Relative Humidity (%)", color="orange")
    ax5.tick_params(axis='y', labelcolor='orange')
    ax5.yaxis.set_label_position("left")  # Ensure the label is on the left side
    ax5.spines['left'].set_visible(True)  # Make the left spine visible
    ax5.yaxis.set_ticks_position('left')  # Ensure ticks are on the left side

    ax6 = ax1.twinx()
    ax6.spines['left'].set_position(('axes', -0.3))  # Move the evaporation axis inward on the left
    ax6.plot(yearly.index, yearly["wind"], color="grey", marker="x", label="Avg Wind (m/s)")
    ax6.set_ylabel("Wind (m/s)", color="grey")
    ax6.tick_params(axis='y', labelcolor='grey')
    ax6.yaxis.set_label_position("left")  # Ensure the label is on the left side
    ax6.spines['left'].set_visible(True)  # Make the left spine visible
    ax6.yaxis.set_ticks_position('left')  # Ensure ticks are on the left side

    ax2 = ax1.twinx()
    ax2.bar(yearly.index-0.15, yearly["rainfall"], width=0.3, color="blue", alpha=0.5, label="Avg Rainfall (mm)")
    ax2.set_ylabel("Rain (mm)", color="blue")
    ax2.tick_params(axis='y', labelcolor='blue')

    ax3 = ax1.twinx()
    ax3.spines['right'].set_position(('outward', 60))  # Offset the third axis
    ax3.bar(yearly.index+0.15, yearly["snowfall"], width=0.3, color="green", alpha=0.5, label="Avg Snowfall (mm)")
    ax3.set_ylabel("Snow (mm)", color="green")
    ax3.tick_params(axis='y', labelcolor='green')
    ax3.invert_yaxis()

    ax1.set_xlabel("Year")
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    lines3, labels3 = ax3.get_legend_handles_labels()
    lines4, labels4 = ax4.get_legend_handles_labels()
    lines5, labels5 = ax5.get_legend_handles_labels()
    lines6, labels6 = ax6.get_legend_handles_labels()
    ax4.legend(
        lines + lines2 + lines3 + lines4 + lines5 + lines6,
        labels + labels2 + labels3 + labels4 + labels5 + labels6,
        loc='upper center',
        bbox_to_anchor=(0.5, -0.18),  # Centered below the axes
        ncol=2,                       # Number of columns in the legend
        frameon=False
    )

    plt.title("Gaula: Yearly Average Temperature, Rainfall, and Snowfall")
    plt.tight_layout()
    plt.show()

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
    colors = ["#6decf0", "#088819", "#15FF00", "#1F58D4", "viridis", "#FF5733", "purple", "orange"]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(dates, actuals[:, 0], label='Streamflow', color=colors[3])
    ax.plot(dates, predictions[:, 0], label='LSTM', color=colors[1], linestyle='dotted')
    ax.plot(dates, predictions_h[:, 0], label='BiLSTM', color='purple', linestyle='dashed')
    ax.plot(dates, HBV[:, 0], label='HBV', color='orange', linestyle='dashdot')

    ax.plot(dates, actuals[:, 1], color=colors[3])
    ax.plot(dates, predictions[:, 1], color=colors[1], linestyle='dotted')
    ax.plot(dates, predictions_h[:, 1], color='purple', linestyle='dashed')
    ax.plot(dates, HBV[:, 1], color='orange', linestyle='dashdot')

    std_deviation, h1, h2 = HMMR_2(actuals, HBV)

    # Plot uncertainty (shaded area)
    ax.fill_between(dates,
                    actuals[:,0] - std_deviation[0, h1],
                    actuals[:,0] + std_deviation[0, h1],
                    color=colors[3], alpha=0.2, label='Standard Deviation')
    ax.fill_between(dates,
                actuals[:,1] - std_deviation[1, h2],
                actuals[:,1] + std_deviation[1, h2],
                color=colors[3], alpha=0.2)

    ax.set_ylabel('Streamflow [m³/s]')

    # Set major ticks at each year and minor ticks at each month
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('\n%Y'))  # Newline for year label below months

    ax.xaxis.set_minor_locator(mdates.MonthLocator())
    ax.xaxis.set_minor_formatter(mdates.DateFormatter('%b'))

    # Show both major and minor ticks
    ax.tick_params(axis='x', which='major', length=10, labelsize=14)
    ax.tick_params(axis='x', which='minor', length=5, labelsize=12)

    # Rotate month labels for clarity
    plt.setp(ax.get_xticklabels(minor=True), ha='right', rotation=90, fontsize=10)

    ax.set_title('Stryn: Predictions with Uncertainty and HBV Comparison', fontsize=16)
    ax.legend()
    plt.tight_layout(pad=2.0)
    plt.show()
    return

def plot_historical_peak(model, hybrid, dataloader, h_dataloader, HBV, scaler_y, scaler_y_h, dates):
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
    colors = ["#6decf0", "#088819", "#15FF00", "#1F58D4", "viridis", "#FF5733", "purple", "orange"]

    lines = []
    labels = []
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    axs = axs.flatten()  # Flatten the 2D array of axes to 1D for easier iteration
    #fig.suptitle('Stryn: Peak Flow Events', fontsize=18)
    for i, ax in enumerate(axs):
        ax.set_ylabel('Streamflow [m³/s]')
        ax.set_title(f'{i+5}', fontsize=16)
    
        ax.plot(dates, actuals[:, 1], label='Streamflow', color=colors[3])
        ax.plot(dates, predictions[:, 1], label='LSTM', color=colors[1], linestyle='dotted')
        ax.plot(dates, predictions_h[:, 1], label='BiLSTM', color='purple', linestyle='dashed')
        ax.plot(dates, HBV[:, 1], label='HBV', color='orange', linestyle='dashdot')

        std_deviation, h1, h2 = HMMR_2(actuals, HBV)

        # Plot uncertainty (shaded area)
        ax.fill_between(dates,
                        actuals[:,1] - std_deviation[1, h2],
                        actuals[:,1] + std_deviation[1, h2],
                        color=colors[3], alpha=0.2, label='Standard Deviation')


        # Set major ticks at each month and minor ticks at each day
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%y'))  # Year-Month format

        ax.xaxis.set_minor_locator(mdates.DayLocator(interval=5))
        ax.xaxis.set_minor_formatter(mdates.DateFormatter('%d'))

        # Show both major and minor ticks
        ax.tick_params(axis='x', which='major', length=10, labelsize=14, pad=10)   # Increase pad for major ticks
        ax.tick_params(axis='x', which='minor', length=5, labelsize=12, pad=2)     # Decrease pad for minor ticks
        ax.tick_params(axis='y', labelsize=12)
        # Rotate month labels for clarity
        plt.setp(ax.get_xticklabels(minor=True), ha='right', rotation=90, fontsize=10)
        
        # Store lines for the legend
        new_line, new_label = ax.get_legend_handles_labels()
        lines.extend(new_line)  # Streamflow
        labels.extend(new_label)  # Streamflow label
    unique = dict(zip(labels, lines))
    fig.legend(unique.values(), unique.keys(), loc='center', bbox_to_anchor=(0.5, 0.02), ncol=5, frameon=False, fontsize=14)
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.show()
    return

def plot_historical_abnormal(model, hybrid, dataloader, h_dataloader, HBV, scaler_y, scaler_y_h, dates):
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
    colors = ["#6decf0", "#088819", "#15FF00", "#1F58D4", "viridis", "#FF5733", "purple", "orange"]

    lines = []
    labels = []
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    axs = axs.flatten()  # Flatten the 2D array of axes to 1D for easier iteration
    #fig.suptitle('Stryn: Abnormal Flow Events', fontsize=18)
    for i, ax in enumerate(axs):
        ax.set_ylabel('Streamflow [m³/s]')
        ax.set_title(f'{i+5}', fontsize=16)
    
        ax.plot(dates, actuals[:, 1], label='Streamflow', color=colors[3])
        ax.plot(dates, predictions[:, 1], label='LSTM', color=colors[1], linestyle='dotted')
        ax.plot(dates, predictions_h[:, 1], label='BiLSTM', color='purple', linestyle='dashed')
        ax.plot(dates, HBV[:, 1], label='HBV', color='orange', linestyle='dashdot')

        std_deviation, h1, h2 = HMMR_2(actuals, HBV)

        # Plot uncertainty (shaded area)
        ax.fill_between(dates,
                        actuals[:,1] - std_deviation[1, h2],
                        actuals[:,1] + std_deviation[1, h2],
                        color=colors[3], alpha=0.2, label='Standard Deviation')


        # Set major ticks at each month and minor ticks at each day
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%y'))  # Year-Month format

        ax.xaxis.set_minor_locator(mdates.DayLocator(interval=5))
        ax.xaxis.set_minor_formatter(mdates.DateFormatter('%d'))

        # Show both major and minor ticks
        ax.tick_params(axis='x', which='major', length=10, labelsize=14, pad=10)   # Increase pad for major ticks
        ax.tick_params(axis='x', which='minor', length=5, labelsize=12, pad=2)     # Decrease pad for minor ticks
        ax.tick_params(axis='y', labelsize=12)
        # Rotate month labels for clarity
        plt.setp(ax.get_xticklabels(minor=True), ha='right', rotation=90, fontsize=10)
        
        # Store lines for the legend
        new_line, new_label = ax.get_legend_handles_labels()
        lines.extend(new_line)  # Streamflow
        labels.extend(new_label)  # Streamflow label
    unique = dict(zip(labels, lines))
    fig.legend(unique.values(), unique.keys(), loc='center', bbox_to_anchor=(0.5, 0.02), ncol=5, frameon=False, fontsize=14)
    plt.tight_layout(rect=[0, 0.04, 1, 1])
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
    colors = ["#6decf0", "#088819", "#15FF00", "#1F58D4", "viridis", "#FF5733", "purple", "orange"]
    #years = [2018, 2019]
    years = [2020, 2021, 2022]
    n_years = len(years)
    fig, axes = plt.subplots(n_years, 1, figsize=(12, 3 * n_years), sharex=False)
    years_arr = dates.dt.year if hasattr(dates, 'dt') else dates.year
    
    for i, year in enumerate(years):
        ax = axes[i]
        mask = years_arr == year
        if not np.any(mask):
            continue  # Skip if no data for this year

        ax.plot(dates[mask], actuals[mask, 0], label="Streamflow", color=colors[3])
        ax.plot(dates[mask], predictions[mask, 0], label="LSTM", color=colors[1], linestyle='dotted')
        ax.plot(dates[mask], predictions_h[mask, 0], label="BiLSTM", color='purple', linestyle='dashed')
        ax.plot(dates[mask], HBV[mask, 0], label="HBV", color='orange', linestyle='dashdot')

        ax.plot(dates[mask], actuals[mask, 1], color=colors[3])
        ax.plot(dates[mask], predictions[mask, 1], color=colors[1], linestyle='dotted')
        ax.plot(dates[mask], predictions_h[mask, 1], color='purple', linestyle='dashed')
        ax.plot(dates[mask], HBV[mask, 1], color='orange', linestyle='dashdot')
        
      
        std_deviation, h1, h2 = HMMR_2(actuals[mask], HBV[mask])
        ax.fill_between(
            dates[mask],
            actuals[mask, 0] - std_deviation[0, h1],
            actuals[mask, 0] + std_deviation[0, h1],
            color=colors[3], alpha=0.2
        )
        ax.fill_between(
            dates[mask],
            actuals[mask, 1] - std_deviation[1, h2],
            actuals[mask, 1] + std_deviation[1, h2],
            color=colors[3], alpha=0.2
        )
        ax.set_ylabel('Streamflow [m³/s]')
        ax.set_title(f'Stryn: {year}')
        
        # Set major ticks at each year and minor ticks at each month
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('\n%Y'))  # Newline for year label below months

        ax.xaxis.set_minor_locator(mdates.MonthLocator())
        ax.xaxis.set_minor_formatter(mdates.DateFormatter('%b'))

        # Show both major and minor ticks
        ax.tick_params(axis='x', which='major', length=10, labelsize=14)
        ax.tick_params(axis='x', which='minor', length=5, labelsize=12)
        ax.tick_params(axis='y', labelsize=12)
        # Store lines for the legend
        lines, labels = ax.get_legend_handles_labels()
        #if i == 0:
        #    # Add legend only for the first subplot
        #    ax.legend(
        #        lines,
        #        labels,
        #        loc='upper center',
        #        bbox_to_anchor=(0.5, -0.18),  # Centered below the axes
        #        ncol=2,                       # Number of columns in the legend
        #        frameon=False
        #    )
    
    # Rotate month labels for clarity
    plt.setp(ax.get_xticklabels(minor=True), ha='right')
    plt.tight_layout(pad=1.0)
    plt.subplots_adjust(hspace=.75) 
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

def interpretability_results(model, test_dataloader, name, channels=1):
    station_list = [0, 1]
    for s in station_list:
        station = s
        print(f"IG Results for Stryn station {station}, Model {name}: \n")
        model.eval()
        all_grads = []
        for batch_x, batch_y in test_dataloader:
            IG_grads = integrated_gradients(model, batch_x, channels=channels, target_index=station)
            all_grads.append(IG_grads.cpu().numpy())
        
        # Shape (287, 14, 5, 7, 14)
        all_grads = np.concatenate(all_grads, axis=0)  # shape: (num_samples, seq_length, channels, height, width)

        all_grads_mean = all_grads.mean(axis=(-2, -1))
        
        # Shape (287, 5, 7 ,14)
        last_step_attributions = all_grads_mean[:, -1, :]  

        labels = ["Precipitation", "Temperature", "Evaporation"]

        # Create a DataFrame with appropriate column names
        df_attr = pd.DataFrame(last_step_attributions, columns=labels)

        # Save to CSV
        df_attr.to_csv(f"Stryn_attributions_{name}_station_{station}.csv", index=False)

    return

def plot_IG(model, file_path, dates, dataloader, scaler, name=None):
    station_list = [0, 1]
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
    

    # Inverse transform the predictions and actuals to the original scale
    predictions = scaler.inverse_transform(predictions)
    actuals = scaler.inverse_transform(actuals)
    
    dates = dates[-1340:]  # Ensure we only plot the last 1340 days
    dates = pd.to_datetime(dates, format="%Y-%m-%d")  # Convert dates to datetime objects
    

    labels = ["Precipitation", "Temperature", "Evaporation"]
    colors = ["#1F58D4",  "#FF5733", "#15FF00"]
    #colors = ["#1F58D4",  "#FF5733", "#15FF00", "#088819","#6decf0"]
    
    u_labels = []
    u_labels = []
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    axes = axes.flatten()  # Flatten the axes array for easier indexing 
    fig.suptitle(f'Stryn: Integrated Gradients Attribution: {name}', fontsize=16)
    for i, ax in enumerate(axes):

        last_step_attributions = pd.read_csv(file_path[i]).values
        total_attr = np.abs(last_step_attributions).sum(axis=1, keepdims=True)
        total_attr[total_attr == 0] = 1e-8
        shares = np.abs(last_step_attributions) / total_attr  # All values between 0 and 1, sum to 1

        shares_T = shares.T  # (n_channels, n_times)
        ax1 = ax
        ax1.stackplot(dates, shares_T, labels=labels, cmap="viridis", alpha=0.8)

        ax1.set_ylabel('Attribution')
        ax2 = ax1.twinx()
        #ax2.plot(dates, predictions[:, station], label='LSTM', color= "#088819", linestyle='dotted')
        ax2.plot(dates, predictions[:, i], label=name, color= "purple", linestyle='dashed')
        ax2.plot(dates, actuals[:, i], label='Streamflow', color=colors[0])
        ax2.set_ylabel('Streamflow [m³/s]', fontsize=12)
        ax2.tick_params(axis='y', labelsize=12)
        
        ax1.set_title(f'Station {i+1}')

        # Rotate month labels for clarity
        plt.setp(ax1.get_xticklabels(minor=True), ha='right', rotation=90, fontsize=10)
       
        # Set major ticks at each year and minor ticks at each month
        ax1.xaxis.set_major_locator(mdates.YearLocator())
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('\n%Y'))  # Newline for year label below months

        ax1.xaxis.set_minor_locator(mdates.MonthLocator(interval=3))
        ax1.xaxis.set_minor_formatter(mdates.DateFormatter('%b'))

        # Show both major and minor ticks
        ax1.tick_params(axis='x', which='major', length=10, labelsize=14)
        ax1.tick_params(axis='x', which='minor', length=5, labelsize=12)
        ax1.tick_params(axis='y', labelsize=12)
        # Store lines for the legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        lines = lines1 + lines2
        labels = labels1 + labels2
        if i == 1:
            u_labels = labels.copy()
            u_lines = lines.copy()
    unique = dict(zip(u_labels, u_lines))     
    fig.legend(unique.values(), unique.keys(), loc='center', bbox_to_anchor=(0.5, 0.03), ncol=5, frameon=False, fontsize=14)
    plt.tight_layout(pad=1.4,rect=[0, 0.08, 1, 1])
    plt.subplots_adjust(hspace=.3)
    plt.show()
    return

def plot_uncertainty_values(forecast, observation, n_states=2):

    """
    1. State Distribution of Streamflow 
    2. Distribution of Residuals
    """

    std_deviation, h1, h2 = HMMR_2(observation, forecast)

    # So the hidden states for each of the feautres is stored in h1, h2, h3.
    # For each station I want to plot the distribution of the Streamflow for 
    # each of the two hidden states. 

    fig, axes = plt.subplots(nrows=n_states, ncols=1, figsize=(10, 6 * n_states), sharex=True)
    fig.suptitle('Stryn: State Distribution of Streamflow', fontsize=16)
    colors = ["#6decf0", "#088819", "#15FF00", "#1F58D4", "viridis", "#FF5733", "purple", "orange"]
    for i in range(n_states):
        ax = axes[i]
        # Plot the distribution of the streamflow for each hidden state
        ax.hist(observation[:, 0][h1 == i], bins=30, alpha=0.5, color=colors[1], label=f'Station 1')
        ax.hist(observation[:, 1][h2 == i], bins=30, alpha=0.5, color=colors[2], label=f'Station 2')
        
        ax.set_title(f'State {i}')
        ax.set_xlabel('Streamflow [m³/s]')
        ax.set_ylabel('Frequency')
        lines, labels = ax.get_legend_handles_labels()
        if i == 0:
            # Add legend only for the first subplot
            ax.legend(
                lines,
                labels,
                loc='upper center',
                bbox_to_anchor=(0.5, -0.125),  # Centered below the axes
                ncol=3,                       # Number of columns in the legend
                frameon=False
            )
    plt.tight_layout(pad=2.0)
    plt.subplots_adjust(hspace=0.4) 
    plt.show()
    return

if __name__ == "__main__":
    main()