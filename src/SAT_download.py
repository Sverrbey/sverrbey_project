import openmeteo_requests
from openmeteo_sdk.Variable import Variable
import pandas as pd
import requests_cache
from retry_requests import retry
import asyncio
import time
import os
import numpy as np
import cv2



# Setup the Open-Meteo API client with cache and retry on error
cache_session = requests_cache.CachedSession('.cache', expire_after = 3600)
retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
openmeteo = openmeteo_requests.Client(session = retry_session)

# Make sure all required weather variables are listed here
# The order of variables in hourly or daily is important to assign them correctly below
url = "https://historical-forecast-api.open-meteo.com/v1/forecast"


sat_file_paths = [
	'data/interpolated_spatial_data/sat_prec.json',
	'data/interpolated_spatial_data/sat_temp.json',
	'data/interpolated_spatial_data/sat_snow.json'
]
resized_file_paths = [
	'data/interpolated_spatial_data/resized_sat_prec.json',
	'data/interpolated_spatial_data/resized_sat_temp.json'
]
api_key = "bq5Ny6WGYkK5ySwwkqjCBQ=="  # HydAPI key

def main():
	# Hydrological and Meterological data
    hyd_station = {
        "STNR": "88.15.0,88.11.0",
        "parameter": "1001",
        "filepath": r"data/hourly_data/insitu_discharge.json",
        "unit": "m3/s"
    }
    
    data_folder_path = "data/hourly_data"
    has_data = any(
        file.startswith("insitu_") and file.endswith(".json")
        for file in os.listdir(data_folder_path)
    )
    if not has_data:
        disc_argv = [
        "-a", api_key,
        "-s", hyd_station["STNR"],
        "-p", hyd_station["parameter"],
        "-r", 60,
        "-t", "2022-11-15/2025-05-31"
        ]
        get_observations(disc_argv,hyd_station["STNR"],filepath=hyd_station["filepath"])

    has_historic = any(
        file.startswith("hist_") and file.endswith(".json")
        for file in os.listdir(data_folder_path)
    )
    if not has_historic:
        #Collecting historic hourly data
        stryn_disch_station = {
            "STNR": "88.15.0,88.11.0",
            "parameter": "1001",
            "filepath": r"data/hourly_data/hist_insitu_discharge.json",
            "unit": "m3/s"
        }
        stryn_disch_argv = [
            "-a", api_key,
            "-s", stryn_disch_station["STNR"],
            "-p", stryn_disch_station["parameter"],
            "-r", 60,
            "-t", "1980-01-01/2022-02-06"
        ]
        get_observations(stryn_disch_argv,stryn_disch_station["STNR"],filepath=stryn_disch_station["filepath"])
    
    #Create interpolated satellite data
    start_date = "2022-11-15"
    end_date   = "2025-05-31"        
    data_folder_path = f"data/satellite_data/"
    
    # DWD Germany Data Download
    long_list_DWD, lat_list_DWD, width_DWD, height_DWD = create_grid(x_degrees, y_degrees, 7) # Create a grid of points for DWD data
    
    has_DWD_data = any(
        file.startswith("DWD_") and file.endswith(".json")
        for file in os.listdir(data_folder_path)
    )
    if not has_DWD_data:
        print("Downloading DWD data...")
        interval_seconds    = 60
        update_number       = len(lat_list_DWD)*len(long_list_DWD)
        lat_len             = len(lat_list_DWD)
        repeat_lat          = lat_list_DWD.copy()  # Copy the latitudes to repeat them for each longitude
        counter = 0
        while counter < lat_len:
            repeat_lat = np.repeat(lat_list_DWD[counter], len(long_list_DWD))
            download_sequence2(long_list_DWD, repeat_lat, start_date, end_date)
            print("Sleeping for 1 minutes before the next download...")
            print(f"{(counter + 1)*len(long_list_DWD)} of {update_number} DWD data points downloaded")
            counter += 1
            time.sleep(interval_seconds)  # Wait for the specified interval before the next download
        print("DWD data downloaded successfull")
    else: 
        print("DWD data already downloaded, skipping download step.")

    # MET Nordic Data Download
    long_list, lat_list, width, height = create_grid(x_degrees, y_degrees, 1) # Create a grid of points
    
    has_MET_data = any(
        file.startswith("MET_") and file.endswith(".json")
        for file in os.listdir(data_folder_path)
    )
    if not has_MET_data:
        print("Downloading MET data...")
        interval_seconds    = 60
        third_width          = len(long_list) // 3
        lat_len             = len(lat_list)
        update_number       = lat_len * len(long_list)
        repeat_lat          = lat_list.copy()  # Copy the latitudes to repeat them for each longitude
        divided_long        = long_list.copy()  # Copy the longitudes to divide them into two halves
        divided_long        = long_list[:third_width]  # Start with the first half of the longitudes
        counter_long        = 2     # 0: [0 - 1080], 1: 1: [1080 - 2160], 2: [2160 - 3240]     

        if counter_long == 1:    
            divided_long = long_list[third_width:third_width*2]
        elif counter_long == 2:
            divided_long = long_list[third_width*2:]
        counter = 0
        while counter < lat_len:
            repeat_lat = np.repeat(lat_list[counter], len(divided_long))
            download_sequence1(divided_long, repeat_lat, start_date, end_date)
            print("Sleeping for 1 minutes before the next download...")
            print(f"{counter, counter_long} of {lat_len, 2} MET data points downloaded")
            counter += 1
            time.sleep(interval_seconds)  # Wait for the specified interval before the next download
        counter_long += 1
        print("MET data downloaded successfull")
    else: 
        print("MET data already downloaded, skipping download step.")

    data_folder_path = "data/interpolated_spatial_data"
    has_resized_data = any(
        file.startswith("resized_") and file.endswith(".json")
        for file in os.listdir(data_folder_path)
    )
    
    if not has_resized_data:
        print("Resizing satellite data...")
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

        # For each row in the dataframe reshape the values in the row into the image, 
        # and then resize the image to the desired size.
        resized_prec = []
        resized_temp = []
        
        for i in range(len(satellite_temp)):
            # Reshape the values in the row into the image
            sat_image_prec = satellite_prec.iloc[i].values.reshape(height, width)
            sat_image_temp = satellite_temp.iloc[i].values.reshape(height, width)
            
            # Resize the image to the desired size
            new_width, new_height = 16, 5
            arr_prec = cv2.resize(sat_image_prec, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
            arr_temp = cv2.resize(sat_image_temp, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
            resized_prec.append(arr_prec.flatten())
            resized_temp.append(arr_temp.flatten())
        
        resized_data_prec = np.array(resized_prec)
        resized_data_temp = np.array(resized_temp)

        # Save the resized data to JSON files
        prec_df = pd.DataFrame(resized_data_prec, index=satellite_prec.index)
        prec_df.to_json(resized_file_paths[0], orient='index')
        temp_df = pd.DataFrame(resized_data_temp, index=satellite_temp.index)
        temp_df.to_json(resized_file_paths[1], orient='index')
        print("Resized satellite data saved successfully.")
    else:   
        print("Resized satellite data already exists, skipping resizing step.")
    

    has_snow_data = any(
        file.startswith("sat_snow") and file.endswith(".json")
        for file in os.listdir(data_folder_path)
    )
    if not has_snow_data:
        print("Downloading DWD snow data...")
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
        
        # Save the resized data to JSON files
        snow_df = pd.DataFrame(satellite_snow, index=satellite_snow.index)
        snow_df.to_json(sat_file_paths[2], orient='index')
        
        print("Stored Snowdata successfully.")
    else:
        print("Snow data already exists, skipping download step.")



def get_MET_data(params):

	responses = openmeteo.weather_api(url, params=params)

	# Process first location. Add a for-loop for multiple locations or weather models
	response = responses[0]
	print(f"Coordinates {response.Latitude()}°N {response.Longitude()}°E")
	print(f"Elevation {response.Elevation()} m asl")
	print(f"Timezone {response.Timezone()}{response.TimezoneAbbreviation()}")
	print(f"Timezone difference to GMT+0 {response.UtcOffsetSeconds()} s")

	# Process hourly data. The order of variables needs to be the same as requested.
	hourly = response.Hourly()
	hourly_temperature_2m = hourly.Variables(0).ValuesAsNumpy()
	hourly_precipitation = hourly.Variables(1).ValuesAsNumpy()

	hourly_data = {"date": pd.date_range(
		start = pd.to_datetime(hourly.Time(), unit = "s", utc = True),
		end = pd.to_datetime(hourly.TimeEnd(), unit = "s", utc = True),
		freq = pd.Timedelta(seconds = hourly.Interval()),
		inclusive = "left"
	)}

	hourly_data["temperature_2m"] 	= hourly_temperature_2m
	hourly_data["precipitation"] 	= hourly_precipitation

	hourly_dataframe = pd.DataFrame(data = hourly_data)
	
	# Write the dataframes to JSON files
	hourly_file_path = f"data/satellite_data/MET_{params['longitude']}_{params['latitude']}.json"
	hourly_dataframe.to_json(hourly_file_path, orient='values')

	return

def get_DWD_data(params):

	responses = openmeteo.weather_api(url, params=params)

	# Process first location. Add a for-loop for multiple locations or weather models
	response = responses[0]
	print(f"Coordinates {response.Latitude()}°N {response.Longitude()}°E")
	print(f"Elevation {response.Elevation()} m asl")
	print(f"Timezone {response.Timezone()}{response.TimezoneAbbreviation()}")
	print(f"Timezone difference to GMT+0 {response.UtcOffsetSeconds()} s")

	# Process hourly data. The order of variables needs to be the same as requested.
	hourly = response.Hourly()
	hourly_snow_depth = hourly.Variables(0).ValuesAsNumpy()

	hourly_data = {"date": pd.date_range(
		start = pd.to_datetime(hourly.Time(), unit = "s", utc = True),
		end = pd.to_datetime(hourly.TimeEnd(), unit = "s", utc = True),
		freq = pd.Timedelta(seconds = hourly.Interval()),
		inclusive = "left"
	)}

	hourly_data["snow_depth"] 	= hourly_snow_depth

	hourly_dataframe = pd.DataFrame(data = hourly_data)
	
	# Write the dataframes to JSON files
	hourly_file_path = f"data/satellite_data/DWD_{params['longitude']}_{params['latitude']}.json"
	hourly_dataframe.to_json(hourly_file_path, orient='values')

	return
