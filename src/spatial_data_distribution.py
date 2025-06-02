import numpy as np
import pandas as pd
import rasterio
import os
from rasterio.transform import from_origin
from src.UKMO_download import *

def reformat_data_stryn(points, perc_values, temp_values, evap_values, disch_values, number_days, dates, stryn_file_paths, extent=(1, 1, 1, 1), spacing=100, power=1):
    
    # Calculate the number of grid points based on the spacing
    num_points_x = int((extent[1] - extent[0]) / spacing) + 1
    num_points_y = int((extent[3] - extent[2]) / spacing) + 1

    # Create a grid of points
    grid_x, grid_y = np.meshgrid(np.linspace(extent[0], extent[1], num_points_x), np.linspace(extent[2], extent[3], num_points_y))
    # For each row of input data, create a grid of interpolated values
    interp_perc = []
    interp_temp = []
    interp_evap = []

    perc_file_path, disch_file_path, temp_file_path, evap_file_path = stryn_file_paths

    for n in range(0, number_days):
        perc = np.array(list(perc_values.values())).T[n]
        temp = np.array(list(temp_values.values())).T[n]
        evap = np.array(list(evap_values.values())).T[n]

        # Filter the values from data_points based on the keys in measurements
        perc_points = [points[key] for key in perc_values.keys()]
        temp_points = [points[key] for key in temp_values.keys()]
        evap_points = [points[key] for key in evap_values.keys()]

        
        interp_perc.append(inverse_distance_weighting(perc_points, perc, x_grid=grid_x[0], y_grid=grid_y.T[0], power=2))
        interp_temp.append(inverse_distance_weighting(temp_points, temp, x_grid=grid_x[0], y_grid=grid_y.T[0], power=1))
        interp_evap.append(inverse_distance_weighting(evap_points, evap, x_grid=grid_x[0], y_grid=grid_y.T[0], power=1))
    
    df_perc = pd.DataFrame()
    df_perc["date"] = dates
    df_perc["interpolated_perc"] = interp_perc

    df_temp = pd.DataFrame()
    df_temp["date"] = dates
    df_temp["interpolated_temp"] = interp_temp

    df_evap = pd.DataFrame()
    df_evap["date"] = dates
    df_evap["interpolated_evap"] = interp_evap

    df_discharge = pd.DataFrame()
    df_discharge["date"] = dates
    df_discharge["discharge"] = list(np.array(list(disch_values.values())).T)

    # save the dataframe as a csv file 
    df_perc.to_json(perc_file_path, orient='values')
    df_discharge.to_json(disch_file_path, orient='values')
    df_temp.to_json(temp_file_path, orient='values')
    df_evap.to_json(evap_file_path, orient='values')

    return

def reformat_data_gaula(points, perc_values, temp_values, rad_values, hyd_values, relHum_values, wind_values, disch_values, number_days, dates, gaula_file_paths, extent=(1, 1, 1, 1), spacing=100, power=1):
    # Calculate the number of grid points based on the spacing
    num_points_x = int((extent[1] - extent[0]) / spacing) + 1
    num_points_y = int((extent[3] - extent[2]) / spacing) + 1

    # Create a grid of points
    grid_x, grid_y = np.meshgrid(np.linspace(extent[0], extent[1], num_points_x), np.linspace(extent[2], extent[3], num_points_y))
    # For each row of input data, create a grid of interpolated values
    interp_perc = []
    interp_temp = []
    interp_rad = []
    interp_hyd = []
    interp_relHum = []
    interp_wind = []

    perc_file_path, disch_file_path, temp_file_path, rad_file_path, hyd_file_path, relHum_file_path, wind_file_path = gaula_file_paths

    for n in range(0, number_days):
        perc = np.array(list(perc_values.values())).T[n]
        temp = np.array(list(temp_values.values())).T[n]
        rad = np.array(list(rad_values.values())).T[n]
        hyd = np.array(list(hyd_values.values())).T[n]
        relHum = np.array(list(relHum_values.values())).T[n]
        wind = np.array(list(wind_values.values())).T[n]

        # Filter the values from data_points based on the keys in measurements
        perc_points = [points[key] for key in perc_values.keys()]
        temp_points = [points[key] for key in temp_values.keys()]
        rad_points = [points[key] for key in rad_values.keys()]
        hyd_points = [points[key] for key in hyd_values.keys()]
        relHum_points = [points[key] for key in relHum_values.keys()]
        wind_points = [points[key] for key in wind_values.keys()]

        
        interp_perc.append(inverse_distance_weighting(perc_points, perc, x_grid=grid_x[0], y_grid=grid_y.T[0], power=2))
        interp_temp.append(inverse_distance_weighting(temp_points, temp, x_grid=grid_x[0], y_grid=grid_y.T[0], power=1))
        interp_rad.append(inverse_distance_weighting(rad_points, rad, x_grid=grid_x[0], y_grid=grid_y.T[0], power=1))
        interp_hyd.append(inverse_distance_weighting(hyd_points, hyd, x_grid=grid_x[0], y_grid=grid_y.T[0], power=1))
        interp_relHum.append(inverse_distance_weighting(relHum_points, relHum, x_grid=grid_x[0], y_grid=grid_y.T[0], power=1))
        interp_wind.append(inverse_distance_weighting(wind_points, wind, x_grid=grid_x[0], y_grid=grid_y.T[0], power=1))
    
    df_perc = pd.DataFrame()
    df_perc["date"] = dates
    df_perc["interpolated_perc"] = interp_perc

    df_temp = pd.DataFrame()
    df_temp["date"] = dates
    df_temp["interpolated_temp"] = interp_temp

    df_rad = pd.DataFrame()
    df_rad["date"] = dates
    df_rad["interpolated_rad"] = interp_rad

    df_hyd = pd.DataFrame()
    df_hyd["date"] = dates
    df_hyd["interpolated_hyd"] = interp_hyd

    df_relHum = pd.DataFrame()
    df_relHum["date"] = dates
    df_relHum["interpolated_relHum"] = interp_relHum

    df_wind = pd.DataFrame()
    df_wind["date"] = dates
    df_wind["interpolated_wind"] = interp_wind

    df_discharge = pd.DataFrame()
    df_discharge["date"] = dates
    df_discharge["discharge"] = list(np.array(list(disch_values.values())).T)

    # save the dataframe as a csv file 
    df_perc.to_json(perc_file_path, orient='values')
    df_discharge.to_json(disch_file_path, orient='values')
    df_temp.to_json(temp_file_path, orient='values')
    df_rad.to_json(rad_file_path, orient='values')
    df_hyd.to_json(hyd_file_path, orient='values')
    df_relHum.to_json(relHum_file_path, orient='values')
    df_wind.to_json(wind_file_path, orient='values')

    return

def inverse_distance_weighting(points, measurements, x_grid, y_grid, power):
    """
    Generate a 2D grid of interpolated values using Inverse Distance Weighting (IDW).
    :param points: Dictionary of (x, y) coordinates of the known data points.
    :param measurements: Dictionary of values at the known data points.
    :param power: Power parameter for IDW (higher values give more weight to closer points).
    :return: List of 2D grids of interpolated values.
    """
    # Replace invalid measurements (-99.0) with NaN
    points = np.array(points)
    measurements = np.array(measurements, dtype=float)
    measurements[measurements == -99.0] = np.nan

    xs = np.array(list(points)).T[0].T
    ys = np.array(list(points)).T[1].T
    zs = []

    for y in y_grid:
        for x in x_grid:
            distances = np.sqrt((x - xs)**2 + (y - ys)**2)
            weights = 1.0 / distances**power

            # Safeguard against division by zero or NaN
            denominator = np.nansum(weights)
            if denominator == 0 or np.isnan(denominator):
                z = np.nan  # Assign NaN or a default value
                zs.append(z)
            else:
                z = np.nansum(weights * measurements) / denominator
                zs.append(z)
           
            
    """ Alternative
    for i, y in enumerate(y_grid):
        for j, x in enumerate(x_grid):
            distances = np.sqrt((x - xs)**2 + (y - ys)**2)
            weights = 1 / (distances**power)
            weights /= weights.sum()
            new_grid[i, j] = np.dot(weights, measurements)
    """
    return np.array(zs).reshape(len(y_grid), len(x_grid))

def IDW(points, measurements, x_grid, y_grid, power):
    """
    Generate a 2D grid of interpolated values using Inverse Distance Weighting (IDW).
    :param points: Dictionary of (x, y) coordinates of the known data points.
    :param measurements: Dictionary of values at the known data points.
    :param power: Power parameter for IDW (higher values give more weight to closer points).
    :return: List of 2D grids of interpolated values.
    """
    # Replace invalid measurements (-99.0) with NaN
    measurements = np.array(measurements, dtype=float)
    measurements[measurements == -99.0] = np.nan

    xs = points.T[0].T
    ys = points.T[1].T
    zs = []

    for y in y_grid:
        for x in x_grid:
            distances = np.sqrt((x - xs)**2 + (y - ys)**2)
            weights = 1.0 / distances**power

            # Safeguard against division by zero or NaN
            denominator = np.nansum(weights)
            if denominator == 0 or np.isnan(denominator):
                z = np.nan  # Assign NaN or a default value
                zs.append(z)
            else:
                z = np.nansum(weights * measurements) / denominator
                zs.append(z)
           
            
    """ Alternative
    for i, y in enumerate(y_grid):
        for j, x in enumerate(x_grid):
            distances = np.sqrt((x - xs)**2 + (y - ys)**2)
            weights = 1 / (distances**power)
            weights /= weights.sum()
            new_grid[i, j] = np.dot(weights, measurements)
    """
    return np.array(zs).reshape(len(y_grid), len(x_grid))

def create_parameter_matrix(df_parameter, extent=(1, 1, 1, 1), spacing=100):
    """ 
    For each of the parameter values, create a grid of values, that a LSTM can alter the values of each cell
    and then the model can use these values to calculate the discharge using a physical model. """

    # Calculate the number of grid points based on the spacing
    num_points_x = int((extent[1] - extent[0]) / spacing) + 1
    num_points_y = int((extent[3] - extent[2]) / spacing) + 1

    # Initialize an empty grid for parameter values
    parameter_grid = np.zeros((num_points_y, num_points_x))
    
    # Iterate over each parameter and create a grid of values
    for param in df_parameter.columns:
        for cal in df_parameter[param]["calibration"]:
            parameter_grid = cal

    df_parameter[param]["grid"] = parameter_grid
    return df_parameter

def rst_to_geotiff(rst_path, geotiff_path, dtype=np.float32, nodata_value=None):
    """
    Convert a .rst raster file to a GeoTIFF file.

    Args:
        rst_path (str): Path to the input .rst file.
        geotiff_path (str): Path to the output GeoTIFF file.
        dtype (numpy dtype): Data type for the output raster.
        nodata_value (float or int, optional): NoData value to set in the output.

    Returns:
        None
    """
    # Read the .rst file using rasterio
    with rasterio.open(rst_path) as src:
        data = src.read(1)
        profile = src.profile.copy()
        # Optionally set nodata
        if nodata_value is not None:
            profile['nodata'] = nodata_value
        profile.update(driver='GTiff', dtype=dtype)

        # Write to GeoTIFF
        with rasterio.open(geotiff_path, 'w', **profile) as dst:
            dst.write(data.astype(dtype), 1)

def create_satellite_data(long_list, lat_list, start_date, end_date, width, height, extent, sat_file_paths):
    
    # Create a dataframe with the data
    data_folder_path = f"data/satellite_data/"
    satellite_temp = pd.DataFrame()
    satellite_prec = pd.DataFrame()

    # Check if the data folder contains any .json files for the given period
    has_data = any(
        file.startswith("UKMO_hourly_data_") and file.endswith(".json")
        for file in os.listdir(data_folder_path)
    )

    if not has_data:
        # Convert the latitudes and longitudes to lists
        for long, lat in zip(long_list, lat_list):
           params = {
               "latitude": lat,
               "longitude": long,
               "start_date": start_date,
               "end_date": end_date,
               "hourly": ["temperature_2m", "precipitation"],
               "models": "ecmwf_ifs025"
           }
           # Get the data from the UKMO API
           hourly_dataframe, _ = get_UKMO_data(params)
    
    
    #For hour create a row in the dataframe with the data
    point = 0
    timestamp = None
    for file in os.listdir(data_folder_path):
        if file.startswith("UKMO_hourly_data_") and file.endswith(".json"):
            file_path = os.path.join(data_folder_path, file)
            hourly_dataframe = pd.read_json(file_path, orient='values')                                
            # Add the data to the dataframe
            if timestamp is None:
                timestamp = hourly_dataframe[0]
            satellite_temp[point]  = hourly_dataframe[1]
            satellite_prec[point]  = hourly_dataframe[2]
            point += 1
    
    sat_points = list(zip(long_list, lat_list))
    
    dt = pd.to_datetime(timestamp, unit='ms', utc=True)
    # Format as "YYYY-MM-DD HH:MM:SS"
    formatted = dt.dt.strftime('%Y-%m-%d %H:%M:%S')
    prec = satellite_prec.values
    temp = satellite_temp.values
    interp_sat_temp = []
    interp_sat_perc = []
    grid_x, grid_y = np.meshgrid(np.linspace(extent[0], extent[1], width), np.linspace(extent[2], extent[3], height))
    
    for i in range(len(prec)):
        # Get the precipitation values for the ith point
        #find the indices if non-zero and non-NaN values
        prec_indices = np.where(~np.isnan(prec[i]) & (prec[i] != 0))[0]  
        temp_indices = np.where(~np.isnan(temp[i]) & (temp[i] != 0))[0]
        
        temp_points = np.array([sat_points[j] for j in temp_indices])
        prec_points = np.array([sat_points[k] for k in prec_indices])

        perc_array = np.array(prec[i][prec_indices])
        temp_array = np.array(temp[i][temp_indices])
        # Interpolate the precipitation values using IDW
        if prec_indices.size != 0:
            interp_sat_perc.append(IDW(prec_points, perc_array, x_grid=grid_x[0], y_grid=grid_y.T[0], power=2))
        else:
            interp_sat_perc.append(np.zeros((height, width)))
        
        # Interpolate the precipitation values using IDW
        if temp_indices.size != 0:
            interp_sat_temp.append(IDW(temp_points, temp_array, x_grid=grid_x[0], y_grid=grid_y.T[0], power=2))
        else:
            interp_sat_temp.append(np.zeros((height, width)))

    df_perc = pd.DataFrame({
        "date": formatted,
        "interpolated_perc": interp_sat_perc
    })
    df_perc = df_perc.set_index("date")
    df_temp = pd.DataFrame({
        "date": formatted,
        "interpolated_temp": interp_sat_temp
    })
    df_temp = df_temp.set_index("date")
    df_perc.to_json(sat_file_paths[0], orient='index')
    df_temp.to_json(sat_file_paths[1], orient='index')