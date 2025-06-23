import numpy as np
import pandas as pd
import rasterio
import os
import rasterio
from rasterio.crs import CRS
from src.SAT_download import *
def reformat_data(points, prec_values, temp_values, disch_values, number_days, file_paths, evap_values=None, rad_values=None, relHum_values=None, wind_values=None, extent=(1, 1, 1, 1), spacing=7000):
    
    # Calculate the number of grid points based on the spacing
    width = int((extent[1] - extent[0]) / spacing) + 1
    height = int((extent[3] - extent[2]) / spacing) + 1

    # Create a grid of points
    x = np.linspace(extent[0], extent[1], width)
    y = np.linspace(extent[2], extent[3], height)
    grid_x, grid_y = np.meshgrid(x,y)

    # For each row of input data, create a grid of interpolated values
    interp_prec = []
    interp_temp = []
    interp_evap = []
    interp_rad = []
    interp_relHum = []
    interp_wind = []
    
    for n in range(0, number_days):
        prec = np.array(list(prec_values.values())).T[n]
        temp = np.array(list(temp_values.values())).T[n]
        if evap_values is not None:
            evap = np.array(list(evap_values.values())).T[n]
        if rad_values is not None:
            rad = np.array(list(rad_values.values())).T[n]
        if relHum_values is not None:
            relHum = np.array(list(relHum_values.values())).T[n]
        if wind_values is not None:
            wind = np.array(list(wind_values.values())).T[n]

        # Filter the values from data_points based on the keys in measurements
        prec_points = np.array([points[key] for key in prec_values.keys()])
        temp_points = np.array([points[key] for key in temp_values.keys()])
        if evap_values is not None:
            evap_points = np.array([points[key] for key in evap_values.keys()])
        if rad_values is not None:
            rad_points = np.array([points[key] for key in rad_values.keys()])
        if relHum_values is not None:
            relHum_points = np.array([points[key] for key in relHum_values.keys()])
        if wind_values is not None:
            wind_points = np.array([points[key] for key in wind_values.keys()])

        # Filter the points based on the values
        prec_points = prec_points[~np.isnan(prec) & (prec != 0)]
        temp_points = temp_points[~np.isnan(temp) & (temp != 0)]
        if evap_values is not None:
            evap_points = evap_points[~np.isnan(evap) & (evap != 0)]
        if rad_values is not None:
            rad_points = rad_points[~np.isnan(rad) & (rad != 0)]
        if relHum_values is not None:
            relHum_points = relHum_points[~np.isnan(relHum) & (relHum != 0)]
        if wind_values is not None:
            wind_points = wind_points[~np.isnan(wind) & (wind != 0)]
        
        # Filter the values from measurements based on the keys in data_points
        prec = prec[~np.isnan(prec) & (prec != 0)]
        temp = temp[~np.isnan(temp) & (temp != 0)]
        if evap_values is not None:
            evap = evap[~np.isnan(evap) & (evap != 0)]
        if rad_values is not None:
            rad = rad[~np.isnan(rad) & (rad != 0)]
        if relHum_values is not None:
            relHum = relHum[~np.isnan(relHum) & (relHum != 0)]
        if wind_values is not None:
            wind = wind[~np.isnan(wind) & (wind != 0)]

        # Interpolate the precipitation values using IDW
        if prec_points.size != 0:
            interp_prec.append(inverse_distance_weighting(prec_points, prec, x_grid=grid_x[0], y_grid=grid_y.T[0], power=2))
        else:
            interp_prec.append(np.full((height, width), np.nan))

        # Interpolate the precipitation values using IDW
        if temp_points.size != 0:
            interp_temp.append(inverse_distance_weighting(temp_points, temp, x_grid=grid_x[0], y_grid=grid_y.T[0], power=1))
        else:
            interp_temp.append(np.full((height, width), np.nan))

        # Interpolate the evaporation values using IDW
        if  evap_values is not None and evap_points.size != 0:
            interp_evap.append(inverse_distance_weighting(evap_points, evap, x_grid=grid_x[0], y_grid=grid_y.T[0], power=1))
        else:
            interp_evap.append(np.full((height, width), np.nan))

        # Interpolate the radiation values using IDW
        if rad_values is not None and rad_points.size != 0:
            interp_rad.append(inverse_distance_weighting(rad_points, rad, x_grid=grid_x[0], y_grid=grid_y.T[0], power=1))
        else:
            interp_rad.append(np.full((height, width), np.nan))
       
        # Interpolate the relative humidity values using IDW
        if relHum_values is not None and relHum_points.size != 0:
            interp_relHum.append(inverse_distance_weighting(relHum_points, relHum, x_grid=grid_x[0], y_grid=grid_y.T[0], power=1))
        else:
            interp_relHum.append(np.full((height, width), np.nan))
        # Interpolate the wind values using IDW
        if wind_values is not None and wind_points.size != 0:
            interp_wind.append(inverse_distance_weighting(wind_points, wind, x_grid=grid_x[0], y_grid=grid_y.T[0], power=1))
        else:
            interp_wind.append(np.full((height, width), np.nan))

    df_prec = pd.DataFrame({
        "interpolated_prec": interp_prec
    })

    df_temp = pd.DataFrame({
        "interpolated_temp": interp_temp
    })

    df_discharge = pd.DataFrame({
        "discharge": list(np.array(list(disch_values.values())).T)
    })
    # save the dataframe as a csv file 
    df_prec.to_json(file_paths[0], orient='index')
    df_temp.to_json(file_paths[1], orient='index')
    df_discharge.to_json(file_paths[2], orient='index')
    if rad_values is not None:
        df_rad = pd.DataFrame({
            "interpolated_rad": interp_rad
        })
        df_rad.to_json(file_paths[3], orient='index')

    if relHum_values is not None:
        df_relHum = pd.DataFrame({
            "interpolated_relHum": interp_relHum
        })
        df_relHum.to_json(file_paths[4], orient='index')
    if wind_values is not None:
        df_wind = pd.DataFrame({
            "interpolated_wind": interp_wind
        })
        df_wind.to_json(file_paths[5], orient='index')
    

    if evap_values is not None:
        df_evap = pd.DataFrame({
            "interpolated_evap": interp_evap
        })
        df_evap.to_json(file_paths[6], orient='index')

    return

def inverse_distance_weighting(points, measurements, x_grid, y_grid, power):
    """
    Generate a 2D grid of interpolated values using Inverse Distance Weighting (IDW).
    :param points: Dictionary of (x, y) coordinates of the known data points.
    :param points: Dictionary of (x, y) coordinates of the known data points.
    :param measurements: Dictionary of values at the known data points.
    :param power: Power parameter for IDW (higher values give more weight to closer points).
    :return: List of 2D grids of interpolated values.
    """

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
           
    return np.array(zs).reshape(len(y_grid), len(x_grid))

def IDW(points, measurements, x_grid, y_grid, power):
    """
    Generate a 2D grid of interpolated values using Inverse Distance Weighting (IDW).
    :param points: Dictionary of (x, y) coordinates of the known data points.
    :param measurements: Dictionary of values at the known data points.
    :param power: Power parameter for IDW (higher values give more weight to closer points).
    :return: List of 2D grids of interpolated values.
    """

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

def geotiff_to_rst(geotiff_path, rst_path, nodata_value=None, utm_zone=None, northern_hemisphere=True):
    """
    Convert a GeoTIFF file to a .rst raster file, optionally specifying UTM zone.
    """
    with rasterio.open(geotiff_path) as dataset:
        data = dataset.read(1)
        profile = dataset.profile.copy()
        # Optionally set nodata
        if nodata_value is not None:
            profile['nodata'] = nodata_value
        profile.update(driver='RST')
        # Optionally set UTM CRS
        if utm_zone is not None:
            epsg_code = 32600 + utm_zone if northern_hemisphere else 32700 + utm_zone
            profile['crs'] = CRS.from_epsg(epsg_code)
        # Write to .rst file
        with rasterio.open(rst_path, 'w', **profile) as dst:
            dst.write(data, 1)
                                                    
def create_satellite_data(long_list, long_list_DWD, lat_list, lat_list_DWD, width, width_DWD, height, height_DWD, start_date, end_date, extent, sat_file_paths):
    # Create a dataframe with the data
    data_folder_path = f"data/satellite_data/"
    satellite_temp  = pd.DataFrame()
    satellite_prec  = pd.DataFrame()
    satellite_dew   = pd.DataFrame()
    satellite_snow  = pd.DataFrame()

    #For hour create a row in the dataframe with the data
    point = 0
    timestamp = None
    for file in os.listdir(data_folder_path):
        if file.startswith("MET_") and file.endswith(".json"):
            file_path = os.path.join(data_folder_path, file)
            hourly_dataframe = pd.read_json(file_path, orient='values')                                
            # Add the data to the dataframe
            if timestamp is None:
                timestamp = hourly_dataframe[0]
            satellite_temp[point]  = hourly_dataframe[1]
            satellite_prec[point]  = hourly_dataframe[2]
            # Increment the point
            point += 1
    
    #For hour create a row in the dataframe with the data
    point = 0
    timestamp = None
    for file in os.listdir(data_folder_path):
        if file.startswith("DWD_") and file.endswith(".json"):
            file_path = os.path.join(data_folder_path, file)
            hourly_dataframe = pd.read_json(file_path, orient='values')                                
            # Add the data to the dataframe
            if timestamp is None:
                timestamp = hourly_dataframe[0]
            satellite_snow[point]  = hourly_dataframe[1]
            # Increment the point
            point += 1
    
    sat_points = list(zip(long_list, lat_list))
    
    dt = pd.to_datetime(timestamp, unit='ms', utc=True)
    # Format as "YYYY-MM-DD HH:MM:SS"
    formatted = dt.dt.strftime('%Y-%m-%d %H:%M:%S')
    dew  = satellite_dew.values
    prec = satellite_prec.values
    temp = satellite_temp.values
    snow = satellite_snow.values

    interp_sat_dew = []
    interp_sat_temp = []
    interp_sat_perc = []
    interp_sat_snow = []
    
    grid_x, grid_y = np.meshgrid(np.linspace(extent[0], extent[1], width), np.linspace(extent[2], extent[3], height))
    
    for i in range(len(prec)):
        # Get the precipitation values for the ith point
        #find the indices if non-zero and non-NaN values
        prec_indices = np.where(~np.isnan(prec[i]) & (prec[i] != 0))[0]  
        temp_indices = np.where(~np.isnan(temp[i]) & (temp[i] != 0))[0]
        dew_indices  = np.where(~np.isnan(dew[i]) & (dew[i] != 0))[0]
        snow_indices = np.where(~np.isnan(snow[i]) & (snow[i] != 0))[0]
        
        temp_points = np.array([sat_points[j] for j in temp_indices])
        prec_points = np.array([sat_points[k] for k in prec_indices])
        dew_points  = np.array([sat_points[l] for l in dew_indices])
        snow_points = np.array([sat_points[m] for m in snow_indices])

        perc_array = np.array(prec[i][prec_indices])
        temp_array = np.array(temp[i][temp_indices])
        dew_array  = np.array(dew[i][dew_indices])
        snow_array = np.array(snow[i][snow_indices])
        
        # Interpolate the precipitation values using IDW
        if prec_indices.size != 0:
            interp_sat_perc.append(IDW(prec_points, perc_array, x_grid=grid_x[0], y_grid=grid_y.T[0], power=1))
        else:
            interp_sat_perc.append(np.full((height, width), np.nan))
        
        # Interpolate the precipitation values using IDW
        if temp_indices.size != 0:
            interp_sat_temp.append(IDW(temp_points, temp_array, x_grid=grid_x[0], y_grid=grid_y.T[0], power=1))
        else:
            interp_sat_temp.append(np.full((height, width), np.nan))

        # Interpolate the dew point values using IDW
        if dew_indices.size != 0:
            interp_sat_dew.append(IDW(dew_points, dew_array, x_grid=grid_x[0], y_grid=grid_y.T[0], power=1))
        else:
            interp_sat_dew.append(np.full((height, width), np.nan))

        # Interpolate the snow depth values using IDW
        if snow_indices.size != 0:
            interp_sat_snow.append(IDW(snow_points, snow_array, x_grid=grid_x[0], y_grid=grid_y.T[0], power=1))
        else:
            interp_sat_snow.append(np.full((height, width), np.nan))

    df_perc = pd.DataFrame({
        "date": formatted,
        "interpolated_prec": interp_sat_perc
    })
    df_perc = df_perc.set_index("date")

    df_dew = pd.DataFrame({
        "date": formatted,
        "interpolated_dew": interp_sat_dew
    })
    df_dew = df_dew.set_index("date")

    df_temp = pd.DataFrame({
        "date": formatted,
        "interpolated_temp": interp_sat_temp
    })
    df_temp = df_temp.set_index("date")

    df_snow = pd.DataFrame({
        "date": formatted,
        "interpolated_snow": interp_sat_snow
    })
    df_snow = df_snow.set_index("date")

    df_perc.to_json(sat_file_paths[0], orient='index')
    df_temp.to_json(sat_file_paths[1], orient='index')
    df_dew.to_json(sat_file_paths[2], orient='index')
    df_snow.to_json(sat_file_paths[3], orient='index')
    return

def download_sequence1(long_list, lat_list, start_date, end_date):
    for long, lat in zip(long_list, lat_list):
        params = {
            "latitude": lat,
            "longitude": long,
            "start_date": start_date,
            "end_date": end_date,
            "hourly": ["temperature_2m", "precipitation"],
            "models": "metno_seamless"
        }
        # Get the data from the Openmeteo API
        
        get_MET_data(params)

def download_sequence2(long_list, lat_list, start_date, end_date):
    for long, lat in zip(long_list, lat_list):
        params = {
            "latitude": lat,
            "longitude": long,
            "start_date": start_date,
            "end_date": end_date,
            "hourly": ["snow_depth"],
            "models": "icon_seamless"
        }
        # Get the data from the Openmeteo API
        get_DWD_data(params)