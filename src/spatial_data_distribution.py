import numpy as np
import pandas as pd

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
    :param data_points: Dictionary of (x, y) coordinates of the known data points.
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
            z = np.nansum(weights * measurements) / np.nansum(weights)
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



