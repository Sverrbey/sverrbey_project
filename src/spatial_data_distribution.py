import numpy as np
import pandas as pd

def reformat_data(prec_points, prec_values, temp_values, disch_points, disch_values, number_days, dates, perc_file_path,disch_file_path,temp_file_path, extent=(1, 1, 1, 1), spacing=100, power=1):
    
    # Calculate the number of grid points based on the spacing
    num_points_x = int((extent[1] - extent[0]) / spacing) + 1
    num_points_y = int((extent[3] - extent[2]) / spacing) + 1

    # Create a grid of points
    grid_x, grid_y = np.meshgrid(np.linspace(extent[0], extent[1], num_points_x), np.linspace(extent[2], extent[3], num_points_y))
    # For each row of input data, create a grid of interpolated values
    interp_temp = []
    interp_perc = []

    for n in range(0, number_days):
        perc = np.array(list(prec_values.values())).T[n]
        temp = np.array(list(temp_values.values())).T[n]
        interp_perc.append(inverse_distance_weighting(prec_points, perc, x_grid=grid_x[0], y_grid=grid_y.T[0], power=2))
        interp_temp.append(inverse_distance_weighting(prec_points, temp, x_grid=grid_x[0], y_grid=grid_y.T[0], power=1))
    
    df_perc = pd.DataFrame()
    df_perc["date"] = dates
    df_perc["interpolated_perc"] = interp_perc

    df_discharge = pd.DataFrame()
    df_discharge["date"] = dates
    df_discharge["discharge"] = list(np.array(list(disch_values.values())).T)

    df_temp = pd.DataFrame()
    df_temp["date"] = dates
    df_temp["interpolated_temp"] = interp_temp

    # save the dataframe as a csv file 
    df_perc.to_json(perc_file_path, orient='values')
    df_discharge.to_json(disch_file_path, orient='values')
    df_temp.to_json(temp_file_path, orient='values')

    return

def inverse_distance_weighting(data_points, measurements, x_grid, y_grid, power):
    """
    Generate a 2D grid of interpolated values using Inverse Distance Weighting (IDW).
    :param data_points: Dictionary of (x, y) coordinates of the known data points.
    :param measurements: Dictionary of values at the known data points.
    :param power: Power parameter for IDW (higher values give more weight to closer points).
    :return: List of 2D grids of interpolated values.
    """
    
    new_grid = np.zeros((len(x_grid), len(y_grid)))
    xs = np.array(list(data_points.values())).T[0].T
    ys = np.array(list(data_points.values())).T[1].T
    zs = []

    for y in y_grid:
        for x in x_grid:
            distances = np.sqrt((x - xs)**2 + (y - ys)**2)
            weights = 1.0 / distances**power
            z = np.sum(weights * measurements) / np.sum(weights)
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



