import numpy as np
import matplotlib.pyplot as plt
import torch
import cv2
import seaborn as sns
import rasterio
from rasterio.plot import show
from rasterio.features import shapes
from shapely.geometry import shape, Point, Polygon

from rasterio.features import shapes
from shapely.geometry import shape, Point, Polygon

import geopandas as gpd
import contextily as ctx
import matplotlib.ticker as mticker
from matplotlib.patches import ConnectionPatch
import matplotlib.ticker as mticker
from matplotlib.patches import ConnectionPatch

from src.helper_functions import *
from src.helper_functions import *

def plot_sensitivity(model, dataloader, scaler_y, number):
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
    
    # Ensure predictions have the same shape as the scaler's expected input
    if predictions.shape[1] != scaler_y.min_.shape[0]:
        raise ValueError(f"Predictions shape {predictions.shape} does not match scaler's expected shape {scaler_y.min_.shape}")

    # Inverse transform the predictions and actuals to the original scale
    predictions = scaler_y.inverse_transform(predictions)
    actuals = scaler_y.inverse_transform(actuals)
    
    plt.plot(actuals[:, 0], color='red')
    plt.plot(predictions[:, 0], label=f'Predicted Data [{2**(3+number)}]', linestyle='--')
    if actuals.shape[1] > 1:  # If there are multiple target features
        plt.plot(actuals[:, 1], color='red')
        plt.plot(predictions[:, 1], label=f'Predicted Data [{2**(3+number)}]', linestyle='--')

def plot_predictions_vs_actuals(model, dataloader, scaler_y, name):
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
    
    # Ensure predictions have the same shape as the scaler's expected input
    if predictions.shape[1] != scaler_y.min_.shape[0]:
        raise ValueError(f"Predictions shape {predictions.shape} does not match scaler's expected shape {scaler_y.min_.shape}")

    # Inverse transform the predictions and actuals to the original scale
    predictions = scaler_y.inverse_transform(predictions)
    actuals = scaler_y.inverse_transform(actuals)
    
    
    # Plot the predictions vs actuals
    plt.figure(figsize=(10, 6))
    plt.plot(actuals[:, 0], label='Actual Data (Feature 1)')
    plt.plot(predictions[:, 0], label='Predicted Data (Feature 1)', linestyle='--')
    if actuals.shape[1] > 1:  # If there are multiple target features
        for i in range(min(6, actuals.shape[1])):  # Plot up to 6 features
            plt.plot(actuals[:, i], label=f'Actual Data (Feature {i + 1})')
            plt.plot(predictions[:, i], label=f'Predicted Data (Feature {i + 1})', linestyle='--')

    plt.xlabel('Time Step [days]')
    plt.ylabel('Value')
    plt.title(name +': Model Predictions vs Actual Data')
    plt.legend()
    plt.show()

def plot_predictions_vs_actuals_multi(model, dataloader, scaler_y, name):
    model.eval()
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for *batch_x, batch_y in dataloader:
            outputs = model(*batch_x)
            predictions.append(outputs.cpu().numpy())
            actuals.append(batch_y.cpu().numpy())
    
    # Concatenate predictions and actuals along the first axis
    predictions = np.concatenate(predictions, axis=0)  # Shape: (num_samples, output_size)
    actuals = np.concatenate(actuals, axis=0)          # Shape: (num_samples, output_size)
    
    # Plot the predictions vs actuals
    plt.figure(figsize=(10, 6))
    plt.plot(actuals[:, 0], label='Actual Data (Feature 1)')
    plt.plot(predictions[:, 0], label='Predicted Data (Feature 1)', linestyle='--')
    if actuals.shape[1] > 1:  # If there are multiple target features
        for i in range(min(4, actuals.shape[1])):  # Plot up to 4 features
            plt.plot(actuals[:, i], label=f'Actual Data (Feature {i + 1})')
            plt.plot(predictions[:, i], label=f'Predicted Data (Feature {i + 1})', linestyle='--')

    plt.xlabel('Time Step [days]')
    plt.ylabel('Value')
    plt.title(name +': Model Predictions vs Actual Data')
    plt.legend()
    plt.show()

def plot_interpolation(grid_list, day, points, path, extent, title):

    # Plot the interpolated grid for the first time step
    plt.rcParams["figure.figsize"] = (20, 7)
    grid_plot = grid_list[day-1]
    grid_plot = np.array(grid_plot, dtype=np.float64)
    plt.imshow(grid_plot, extent=(extent[0], extent[1], extent[2], extent[3]), origin='lower', cmap='viridis')
    #plt.colorbar(label='Interpolated Value')
    plt.scatter(*zip(*points.values()), color='red', label='Known Data Points')
    #plt.scatter(*zip(*disch_points.values()), color='blue', label='Discharge Stations')
    # Annotate each scatter point with its corresponding number
    for key, (x, y) in points.items():
        plt.text(x, y, key, fontsize=12, ha='right', color='white')
    #plt.legend(loc='upper right')
    plt.title(title + ': Data Distribution : Day ' + str(day))
    plt.grid()
    plt.show()
    #plt.savefig(path + '/' + title + '_day_' + str(day) + '.png')

def plot_interpolation_routing(grid_list, extent, title):
    plt.rcParams["figure.figsize"] = (20, 7)
    grid_plot = grid_list.detach().cpu().numpy()
    grid_plot = np.array(grid_plot, dtype=np.float64)
    plt.imshow(grid_plot, extent=(extent[0], extent[1], extent[2], extent[3]), origin='lower', cmap='viridis')
    #plt.colorbar(label='Interpolated Value')
    plt.title(title + ': 2D Routing Data from HBV Model')
    plt.grid()
    #plt.savefig(path + '/' + title + '.png')
    plt.show()
    
def deg2dms(x, pos):
    """Convert decimal degrees to DMS string with N/S/E/W."""
    degrees = int(abs(x))
    minutes = int(abs(x - int(x)) * 60)
    seconds = int((abs(x - int(x)) * 60 - minutes) * 60)
    # Determine direction
    if pos == 0:  # x-axis (longitude)
        direction = 'E' if x >= 0 else 'W'
    else:         # y-axis (latitude)
        direction = 'N' if x >= 0 else 'S'
    return f"{degrees}° {direction}"

def deg2dms_v2(x, pos):
    """Convert decimal degrees to DMS string with N/S/E/W."""
    degrees = int(abs(x))
    minutes = int(abs(x - int(x)) * 60)
    seconds = int((abs(x - int(x)) * 60 - minutes) * 60)
    # Determine direction
    if pos == 0:  # x-axis (longitude)
        direction = 'E' if x >= 0 else 'W'
    else:         # y-axis (latitude)
        direction = 'N' if x >= 0 else 'S'
    return f"{degrees}°{minutes:02d}' {direction}"

def create_catchment_map(catchment_name, catchment_geotiff_path, met_stations, hydro_stations):
    """
    Create GeoDataFrames for catchment area, meteorological stations, and hydrological stations
    using a GeoTIFF raster for the catchment area.

    Create GeoDataFrames for catchment area, meteorological stations, and hydrological stations
    using a GeoTIFF raster for the catchment area.

    Parameters:
        catchment_name (str): Name of the catchment.
        catchment_geotiff_path (str): Path to the GeoTIFF file representing the catchment mask.
        met_stations (dict): {'name1': (lon, lat), ...}
        hydro_stations (dict): {'name1': (lon, lat), ...}

    Returns:
        catchment_gdf (GeoDataFrame): Catchment area polygons.
        met_gdf (GeoDataFrame): Meteorological stations.
        hydro_gdf (GeoDataFrame): Hydrological stations.
    """
    with rasterio.open(catchment_geotiff_path) as src:
        raster = src.read(1)
        mask = raster != src.nodata

        # Extract polygons from the raster mask
        results = (
            {'properties': {'raster_val': v}, 'geometry': s}
            for s, v in shapes(raster, mask=mask, transform=src.transform)
            if v != src.nodata
        )
        polygons = [shape(feature['geometry']) for feature in results]
        catchment_gdf = gpd.GeoDataFrame(
            {'name': [catchment_name] * len(polygons)},
            geometry=polygons,
            crs=src.crs
        )

    # Reproject to EPSG:4326 if needed
    if catchment_gdf.crs != "EPSG:4326":
        catchment_gdf = catchment_gdf.to_crs("EPSG:4326")
        catchment_name (str): Name of the catchment.
        catchment_geotiff_path (str): Path to the GeoTIFF file representing the catchment mask.
        met_stations (dict): {'name1': (lon, lat), ...}
        hydro_stations (dict): {'name1': (lon, lat), ...}

    Returns:
        catchment_gdf (GeoDataFrame): Catchment area polygons.
        met_gdf (GeoDataFrame): Meteorological stations.
        hydro_gdf (GeoDataFrame): Hydrological stations.
    """
    with rasterio.open(catchment_geotiff_path) as src:
        raster = src.read(1)
        mask = raster != src.nodata

        # Extract polygons from the raster mask
        results = (
            {'properties': {'raster_val': v}, 'geometry': s}
            for s, v in shapes(raster, mask=mask, transform=src.transform)
            if v != src.nodata
        )
        polygons = [shape(feature['geometry']) for feature in results]
        catchment_gdf = gpd.GeoDataFrame(
            {'name': [catchment_name] * len(polygons)},
            geometry=polygons,
            crs=src.crs
        )

    # Reproject to EPSG:4326 if needed
    if catchment_gdf.crs != "EPSG:4326":
        catchment_gdf = catchment_gdf.to_crs("EPSG:4326")

    # Create GeoDataFrame for meteorological stations
    met_gdf = gpd.GeoDataFrame(
        {'name': list(met_stations.keys())},
        geometry=[Point(coord) for coord in met_stations.values()],
        crs="EPSG:4326"
    )
    # Create GeoDataFrame for meteorological stations
    met_gdf = gpd.GeoDataFrame(
        {'name': list(met_stations.keys())},
        geometry=[Point(coord) for coord in met_stations.values()],
        crs="EPSG:4326"
    )

     # Create GeoDataFrame for meteorological stations
    hydro_gdf = gpd.GeoDataFrame(
        {'name': list(hydro_stations.keys())},
        geometry=[Point(coord) for coord in hydro_stations.values()],
        crs="EPSG:4326"
    )


    return catchment_gdf, met_gdf, hydro_gdf

def plot_cubic_interpolation(height=30, width=108):
    y = np.arange(height)
    x = np.arange(width)
    xx, yy = np.meshgrid(x, y)
    # Create an empty grid
    image_test = np.zeros((height, width))

    # Add random Gaussian blobs
    num_blobs = 10
    np.random.seed(42)  # For reproducibility
    for _ in range(num_blobs):
        cx = np.random.randint(0, width)
        cy = np.random.randint(0, height)
        sigma = np.random.uniform(2, 6)
        amplitude = np.random.uniform(5, 15)
        # Create a Gaussian blob
        blob = amplitude * np.exp(-(((xx - x[cx])**2 + (yy - y[cy])**2) / (2 * sigma**2)))
        image_test += blob

    new_width, new_height = 16, 5
    arr_resized = cv2.resize(image_test, (new_width, new_height), interpolation=cv2.INTER_CUBIC)       

    # Plot the original and resized images side by side
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle('Before and After Cubic Interpolation', fontsize=16)
    # Original image
    im0 = axes[0].imshow(image_test, cmap='viridis', origin='lower', aspect='auto')
    axes[0].set_title('Spatial Distribution (1km resolution)')
    plt.colorbar(im0, ax=axes[0], label='Value')

    # Resized image
    im1 = axes[1].imshow(arr_resized, cmap='viridis', origin='lower', aspect='auto')
    axes[1].set_title('Resized Image (7km resolution)')
    plt.colorbar(im1, ax=axes[1], label='Value')

    plt.tight_layout()
    plt.show()

def plot_flood_hydrograph():
    test_precipitaton = np.array([[0.0, 0.0],
                             [0.0, 0.0],
                             [0.0, 0.0],
                             [0.0, 0.0],
                             [6.686111, 61.736389],
                             [0.0, 0.0],
                             [6.705833, 61.756944],
                             [20.20, 61.756944],
                             [0.0, 0.0],
                             [0.0, 0.0],
                             [0.0, 0.0],
                             [0.0, 0.0],
                             [0.0, 0.0],
                             [0.0, 0.0],
                             [0.0, 0.0],
                             [0.0, 0.0],
                             [0.0, 0.0],
                             [0.0, 0.0],
                             [0.0, 0.0]])
    
    hydrographs = np.zeros_like(test_precipitaton)
    unit_h = np.array([0.3,0.5,0.7,1.7,1.6,1.3,1.2,1.15,1.05,0.6,0.4])  # Example unit hydrograph values
    unit_h = unit_h / unit_h.sum()  # Ensure it sums to 1
    window_size = len(unit_h)
    
    flood_indices = [4, 5, 6, 7]
    for idx in flood_indices:
        start = idx
        end = min(test_precipitaton.shape[0], idx + window_size)
        actual_window = end - start
        for i in range(test_precipitaton.shape[1]):
            # Adjust unit_h if at the edge
            h = unit_h[:actual_window]
            hydrographs[start:end, i] += test_precipitaton[idx, i] * h

    adjusted_prec = test_precipitaton.copy()
    adjusted_indices = np.where(hydrographs > 0)
    adjusted_prec[adjusted_indices] = hydrographs[adjusted_indices]


    plt.figure(figsize=(10, 6))
    x = np.arange(test_precipitaton.shape[0])
    plt.bar(x, test_precipitaton[:,0], color='blue', alpha=0.3, label='Precipitation')
    #plt.bar(x, test_precipitaton[:,1], color='blue', alpha=1, label='Precipitation cell 2')
    plt.bar(x, adjusted_prec[:, 0],  color='c', alpha=0.3, label='Hydrograph')
    #plt.bar(x, adjusted_prec[:, 1],  color='purple', alpha=0.3, label='Hydrograph cell 2')
    plt.title('Flood Hydrograph')
    plt.xlabel('Time Step')
    plt.ylabel('Value (mm)')
    plt.xticks(range(len(adjusted_prec)), [f'Hour {i+1}' for i in range(len(adjusted_prec))])
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()