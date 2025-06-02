import numpy as np
import matplotlib.pyplot as plt
import torch
import seaborn as sns
import rasterio
from rasterio.plot import show
from rasterio.features import shapes
from shapely.geometry import shape, Point, Polygon

import geopandas as gpd
import contextily as ctx
import matplotlib.ticker as mticker
from matplotlib.patches import ConnectionPatch

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

def plot_predictions_vs_actuals(model, dataloader, scaler_y, scaler_x, name):
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
        for i in range(min(4, actuals.shape[1])):  # Plot up to 4 features
            plt.plot(actuals[:, i], label=f'Actual Data (Feature {i + 1})')
            plt.plot(predictions[:, i], label=f'Predicted Data (Feature {i + 1})', linestyle='--')

    plt.xlabel('Time Step [days]')
    plt.ylabel('Value')
    plt.title(name +': Model Predictions vs Actual Data')
    plt.legend()
    plt.show()

def plot_predictions_vs_actuals_enhanced(model, dataloader, name):
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
    plt.title(title + ': 2D Interpolated Data Representation using IDW : Day ' + str(day))
    plt.grid()
    plt.savefig(path + '/' + title + '_day_' + str(day) + '.png')

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


def EDA(EDA_prec, EDA_temp, EDA_disc):

    return
    

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

def create_catchment_map(catchment_name, catchment_geotiff_path, met_stations, hydro_stations, country_shapefile):
    """
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

    # Read continent shapefile (e.g., Europe)
    country_gdf = gpd.read_file(country_shapefile).to_crs(epsg=4326)

    return catchment_gdf, met_gdf, hydro_gdf, country_gdf

def plot_catchment_map(catchment_gdf, met_gdf, hydro_gdf, catchment_name="Catchment"):
    
    # Plotting
    fig, ax = plt.subplots(figsize=(10, 6))

    #catchment_gdf.plot(ax=ax, color='lightblue', edgecolor='blue', alpha=0.4, label="Catchment Area")
    catchment_gdf.to_crs(epsg=4326).plot(ax=ax, color='none', edgecolor='blue', alpha=1, label="Catchment Area")
    #met_gdf.to_crs(epsg=4326).plot(ax=ax, color='red', marker='^', markersize=100, label='Meteorological Stations')
    #hydro_gdf.to_crs(epsg=4326).plot(ax=ax, color='green', marker='o', markersize=100, label='Hydrological Stations')

    
    # Add basemap
    #ctx.add_basemap(ax, source=ctx.providers.OpenTopoMap)

    # Add title and legend
    ax.set_title(f"{catchment_name} Catchment Area (Norway)", fontsize=14)
    ax.legend()
    #ax.axis('off')

    plt.show()

def plot_country_and_catchment_zoom(country_gdf, stryn_catchment_gdf, stryn_met_gdf, stryn_hydro_gdf, gaula_gdf, gaula_met_gdf, gaula_hydro_gdf):
    """ 
    Plot the country with the catchment area highlighted, and a zoomed-in plot of the catchment.
    Lines connect the catchment area on the country map to the zoomed-in plot.
    """
    # Ensure both are in the same CRS (EPSG:4326 for lat/lon)
    country_gdf = country_gdf.to_crs(epsg=4326)
    stryn_catchment_gdf = stryn_catchment_gdf.to_crs(epsg=4326)
    gaula_gdf = gaula_gdf.to_crs(epsg=4326)


    # Get bounds of the catchment for zoom
    s_minx, s_miny, s_maxx, s_maxy = stryn_catchment_gdf.total_bounds
    g_minx, g_miny, g_maxx, g_maxy = gaula_gdf.total_bounds


    fig = plt.figure(figsize=(10, 6))
    ax_country = fig.add_axes([0.05, 0.05, 0.4, 0.8])
    ax_1 = fig.add_axes([0.55, 0.05, 0.4, 0.425])   # Lower zoomed-in plot
    ax_2 = fig.add_axes([0.55, 0.55, 0.4, 0.425])   # Upper zoomed-in plot

    # Main country plot
    country_gdf.plot(ax=ax_country, color='none', edgecolor='gray')
    stryn_catchment_gdf.plot(ax=ax_country, color='none', edgecolor='red', linewidth=2, label="Stryn")
    gaula_gdf.plot(ax=ax_country, color='none', edgecolor='red', linewidth=2, label="Gaula")
    ctx.add_basemap(ax_country, source=ctx.providers.OpenTopoMap, zoom=5, crs=stryn_catchment_gdf.crs)
    #ax_country.set_title(f"{catchment_name} in Norway")
    # Format axes as DMS
    ax_country.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, pos:deg2dms(x, 0)))
    ax_country.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, pos:deg2dms(x, 1)))
    ax_country.set_ylim(54, 75)  # Adjust y-limits for better visibility
    

    # Draw lines (ConnectionPatch) from country to zoom
    # Corners of the catchment bounding box
    corners = [(s_minx, s_miny), (s_minx, s_maxy)] #, (maxx, miny), (maxx, maxy)
    for (x, y) in corners:
        con = ConnectionPatch(
            xyA=(x, y), coordsA=ax_1.transData,
            xyB=(x, y), coordsB=ax_country.transData,
            color="red", linewidth=1, linestyle="--"
        )
        fig.add_artist(con)

    stryn_catchment_gdf.plot(ax=ax_1, color='none', edgecolor='blue', alpha=0.4)
    ctx.add_basemap(ax_1, source=ctx.providers.OpenTopoMap, crs=stryn_catchment_gdf.crs)
    stryn_met_gdf.to_crs(epsg=4326).plot(ax=ax_1, color='red', marker='^', markersize=50)
    stryn_hydro_gdf.to_crs(epsg=4326).plot(ax=ax_1, color='blue', marker='o', markersize=50)

    corners = [(g_minx, g_miny), (g_minx, g_maxy)] #, (maxx, miny), (maxx, maxy)
    for (x, y) in corners:
        con = ConnectionPatch(
            xyA=(x, y), coordsA=ax_2.transData,
            xyB=(x, y), coordsB=ax_country.transData,
            color="red", linewidth=1, linestyle="--"
        )
        fig.add_artist(con)

    gaula_gdf.plot(ax=ax_2, color='none', edgecolor='blue', alpha=0.4)
    ctx.add_basemap(ax_2, source=ctx.providers.OpenTopoMap, crs=gaula_gdf.crs)
    gaula_met_gdf.to_crs(epsg=4326).plot(ax=ax_2, color='red', marker='^', markersize=50, label='Meteorological Stations')
    gaula_hydro_gdf.to_crs(epsg=4326).plot(ax=ax_2, color='blue', marker='o', markersize=50, label='Hydrological Stations')

    # Format axes as DMS
    ax_1.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, pos:deg2dms_v2(x, 0)))
    ax_1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, pos:deg2dms_v2(x, 1)))
    ax_1.set_xlim(s_minx, s_maxx)
    ax_1.set_ylim(s_miny, s_maxy)
    ax_1.set_title(f"Stryn Catchment Area")
    
    ax_2.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, pos:deg2dms_v2(x, 0)))
    ax_2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, pos:deg2dms_v2(x, 1)))
    #ax_2.set_xlim(g_minx, g_maxx)
    #ax_2.set_ylim(g_miny, g_maxy)
    ax_2.set_title(f"Gaula Catchment Area")

    ax_1.tick_params(axis='x')
    ax_2.tick_params(axis='x', rotation=15)
    
    handles, labels = ax_2.get_legend_handles_labels()
    fig.legend(handles, labels, bbox_to_anchor=(0.25, 0.9))
    plt.tight_layout()
    plt.show()

