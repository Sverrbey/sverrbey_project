import numpy as np
import matplotlib.pyplot as plt
import torch
import seaborn as sns
import rasterio
from rasterio.plot import show
import geopandas as gpd
from shapely.geometry import Point, Polygon
import contextily as ctx


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

    # Display basic statistics
    print("Combined Data Statistics:")
    print(combined_df.describe())

    print("Target Data Statistics:")
    print(y_df.describe())
    # Write combined_df and y_df to CSV files

    combined_df.to_csv("combined_data.csv", index=False)
    y_df.to_csv("target_data.csv", index=False)

    # Plot a few time series from combined_data
    plt.figure(figsize=(12, 6))
    plt.plot(combined_data[:, 0], label="Precipitation")
    plt.plot(combined_data[:, 1], label="Temperature")
    plt.legend()
    plt.title("Time Series of Precipitation and Temperature")
    plt.xlabel("Time Steps")
    plt.ylabel("Values")
    plt.show()

    # Plot target data (discharge)
    plt.figure(figsize=(12, 6))
    plt.plot(y_data, label="Discharge")
    plt.legend()
    plt.title("Time Series of Discharge")
    plt.xlabel("Time Steps")
    plt.ylabel("Discharge")
    plt.show()

    ## Plot histograms for combined_data
    #combined_df.hist(bins=30, figsize=(15, 10))
    #plt.suptitle("Histograms of Combined Data Features")
    #plt.show()
    #
    ## Plot histogram for target data
    #y_df.hist(bins=30, figsize=(6, 4))
    #plt.suptitle("Histogram of Target Data (Discharge)")
    #plt.show()
    #
    # Check for missing values
    print("Missing values in combined_data:", np.isnan(combined_data).sum())
    print("Missing values in y_data:", np.isnan(y_data).sum())

    # Check for specific invalid values (e.g., -99)
    print("Invalid values (-99) in combined_data:", np.sum(combined_data == -99))
    print("Invalid values (-99) in y_data:", np.sum(y_data == -99))
    #
    ## Compute correlation matrix
    #correlation_matrix = combined_df.corr()
    #
    ## Plot correlation heatmap
    #plt.figure(figsize=(10, 8))
    #sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
    #plt.title("Correlation Matrix of Combined Data")
    #plt.show()

    # Scatter plot between features and target
    for i in range(combined_data.shape[1]):
        plt.figure(figsize=(6, 4))
        plt.scatter(combined_data[:, i], y_data[:], alpha=0.5)
        plt.title(f"Feature {i+1} vs Discharge")
        plt.xlabel(f"Feature {i+1}")
        plt.ylabel("Discharge")
        plt.show()

    # Visualize a sample sequence
    seq_length = 7  # Example sequence length
    sample_sequence = combined_data[:seq_length]

    plt.figure(figsize=(12, 6))
    for i in range(sample_sequence.shape[1]):
        plt.plot(sample_sequence[:, i], label=f"Feature {i+1}")
    plt.legend()
    plt.title("Sample Sequence")
    plt.xlabel("Time Steps")
    plt.ylabel("Values")
    plt.show()


def plot_catchment_map(catchment_name, catchment_coords, met_stations, hydro_stations):
    """
    Plot catchment with meteorological and hydrological stations.
    
    Parameters:
        catchment_name (str): Name of the catchment
        catchment_coords (list of tuples): Coordinates [(lon, lat), ...] forming the catchment polygon
        met_stations (dict): {'name1': (lon, lat), 'name2': (lon, lat), ...}
        hydro_stations (dict): {'name1': (lon, lat), 'name2': (lon, lat), ...}
    """

    # Create GeoDataFrame for Catchment Polygon
    catchment_poly = Polygon(catchment_coords)
    catchment_gdf = gpd.GeoDataFrame({'name': [catchment_name]}, geometry=[catchment_poly], crs="EPSG:4326")

    # Create GeoDataFrame for Met Stations
    met_gdf = gpd.GeoDataFrame({
        'name': list(met_stations.keys()),
        'geometry': [Point(coord) for coord in met_stations.values()]
    }, crs="EPSG:4326")

    # Create GeoDataFrame for Hydro Stations
    hydro_gdf = gpd.GeoDataFrame({
        'name': list(hydro_stations.keys()),
        'geometry': [Point(coord) for coord in hydro_stations.values()]
    }, crs="EPSG:4326")

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 10))

    catchment_gdf.to_crs(epsg=3857).plot(ax=ax, color='lightblue', edgecolor='blue', alpha=0.4, label="Catchment Area")
    met_gdf.to_crs(epsg=3857).plot(ax=ax, color='red', marker='^', markersize=100, label='Meteorological Stations')
    hydro_gdf.to_crs(epsg=3857).plot(ax=ax, color='green', marker='o', markersize=100, label='Hydrological Stations')

    # Annotate stations
    for idx, row in met_gdf.iterrows():
        ax.annotate(row['name'], xy=(row.geometry.x, row.geometry.y), xytext=(3, 3), textcoords='offset points', fontsize=9)

    for idx, row in hydro_gdf.iterrows():
        ax.annotate(row['name'], xy=(row.geometry.x, row.geometry.y), xytext=(3, -10), textcoords='offset points', fontsize=9)

    # Add basemap
    ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)

    # Add title and legend
    ax.set_title(f"{catchment_name} Catchment Area (Norway)", fontsize=14)
    ax.legend()
    ax.axis('off')

    plt.show()


