"""
This python file uses data exported from 'hoydedata.no' in order to create the nodes in a graphical
representation of a catchment. This serves as a kind of input modulation of the geospatial data, so
that the computational complexity isn't unreasonably big or small.
"""


import rasterio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import Delaunay
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.patches as patches
import random

def read_tfw(file_path):
    """
    Reads a TFW file and displays the georeferencing information.

    Parameters:
        file_path (str): Path to the .tfw file.
    """
    try:
        with open(file_path, 'r') as file:
            # Read all lines from the file
            lines = file.readlines()

            if len(lines) != 6:
                raise ValueError("TFW file does not have the expected 6 lines.")

            # Extract georeferencing values
            x_pixel_size = float(lines[0].strip())  # Pixel size in X direction
            rotation_x = float(lines[1].strip())  # Rotation (usually 0)
            rotation_y = float(lines[2].strip())  # Rotation (usually 0)
            y_pixel_size = float(lines[3].strip())  # Pixel size in Y direction (negative for North-up images)
            x_coordinate = float(lines[4].strip())  # X coordinate of the upper-left corner
            y_coordinate = float(lines[5].strip())  # Y coordinate of the upper-left corner

            # Display the information
            print("TFW File Information:")
            print(f"Pixel Size (X): {x_pixel_size}")
            print(f"Pixel Size (Y): {y_pixel_size}")
            print(f"Rotation X: {rotation_x}")
            print(f"Rotation Y: {rotation_y}")
            print(f"Upper-Left Corner (X): {x_coordinate}")
            print(f"Upper-Left Corner (Y): {y_coordinate}")
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except ValueError as e:
        print(f"Error reading TFW file: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


def extract_altitude_points(tif_file_path, output_csv=None):
    """
    Extract altitude points from a georeferenced TIFF file and optionally save them to a CSV file.

    Parameters:
        tif_file_path (str): Path to the .tif file.
        output_csv (str): Path to save the extracted altitude points as a CSV file (optional).

    Returns:
        DataFrame: A DataFrame containing the X, Y coordinates and altitude values.
    """
    try:
        # Open the TIFF file
        with rasterio.open(tif_file_path) as src:
            # Read the raster data
            data = src.read(1)  # Read the first band (assumes single-band DEM)
            transform = src.transform  # Georeferencing transform

            # Get the dimensions of the raster
            rows, cols = data.shape

            # Create arrays for row and column indices
            row_indices, col_indices = np.indices((rows, cols))

            # Convert pixel indices to spatial coordinates
            xs, ys = rasterio.transform.xy(transform, row_indices, col_indices, offset='center')

            # Flatten the arrays for easier processing
            xs = np.array(xs).flatten()
            ys = np.array(ys).flatten()
            altitudes = data.flatten()

            # Mask out nodata values
            nodata = src.nodata
            if nodata is not None:
                valid_mask = altitudes != nodata
                xs = xs[valid_mask]
                ys = ys[valid_mask]
                altitudes = altitudes[valid_mask]

            # Create a DataFrame to store the altitude points
            altitude_points = pd.DataFrame({
                'X': xs,
                'Y': ys,
                'Altitude': altitudes
            })

            # Save to CSV if output path is provided
            if output_csv:
                altitude_points.to_csv(output_csv, index=False)
                print(f"Altitude points saved to {output_csv}")

            return altitude_points

    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def cluster_elevation_data(csv_file_path, n_clusters=3, output_csv=None):
    """
    Clusters elevation data from a CSV file and visualizes the clusters.

    Parameters:
        csv_file_path (str): Path to the CSV file containing X, Y, and Altitude columns.
        n_clusters (int): Number of clusters for K-Means clustering.
        output_csv (str): Path to save the clustered data as a CSV file (optional).

    Returns:
        DataFrame: DataFrame with cluster labels appended.
    """
    try:
        # Load the CSV file
        df = pd.read_csv(csv_file_path)

        # Check for required columns
        if not {'X', 'Y', 'Altitude'}.issubset(df.columns):
            raise ValueError("The CSV file must contain 'X', 'Y', and 'Altitude' columns.")

        # Extract features (X, Y, Altitude)
        features = df[['X', 'Y', 'Altitude']].values

        # Standardize the features
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features)

        # Perform K-Means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        df['Cluster'] = kmeans.fit_predict(scaled_features)

        # Visualize the clusters
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(df['X'], df['Y'], c=df['Cluster'], cmap='viridis', s=10)
        plt.colorbar(scatter, label='Cluster')
        plt.title(f"Elevation Data Clustering (k={n_clusters})")
        plt.xlabel("X Coordinate")
        plt.ylabel("Y Coordinate")
        plt.grid()
        plt.show()

        # Save to CSV if output path is provided
        if output_csv:
            df.to_csv(output_csv, index=False)
            print(f"Clustered data saved to {output_csv}")

        return df

    except FileNotFoundError:
        print(f"File not found: {csv_file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")


def create_tin_from_clustered_csv(clustered_csv_file_path, output_csv=None):
    """
    Creates and visualizes a TIN (Triangulated Irregular Network) from a clustered CSV file.

    Parameters:
        clustered_csv_file_path (str): Path to the clustered CSV file containing X, Y, Altitude, and Cluster columns.
        output_csv (str): Path to save the simplified TIN points as a CSV file (optional).

    Returns:
        delaunay (scipy.spatial.Delaunay): Delaunay triangulation object.
    """
    try:
        # Load the clustered CSV file
        df = pd.read_csv(clustered_csv_file_path)

        # Check for required columns
        if not {'X', 'Y', 'Altitude', 'Cluster'}.issubset(df.columns):
            raise ValueError("The CSV file must contain 'X', 'Y', 'Altitude', and 'Cluster' columns.")

        # Simplify data by selecting representative points for each cluster
        simplified_points = df.groupby('Cluster').apply(lambda group: group.sample(n=1)).reset_index(drop=True)

        # Extract coordinates and altitude for simplified TIN
        points = simplified_points[['X', 'Y']].values
        altitudes = simplified_points['Altitude'].values

        # Perform Delaunay triangulation
        delaunay = Delaunay(points)

        # Visualize the TIN
        plt.figure(figsize=(10, 8))
        plt.triplot(points[:, 0], points[:, 1], delaunay.simplices, color='gray', linewidth=0.5)
        scatter = plt.scatter(points[:, 0], points[:, 1], c=altitudes, cmap='terrain', s=50, edgecolor='black')
        plt.colorbar(scatter, label='Altitude')
        plt.title("TIN (Triangulated Irregular Network) from Simplified Data")
        plt.xlabel("X Coordinate")
        plt.ylabel("Y Coordinate")
        plt.grid()
        plt.show()

        # Save simplified points to CSV if output path is provided
        if output_csv:
            simplified_points.to_csv(output_csv, index=False)
            print(f"Simplified TIN points saved to {output_csv}")

        return delaunay

    except FileNotFoundError:
        print(f"File not found: {clustered_csv_file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")


def create_colored_tin_from_clustered_csv(clustered_csv_file_path, output_csv=None):
    """
    Creates and visualizes a colored TIN (Triangulated Irregular Network) from a clustered CSV file.

    Parameters:
        clustered_csv_file_path (str): Path to the clustered CSV file containing X, Y, Altitude, and Cluster columns.
        output_csv (str): Path to save the simplified TIN points as a CSV file (optional).

    Returns:
        delaunay (scipy.spatial.Delaunay): Delaunay triangulation object.
    """
    try:
        # Load the clustered CSV file
        df = pd.read_csv(clustered_csv_file_path)

        # Check for required columns
        if not {'X', 'Y', 'Altitude', 'Cluster'}.issubset(df.columns):
            raise ValueError("The CSV file must contain 'X', 'Y', 'Altitude', and 'Cluster' columns.")

        # Simplify data by selecting representative points for each cluster
        simplified_points = df.groupby('Cluster').apply(lambda group: group.sample(n=1)).reset_index(drop=True)

        # Extract coordinates and altitude for simplified TIN
        points = simplified_points[['X', 'Y']].values
        altitudes = simplified_points['Altitude'].values

        # Perform Delaunay triangulation
        delaunay = Delaunay(points)

        # Calculate altitude for each triangle (average of the vertices)
        triangles = delaunay.simplices
        triangle_altitudes = np.mean(altitudes[triangles], axis=1)

        # Visualize the TIN with colored areas
        plt.figure(figsize=(10, 8))
        plt.tripcolor(points[:, 0], points[:, 1], triangles, facecolors=triangle_altitudes, cmap='terrain', edgecolors='gray', linewidth=0.5)
        scatter = plt.scatter(points[:, 0], points[:, 1], c=altitudes, cmap='terrain', s=50, edgecolor='black')
        plt.colorbar(label='Altitude')
        plt.title("TIN (Triangulated Irregular Network) with Colored Areas")
        plt.xlabel("X Coordinate")
        plt.ylabel("Y Coordinate")
        plt.grid()
        plt.show()

        # Save simplified points to CSV if output path is provided
        if output_csv:
            simplified_points.to_csv(output_csv, index=False)
            print(f"Simplified TIN points saved to {output_csv}")

        return delaunay

    except FileNotFoundError:
        print(f"File not found: {clustered_csv_file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")


def overlay_tin_on_tfw(clustered_csv_file_path, tif_file_path, output_csv=None):
    """
    Creates and overlays a TIN (Triangulated Irregular Network) on a raster area from a TIF file.

    Parameters:
        clustered_csv_file_path (str): Path to the clustered CSV file containing X, Y, Altitude, and Cluster columns.
        tif_file_path (str): Path to the raster (e.g., TIF) file georeferenced by a TFW file.
        output_csv (str): Path to save the simplified TIN points as a CSV file (optional).

    Returns:
        None
    """
    try:
        # Load the clustered CSV file
        df = pd.read_csv(clustered_csv_file_path)

        # Check for required columns
        if not {'X', 'Y', 'Altitude', 'Cluster'}.issubset(df.columns):
            raise ValueError("The CSV file must contain 'X', 'Y', 'Altitude', and 'Cluster' columns.")

        # Simplify data by selecting representative points for each cluster
        simplified_points = df.groupby('Cluster').apply(lambda group: group.sample(n=1)).reset_index(drop=True)

        # Extract coordinates and altitude for simplified TIN
        points = simplified_points[['X', 'Y']].values
        altitudes = simplified_points['Altitude'].values

        # Perform Delaunay triangulation
        delaunay = Delaunay(points)

        # Calculate altitude for each triangle (average of the vertices)
        triangles = delaunay.simplices
        triangle_altitudes = np.mean(altitudes[triangles], axis=1)

        # Read the raster file
        with rasterio.open(tif_file_path) as src:
            raster = src.read(1)  # Read the first band of the raster
            transform = src.transform
            extent = [
                transform[2],  # Left
                transform[2] + raster.shape[1] * transform[0],  # Right
                transform[5] + raster.shape[0] * transform[4],  # Bottom
                transform[5],  # Top
            ]

        # Plot the raster as a background
        plt.figure(figsize=(12, 10))
        plt.imshow(raster, extent=extent, cmap='gray', origin='upper')
        plt.colorbar(label='Raster Intensity')
        plt.title("TIN Overlaid on Raster")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")

        # Overlay the TIN
        plt.tripcolor(points[:, 0], points[:, 1], triangles, facecolors=triangle_altitudes, cmap='terrain', edgecolors='gray', linewidth=0.5, alpha=0.2)
        scatter = plt.scatter(points[:, 0], points[:, 1], c=altitudes, cmap='terrain', s=50, edgecolor='black', label='TIN Points')
        plt.colorbar(scatter, label='Altitude')

        plt.legend()
        plt.grid()
        plt.show()

        # Save simplified points to CSV if output path is provided
        if output_csv:
            simplified_points.to_csv(output_csv, index=False)
            print(f"Simplified TIN points saved to {output_csv}")

    except FileNotFoundError as e:
        print(f"File not found: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")


def generate_contour_map(tif_file_path, contour_interval=10):
    """
    Generate and plot a contour map from a georeferenced TIFF file.

    Parameters:
        tif_file_path (str): Path to the .tif file.
        contour_interval (int): Interval between contour levels.
    """
    try:
        # Open the TIFF file
        with rasterio.open(tif_file_path) as src:
            # Read the raster data
            data = src.read(1)  # Read the first band (assumes single-band DEM)
            transform = src.transform  # Georeferencing transform
            bounds = src.bounds  # Raster bounds
            crs = src.crs  # Coordinate reference system

            # Create coordinate arrays for the raster
            rows, cols = data.shape
            x = np.linspace(bounds.left, bounds.right, cols)
            y = np.linspace(bounds.top, bounds.bottom, rows)
            X, Y = np.meshgrid(x, y[::-1])  # Flip Y to match raster orientation

            # Mask invalid data (e.g., nodata values)
            data = np.ma.masked_where(data == src.nodata, data)

        # Plot the contour map
        plt.figure(figsize=(10, 8))
        contour = plt.contour(X, Y, data, levels=np.arange(data.min(), data.max(), contour_interval), cmap="terrain")
        plt.clabel(contour, inline=True, fontsize=8, fmt='%1.0f')
        plt.title("Contour Map")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.colorbar(label="Elevation (units)")
        plt.grid()
        plt.show()

    except Exception as e:
        print(f"An error occurred: {e}")

