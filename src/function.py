
import rasterio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import Delaunay
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

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

