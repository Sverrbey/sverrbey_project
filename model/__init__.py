"""
This file is run in order to calibrate the model and make sure that it runs efficiently. The calibration
process consists of:
    - Clustering the altitude data
"""

import os
from src.function import extract_altitude_points, create_tin_from_clustered_csv, cluster_elevation_data

## 0. Calibration
if os.path.exists('../data/clustered_altitude_points.csv') == False:

    # Extracting altitude poits from tif file
    tfw_file_path = "../data/dtm50/dtm50_7002_50m_33.tfw"
    tif_file_path = "../data/dtm50/dtm50_7002_50m_33.tif"
    output_csv_path = "../data/altitude_points.csv"  # Replace with desired CSV file path or set to None

    altitude_points = extract_altitude_points(tif_file_path, output_csv_path)

    # Display a preview of the altitude points
    if altitude_points is not None:
        print(altitude_points.head())

    ''' Create contour map from tif file 
    generate_contour_map(tif_file_path, contour_interval=20)
    '''

    # Clustering elevation points 
    csv_file_path = "../data/altitude_points.csv"  # Replace with your CSV file path
    output_csv_path = "../data/clustered_altitude_points.csv"  # Replace or set to None if not saving
    clustered_df = cluster_elevation_data(csv_file_path, n_clusters=100, output_csv=output_csv_path)

    # Display a preview of the clustered data
    if clustered_df is not None:
        print(clustered_df.head())


    # TIN from clusted data
    clustered_csv_file_path = "../data/clustered_altitude_points.csv"  # Replace with your clustered CSV file path
    output_csv_path = "simplified_tin_points.csv"  # Replace or set to None if not saving
    create_tin_from_clustered_csv(clustered_csv_file_path, output_csv=output_csv_path)
