import torch
import numpy as np
import pandas as pd


''' Overlay '''
clustered_csv_file_path = "clustered_altitude_points.csv"  # Replace with your clustered CSV file path
tif_file_path = "../data/dtm50/dtm50_7002_50m_33.tif"
output_csv_path = "simplified_tin_points.csv"  # Replace or set to None if not saving
overlay_tin_on_tfw(clustered_csv_file_path, tif_file_path, output_csv=output_csv_path)


''' Colored TIN
clustered_csv_file_path = "clustered_altitude_points.csv"  # Replace with your clustered CSV file path
output_csv_path = "simplified_tin_points.csv"  # Replace or set to None if not saving
create_colored_tin_from_clustered_csv(clustered_csv_file_path, output_csv=output_csv_path)
 '''

''' Clustering elevation points 
csv_file_path = "altitude_points.csv"  # Replace with your CSV file path
output_csv_path = "clustered_altitude_points.csv"  # Replace or set to None if not saving
clustered_df = cluster_elevation_data(csv_file_path, n_clusters=10, output_csv=output_csv_path)

# Display a preview of the clustered data
if clustered_df is not None:
   print(clustered_df.head())
'''
''' TIN from clusted data'''
#clustered_csv_file_path = "clustered_altitude_points.csv"  # Replace with your clustered CSV file path
#output_csv_path = "simplified_tin_points.csv"  # Replace or set to None if not saving
#create_tin_from_clustered_csv(clustered_csv_file_path, output_csv=output_csv_path)

''' TIN from csv altitude data
csv_file_path = "altitude_points.csv"  # Replace with your CSV file path
create_tin_from_csv(csv_file_path)
'''

''' Read tfw file
tfw_file_path = "../data/dtm50/dtm50_7002_50m_33.tfw"
read_tfw(tfw_file_path)
'''
''' Create contour map from tif file
tif_file_path = "../data/dtm50/dtm50_7002_50m_33.tif"
generate_contour_map(tif_file_path, contour_interval=20)
'''

''' Create altitude poits from tif file
output_csv_path = "altitude_points.csv"  # Replace with desired CSV file path or set to None
altitude_points = extract_altitude_points(tif_file_path, output_csv_path)

# Display a preview of the altitude points
if altitude_points is not None:
    print(altitude_points.head())
'''
