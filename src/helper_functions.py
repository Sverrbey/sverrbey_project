
import torch
import matplotlib.pyplot as plt
import numpy as np
from torch import nn
import os
import zipfile
from pathlib import Path
import requests

from typing import List
#import torchvision
import os

def walk_through_dir(dir_path):
    """
    Walks through dir_path returning its contents.
    Args:
    dir_path (str): target directory

    Returns:
    A print out of:
      number of subdiretories in dir_path
      number of images (files) in each subdirectory
      name of each subdirectory
    """
    for dirpath, dirnames, filenames in os.walk(dir_path):
        print(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'.")

def plot_decision_boundary(model: torch.nn.Module, X: torch.Tensor, y: torch.Tensor):
    """Plots decision boundaries of model predicting on X in comparison to y.

    Source - https://madewithml.com/courses/foundations/neural-networks/ (with modifications)
    """
    # Put everything to CPU (works better with NumPy + Matplotlib)
    model.to("cpu")
    X, y = X.to("cpu"), y.to("cpu")

    # Setup prediction boundaries and grid
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 101), np.linspace(y_min, y_max, 101))

    # Make features
    X_to_pred_on = torch.from_numpy(np.column_stack((xx.ravel(), yy.ravel()))).float()

    # Make predictions
    model.eval()
    with torch.inference_mode():
        y_logits = model(X_to_pred_on)

    # Test for multi-class or binary and adjust logits to prediction labels
    if len(torch.unique(y)) > 2:
        y_pred = torch.softmax(y_logits, dim=1).argmax(dim=1)  # mutli-class
    else:
        y_pred = torch.round(torch.sigmoid(y_logits))  # binary

    # Reshape preds and plot
    y_pred = y_pred.reshape(xx.shape).detach().numpy()
    plt.contourf(xx, yy, y_pred, cmap=plt.cm.RdYlBu, alpha=0.7)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.RdYlBu)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())

def simplify_tiff(input_tiff_path, output_tiff_path, scale_factor=2):
    """
    Simplifies a TIFF file by downsampling it.

    Parameters:
    - input_tiff_path (str): Path to the input TIFF file.
    - output_tiff_path (str): Path to save the simplified TIFF file.
    - scale_factor (int): Factor by which to downsample the image. Default is 2.
    """
    with rasterio.open(input_tiff_path) as src:
        # Calculate the new dimensions
        new_height = src.height // scale_factor
        new_width = src.width // scale_factor

        # Read the data and downsample
        data = src.read(
            out_shape=(
                src.count,
                new_height,
                new_width
            ),
            resampling=Resampling.bilinear
        )

        # Scale image transform
        transform = src.transform * src.transform.scale(
            (src.width / data.shape[-1]),
            (src.height / data.shape[-2])
        )

        # Update metadata
        metadata = src.meta.copy()
        metadata.update({
            'height': new_height,
            'width': new_width,
            'transform': transform
        })

        # Write the simplified TIFF file
        with rasterio.open(output_tiff_path, 'w', **metadata) as dst:
            dst.write(data)

def reformat_elevation_map(tif_path, target_shape):
    """
    Reformat a .tif elevation map to match the target shape.

    Args:
        tif_path (str): Path to the .tif elevation map.
        target_shape (tuple): Desired shape (height, width) to match x_data.

    Returns:
        np.ndarray: Resampled elevation map with the target shape.
    """
    # Open the .tif file
    with rasterio.open(tif_path) as src:
        # Read the elevation data
        elevation_data = src.read(1)  # Read the first band (assumes single-band elevation map)

        # Get the original shape of the elevation map
        original_shape = elevation_data.shape

        # Calculate the resampling scale factors
        scale_y = target_shape[0] / original_shape[0]
        scale_x = target_shape[1] / original_shape[1]

        # Resample the elevation data to the target shape
        elevation_resampled = src.read(
            1,
            out_shape=(int(target_shape[0]), int(target_shape[1])),
            resampling=Resampling.bilinear  # Use bilinear interpolation for resampling
        )

        # Ensure the output is a 2D NumPy array
        elevation_resampled = np.array(elevation_resampled)

    return elevation_resampled
