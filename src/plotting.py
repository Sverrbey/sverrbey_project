import numpy as np
import matplotlib.pyplot as plt
import torch


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
            outputs = model(batch_x, scaler_x)
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