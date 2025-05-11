import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt

"""
TODO: Print out the results from the physical model, and try to improve the simple HBV model, then integrate it with the LSTM model.
    - Because, one of the assumptions for using the conceptual model is that it performs satisfactory in the first place.
"""

class HBV_LSTM(nn.Module):
    def __init__(self, input_channels, output_channels, height, width, hidden_size, elevation_map, points, df_parameter, num_layers=1, dropout=0.3):
        super(HBV_LSTM, self).__init__()
        
        # Flatten spatial dimensions (height * width) into a single feature vector
        self.input_size = input_channels * height * width
        self.output_size = output_channels* height * width
        self.output_channels = output_channels
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=self.input_size,  # Flattened input size
            hidden_size=hidden_size,    # Number of hidden units
            num_layers=num_layers,      # Number of LSTM layers
            batch_first=True,           # Input shape: (batch_size, seq_length, input_size)
            dropout=dropout,             # Dropout for regularization
            bidirectional=False  # Enable bidirectional LSTM
        )
        
         # Define ranges for each channel (min, max)
        self.channel_ranges = df_parameter

        # Fully connected layer for final output
        self.fc = nn.Linear(hidden_size, self.output_size)

        # Convert elevation map to PyTorch tensor for compatibility
        self.elevation_map = torch.tensor(elevation_map, dtype=torch.float32)

        # HBV_Model_LSTM instance
        self.hbv_model = HBV(height, width, self.elevation_map, self.channel_ranges)

        self.points = points

    def forward(self, x, scaler_x):
        # Input shape: (batch_size, seq_length, channels, height, width)
        batch_size, seq_length, input_channels, height, width = x.size()

        hbv_input = x.view(batch_size * seq_length, -1).detach().cpu().numpy()
        # Flatten spatial dimensions (channels, height, width) into a single feature vector
        x = x.view(batch_size, seq_length, -1)  # Shape: (batch_size, seq_length, input_size)
        
        # Pass through LSTM
        lstm_out, _ = self.lstm(x)  # Output shape: (batch_size, seq_length, hidden_size)
        
        # Take the output of the last time step
        out = lstm_out[:, -1, :]  # Shape: (batch_size, hidden_size)
        
        # Pass through the fully connected layer
        out = self.fc(out)  # Shape: (batch_size, output_channels * height * width)
        
        # Reshape to (batch_size, output_channels, height, width)
        out = out.view(batch_size, self.output_channels, height, width)

        # Apply clamping for each channel
        for channel in self.channel_ranges.keys():
            min_val = self.channel_ranges[channel]["interval"][0]
            max_val = self.channel_ranges[channel]["interval"][1]
            out[:, channel, :, :] = torch.clamp(out[:, channel, :, :], min=min_val, max=max_val)

        x_inverse = scaler_x.inverse_transform(hbv_input)
        x = torch.tensor(x_inverse).view(batch_size, seq_length, input_channels, height, width)

        hbv_outputs = self.hbv_model(x, out)

        routing_output = self.hbv_model.routing(hbv_outputs)
        physcial_outputs = []
        for (i, j) in self.points:
            # Extract the output for the specific point
            physcial_outputs.append(routing_output[:, i, j])  # Extract values for each batch
        # Convert to a tensor
        outputs = torch.stack(physcial_outputs, dim=1)  # Shape: (batch_size, num_points)
        outputs_min = outputs.min(dim=0, keepdim=True)[0]
        outputs_max = outputs.max(dim=0, keepdim=True)[0]
        outputs_scaled = (outputs - outputs_min) / (outputs_max - outputs_min + 1e-6)  # Add epsilon to avoid division by zero

        return outputs_scaled
    

class HBV(nn.Module):
    def __init__(self, height, width, elevation_map, channel_ranges):
        super(HBV, self).__init__()
        # HBV Parameters
        # Define ranges for each channel (min, max)
        self.channel_ranges = channel_ranges

        # Initialize parameter tensors
        self.TT     = nn.Parameter(torch.rand((height, width))*(self.channel_ranges[0]["interval"][1] - self.channel_ranges[0]["interval"][0]) + self.channel_ranges[0]["interval"][0])
        self.CFR    = nn.Parameter(torch.rand((height, width))*(self.channel_ranges[1]["interval"][1] - self.channel_ranges[1]["interval"][0]) + self.channel_ranges[1]["interval"][0])
        self.CFMAX  = nn.Parameter(torch.rand((height, width))*(self.channel_ranges[2]["interval"][1] - self.channel_ranges[2]["interval"][0]) + self.channel_ranges[2]["interval"][0])
        self.FC     = nn.Parameter(torch.rand((height, width))*(self.channel_ranges[3]["interval"][1] - self.channel_ranges[3]["interval"][0]) + self.channel_ranges[3]["interval"][0])
        self.UZL    = nn.Parameter(torch.rand((height, width))*(self.channel_ranges[4]["interval"][1] - self.channel_ranges[4]["interval"][0]) + self.channel_ranges[4]["interval"][0])
        self.K0     = nn.Parameter(torch.rand((height, width))*(self.channel_ranges[5]["interval"][1] - self.channel_ranges[5]["interval"][0]) + self.channel_ranges[5]["interval"][0])
        self.K1     = nn.Parameter(torch.rand((height, width))*(self.channel_ranges[6]["interval"][1] - self.channel_ranges[6]["interval"][0]) + self.channel_ranges[6]["interval"][0])
        self.K2     = nn.Parameter(torch.rand((height, width))*(self.channel_ranges[7]["interval"][1] - self.channel_ranges[7]["interval"][0]) + self.channel_ranges[7]["interval"][0])
        self.PERC   = nn.Parameter(torch.rand((height, width))*(self.channel_ranges[8]["interval"][1] - self.channel_ranges[8]["interval"][0]) + self.channel_ranges[8]["interval"][0])
        self.CN     = nn.Parameter(torch.rand((height, width))*(self.channel_ranges[9]["interval"][1] - self.channel_ranges[9]["interval"][0]) + self.channel_ranges[9]["interval"][0])
        self.BETA   = nn.Parameter(torch.rand((height, width))*(self.channel_ranges[10]["interval"][1] - self.channel_ranges[10]["interval"][0]) + self.channel_ranges[10]["interval"][0])
        self.LP     = nn.Parameter(torch.rand((height, width))*(self.channel_ranges[11]["interval"][1] - self.channel_ranges[11]["interval"][0]) + self.channel_ranges[11]["interval"][0])
        

        # Variables 
        self.temp       = torch.zeros((height, width))
        self.prec       = torch.zeros((height, width))
        self.rad        = torch.zeros((height, width))
        self.hyd        = torch.zeros((height, width))
        self.relHum     = torch.zeros((height, width))
        self.wind       = torch.zeros((height, width))
        self.evap       = torch.zeros((height, width))
        self.SM         =  torch.ones((height, width))
        self.recharge   = torch.zeros((height, width))
        self.UZ        = torch.zeros((height, width)) 
        self.LZ        = torch.zeros((height, width)) 
        self.Q_0        = torch.zeros((height, width))
        self.Q_1        = torch.zeros((height, width)) 
        self.snow_acc   = torch.zeros((height, width))
        
        # Parameters
        self.max_snow = 500 
        self.elevation_map = elevation_map
       
    def curve_number_runoff(self):
        """
        Calculate surface runoff using the SCS Curve Number method.
        """
        S = 25.4 * (1000 / self.CN - 10)  # Potential maximum retention (mm)
        Ia = 0.2 * S  # Initial abstraction (mm)
        
        # Calculate runoff using a mask for efficient computation
        mask_absorption = self.prec > Ia
        runoff = torch.zeros_like(self.prec)
        runoff[mask_absorption] = ((self.prec[mask_absorption] - Ia[mask_absorption]) ** 2) / \
                                  (self.prec[mask_absorption] - Ia[mask_absorption] + S[mask_absorption])
        return runoff

    def snow_routine(self):
        """
        Calculate snow storage and release in the snowpack.
        """
        # Store snow
        mask_storing = (self.temp <= self.TT) & (self.snow_acc <= self.max_snow)
        self.snow_acc = torch.where(mask_storing, self.snow_acc + self.prec, self.snow_acc)

        # Release snow
        mask_snow_stored = self.snow_acc > 0
        melt = torch.zeros_like(self.snow_acc)
        refreezing = torch.zeros_like(self.snow_acc)
        melt[mask_snow_stored] = self.CFMAX[mask_snow_stored] * (self.temp[mask_snow_stored] - self.TT[mask_snow_stored])
        refreezing[mask_snow_stored] = self.CFR[mask_snow_stored] * melt[mask_snow_stored]
        self.Q_0 = torch.where(mask_snow_stored, melt - refreezing, self.Q_0)
        self.snow_acc = torch.where(mask_snow_stored, self.snow_acc - self.Q_0, self.snow_acc)
        self.snow_acc = torch.clamp(self.snow_acc, min=0)

    def soil_moisture_routine(self):
        """
        Calculate the soil moisture routine and water flow between reservoirs.
        """
        epsilon = 1e-2
        soil_resistance = torch.exp(self.SM / (self.FC + epsilon) - 1)
        Q_SM = (1 - soil_resistance) * (self.Q_0 + self.prec)
        self.recharge = self.Q_0 + self.prec - Q_SM

        self.SM = self.SM + Q_SM - self.evap
        mask_soil_saturation = self.SM >= self.FC
        self.recharge = torch.where(mask_soil_saturation, self.recharge + (self.SM - self.FC), self.recharge)
        self.SM = torch.where(mask_soil_saturation, self.FC, self.SM)

        # Recharge to UZ
        self.UZ = self.UZ + self.recharge
        mask_overflow = self.UZ > self.FC
        overflow = torch.where(mask_overflow, self.UZ - self.FC, torch.zeros_like(self.UZ))
        self.UZ = torch.where(mask_overflow, self.FC, self.UZ)

        # Calculate q0 and q1
        mask_non_zero = self.UZ > 0
        q0 = torch.zeros_like(self.UZ)
        q1 = torch.zeros_like(self.UZ)
        mask_UZ_UZL = self.UZ > self.UZL
        q0 = torch.where(mask_UZ_UZL, self.K0 * (self.UZ - self.UZL), q0)
        q1 = torch.where(mask_non_zero, self.K1 * self.UZ, q1)
        self.UZ = torch.where(mask_UZ_UZL, self.UZ - q0, self.UZ)
        self.UZ = torch.where(mask_non_zero, self.UZ - q1, self.UZ)

        # Percolation to LZ
        self.LZ = self.LZ + overflow
        q2 = torch.zeros_like(self.LZ)
        mask_LZ_non_zero = self.LZ > 0
        q2 = torch.where(mask_LZ_non_zero, self.K2 * self.LZ, q2)
        self.LZ = torch.where(mask_LZ_non_zero, self.LZ - q2, self.LZ)
        self.LZ = torch.clamp(self.LZ, min=0)

        # Update Q_1
        self.Q_1 = self.Q_1 + q0 + q1
        self.Q_1 = torch.where(mask_LZ_non_zero, self.Q_1 + q2, self.Q_1)
        self.Q_1 = torch.where(mask_overflow, self.Q_1 + overflow, self.Q_1)
    
    def routing(self, flow_matrix):
        """
        Routes the flow based on the elevation map.

        Args:
            flow_matrix (torch.Tensor): Tensor of shape (batch_size, seq_length, height, width) representing flow/discharge.
            elevation_map (np.ndarray): 2D NumPy array of shape (height, width) representing the elevation.

        Returns:
            torch.Tensor: Routed flow matrix with the same shape as the input flow_matrix.
        """
        # Get dimensions
        batch_size, height, width = flow_matrix.shape

        # Define neighbor offsets (8 directions: N, NE, E, SE, S, SW, W, NW)
        neighbor_offsets = [
            (-1, 0),  # North
            (-1, 1),  # North-East
            (0, 1),   # East
            (1, 1),   # South-East
            (1, 0),   # South
            (1, -1),  # South-West
            (0, -1),  # West
            (-1, -1)  # North-West
        ]

        # Initialize routed flow matrix
        routed_flow = torch.zeros_like(flow_matrix)

        # Iterate over each batch and time step
        for batch in range(batch_size):
            # Get the flow for the current time step
            current_flow = flow_matrix[batch]

            # Iterate over each cell in the grid
            for y in range(height):
                for x in range(width):
                    # Skip if the current cell has no flow
                    if current_flow[y, x] == 0:
                        continue

                    # Find the steepest downhill neighbor
                    steepest_slope = 0
                    target_y, target_x = y, x  # Default to the current cell

                    for dy, dx in neighbor_offsets:
                        ny, nx = y + dy, x + dx

                        # Check if the neighbor is within bounds
                        if 0 <= ny < height and 0 <= nx < width:
                            # Calculate the slope to the neighbor
                            slope = self.elevation_map[y, x] - self.elevation_map[ny, nx]

                            # Update the steepest slope and target cell
                            if slope > steepest_slope:
                                steepest_slope = slope
                                target_y, target_x = ny, nx

                    # Route the flow to the target cell
                    routed_flow[batch, target_y, target_x] += current_flow[y, x]

        return routed_flow
    
    def penman_montieth(self):
        """
        Calculate potential evaporation using the Penman-Monteith equation.
        """
        # Constants
        delta = 0.6108 * torch.exp((17.27 * self.temp) / (self.temp + 237.3))
        gamma = 0.665 * (1.0 / self.relHum) * (self.temp + 273.15)
        # Calculate potential evaporation
        numerator = (delta * self.rad) + (gamma * self.wind)
        denominator = delta + gamma
        self.evap = numerator / denominator
        # Ensure evaporation is non-negative
        self.evap = torch.clamp(self.evap, min=0)
        return 

    def forward(self, input_x, parameters):
        # Input shape: (batch_size, seq_length, channels, height, width)
        batch_size, seq_length, channels, height, width = input_x.shape
       
        # Create a grid of points
        Q_sim = torch.zeros((batch_size, height, width))
        for batch in range(batch_size):

            self.CFR    = nn.Parameter(parameters[batch, 1,:,:])
            self.CFMAX  = nn.Parameter(parameters[batch, 2,:,:])
            self.FC     = nn.Parameter(parameters[batch, 3,:,:])
            self.UZL    = nn.Parameter(parameters[batch, 4,:,:])
            self.K0     = nn.Parameter(parameters[batch, 5,:,:])
            self.K1     = nn.Parameter(parameters[batch, 6,:,:])
            self.K2     = nn.Parameter(parameters[batch, 7,:,:])
            self.PERC   = nn.Parameter(parameters[batch, 8,:,:])
            self.CN     = nn.Parameter(parameters[batch, 9,:,:])
            self.BETA   = nn.Parameter(parameters[batch, 10,:,:])
            self.LP     = nn.Parameter(parameters[batch, 11,:,:])

            for seq in range(seq_length):
                # Extract the input data for the current batch and time step
                self.prec    = input_x[batch, seq, 0,:,:]
                self.temp    = input_x[batch, seq, 1,:,:]
                self.rad     = input_x[batch, seq, 2,:,:]
                self.hyd     = input_x[batch, seq, 3,:,:]
                self.relHum  = input_x[batch, seq, 4,:,:]
                self.wind    = input_x[batch, seq, 5,:,:]
        
                # Pass through HBV model
                # Step 1. Calculate the flow from the snow pack 
                self.snow_routine()
                # Step 2-3. Run the soil moisture routine and routing function
                self.penman_montieth()
                self.soil_moisture_routine()
            Q_sim[batch] = self.Q_1
    
        Q_sim = Q_sim.reshape(batch_size, height, width)
        return Q_sim



