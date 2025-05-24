import pyomo as pyo
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt

"""
TODO: Print out the results from the physical model, and try to improve the simple HBV model, then integrate it with the LSTM model.
    - Because, one of the assumptions for using the conceptual model is that it performs satisfactory in the first place.
"""
def main():
    
    

    return 


def HBV_Enhanced(height, width, elevation_map, points, df_parameter):
        model = pyo.ConcreteModel()

        # Sets
        model.c = pyo.Set(initialize=range(0, height*width))

        # HBV Parameters
        model.df_parameter = df_parameter
        model.points = points

        # Initialize parameter 
        model.TT     = pyo.Parameter(model.c, within = model.df_paramter[0]["interval"])
        model.CFR    = pyo.Parameter(model.c, within = model.df_paramter[1]["interval"])
        model.CFMAX  = pyo.Parameter(model.c, within = model.df_paramter[2]["interval"])
        model.FC     = pyo.Parameter(model.c, within = model.df_paramter[3]["interval"])
        model.UZL    = pyo.Parameter(model.c, within = model.df_paramter[4]["interval"])
        model.K0     = pyo.Parameter(model.c, within = model.df_paramter[5]["interval"])
        model.K1     = pyo.Parameter(model.c, within = model.df_paramter[6]["interval"])
        model.K2     = pyo.Parameter(model.c, within = model.df_paramter[7]["interval"])
        model.PERC   = pyo.Parameter(model.c, within = model.df_paramter[8]["interval"])
        model.CN     = pyo.Parameter(model.c, within = model.df_paramter[9]["interval"])
        model.BETA   = pyo.Parameter(model.c, within = model.df_paramter[10]["interval"])
        model.LP     = pyo.Parameter(model.c, within = model.df_paramter[11]["interval"])

        # Variables 
        model.temp          = pyo.Var(within= pyo.Reals, initialize=0)
        model.prec          = pyo.Var(within= pyo.Reals, initialize=0)
        model.rad           = pyo.Var(within= pyo.Reals, initialize=0)
        model.hyd           = pyo.Var(within= pyo.Reals, initialize=0)
        model.relHum        = pyo.Var(within= pyo.Reals, initialize=0)
        model.wind          = pyo.Var(within= pyo.Reals, initialize=0)
        model.evap          = pyo.Var(within= pyo.Reals, initialize=0)
        model.SM            = pyo.Var(within= pyo.Reals, initialize=0)
        model.recharge      = pyo.Var(within= pyo.Reals, initialize=0)
        model.UZ            = pyo.Var(within= pyo.Reals, initialize=0)
        model.LZ            = pyo.Var(within= pyo.Reals, initialize=0)
        model.Q_0           = pyo.Var(within= pyo.Reals, initialize=0)
        model.Q_1           = pyo.Var(within= pyo.Reals, initialize=0)
        model.snow_acc      = pyo.Var(within= pyo.Reals, initialize=0)

        # Parameters
        model.max_snow = 500 
        model.elevation_map = elevation_map
       
        # Initialize routed flow matrix
        model.routed_flow = np.zeros((height, width))

        def curve_number_runoff(model, c):
           """
           Calculate surface runoff using the SCS Curve Number method.
           """
           epsilon = 1e-6
           S = 25.4 * (1000 / model.CN - 10)  # Potential maximum retention (mm)
           Ia = 0.2 * S  # Initial abstraction (mm)
           
           # Calculate runoff using a mask for efficient computation
           mask_absorption = model.prec > Ia
           runoff = np.zeros_like(model.prec)
           runoff[mask_absorption] = ((model.prec[mask_absorption] - Ia[mask_absorption]) ** 2) / \
                                     (model.prec[mask_absorption] - Ia[mask_absorption] + S[mask_absorption] + epsilon)
           return runoff
        model.curve_number = pyo.Constraint(model.c, rule=curve_number_runoff)
        
        def snow_routine(model):
           """
           Calculate snow storage and release in the snowpack.
           """
           epsilon = 1e-6
           # Store snow
           mask_storing = (model.temp <= model.TT) & (model.snow_acc <= model.max_snow)
           model.snow_acc = np.where(mask_storing, model.snow_acc + model.prec + epsilon, model.snow_acc)

           # Release snow
           mask_snow_stored = model.snow_acc > 0
           melt = np.zeros_like(model.snow_acc)
           refreezing = np.zeros_like(model.snow_acc)
           melt[mask_snow_stored] = model.CFMAX[mask_snow_stored] * (model.temp[mask_snow_stored] - model.TT[mask_snow_stored])
           refreezing[mask_snow_stored] = model.CFR[mask_snow_stored] * melt[mask_snow_stored]
           model.Q_0 = np.where(mask_snow_stored, melt - refreezing + epsilon, model.Q_0)
           model.snow_acc = np.where(mask_snow_stored, model.snow_acc - model.Q_0 + epsilon, model.snow_acc)
           model.snow_acc = np.relu(model.snow_acc)

        def soil_moisture_routine(model):
           """
           Calculate the soil moisture routine and water flow between reservoirs.
           """
           epsilon = 1e-6
           # Define the variables
           Q_SM            = np.zeros_like(model.Q_0)
           soil_resistance = np.zeros_like(model.SM)
           overflow        = np.zeros_like(model.Q_0)
           q0              = np.zeros_like(model.UZ)
           q1              = np.zeros_like(model.UZ)
           q2              = np.zeros_like(model.LZ)

           mask_positive   = model.FC > 0
           soil_resistance = np.where(mask_positive, np.exp(model.SM / (model.FC + epsilon) - 1), np.zeros_like(model.SM))
           mask_positive_SM   = model.SM > 0
           Q_SM            = np.where(mask_positive_SM, (1 - soil_resistance) * (model.Q_0 + model.prec + epsilon), np.zeros_like(model.Q_0))
           model.recharge   = np.where(mask_positive_SM, model.Q_0 + model.prec - Q_SM , np.zeros_like(model.Q_0))
          

           model.SM = model.SM + Q_SM - model.evap
           mask_soil_saturation = model.SM >= model.FC
           model.recharge = np.where(mask_soil_saturation, model.recharge + (model.SM - model.FC) + epsilon, model.recharge)
           model.SM = np.where(mask_soil_saturation, model.FC + epsilon, model.SM)

           # Recharge to UZ
           model.UZ = model.UZ + model.recharge
           mask_overflow = model.UZ > model.FC
           overflow = np.where(mask_overflow, model.UZ - model.FC + epsilon, np.zeros_like(model.UZ))
           model.UZ = np.where(mask_overflow, model.FC + epsilon, model.UZ)

           # Calculate q0 and q1
           mask_non_zero = model.UZ > 0
           mask_UZ_UZL = model.UZ > model.UZL
           
           q0 = np.where(mask_UZ_UZL, model.K0 * (model.UZ - model.UZL) + epsilon, q0)
           q1 = np.where(mask_non_zero, model.K1 * model.UZ + epsilon, q1)

           model.UZ = np.where(mask_UZ_UZL, model.UZ - q0 + epsilon, model.UZ)
           model.UZ = np.where(mask_non_zero, model.UZ - q1 + epsilon, model.UZ)

           # Percolation to LZ
           model.LZ = model.LZ + overflow
           mask_LZ_non_zero = model.LZ > 0
           q2 = np.where(mask_LZ_non_zero, model.K2 * model.LZ + epsilon, q2)
           model.LZ = np.where(mask_LZ_non_zero, model.LZ - q2 + epsilon, model.LZ)
           model.LZ = np.relu(model.LZ)

           # Update Q_1
           model.Q_1 = model.Q_1 + q0 + q1
           model.Q_1 = np.where(mask_LZ_non_zero, model.Q_1 + q2 + epsilon, model.Q_1)
           model.Q_1 = np.where(mask_overflow, model.Q_1 + overflow + epsilon, model.Q_1)

    def routing(model, flow_matrix):
        """
        Routes the flow based on the elevation map, ensuring the resulting flow matrix supports gradient computation.

        Args:
            flow_matrix (np.Tensor): Tensor of shape (batch_size, height, width) representing flow/discharge.

        Returns:
            np.Tensor: Routed flow matrix with the same shape as the input flow_matrix.
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

        resulting_flow = np.zeros_like(flow_matrix, requires_grad=True)

        # Iterate over each batch
        for batch in range(batch_size):
            # Get the flow for the current batch
            model.routed_flow = flow_matrix[batch]

            # Create a tensor to store contributions to each cell
            contributions = np.zeros_like(model.routed_flow)
            
            # Iterate over each cell in the grid
            for y in range(height):
                for x in range(width):
                    # Skip if the current cell has no flow
                    if model.routed_flow[y, x] == 0:
                        continue

                    # Find the steepest downhill neighbor
                    steepest_slope = 0
                    target_y, target_x = y, x  # Default to the current cell

                    for dy, dx in neighbor_offsets:
                        ny, nx = y + dy, x + dx

                        # Check if the neighbor is within bounds
                        if 0 <= ny < height and 0 <= nx < width:
                            # Calculate the slope to the neighbor
                            slope = model.elevation_map[y, x] - model.elevation_map[ny, nx]

                            # Update the steepest slope and target cell
                            if slope > steepest_slope:
                                steepest_slope = slope
                                target_y, target_x = ny, nx

                    # Route the flow to the target cell
                    contributions[target_y, target_x] = contributions[target_y, target_x] + model.routed_flow[y, x]

            # Add contributions to the routed flow for the current batch
            resulting_flow = resulting_flow.clone()
            resulting_flow[batch] = resulting_flow[batch] + contributions

        return resulting_flow
    
    def penman_montieth(model):
        """
        Calculate potential evaporation using the Penman-Monteith equation.
        """
        epsilon = 1e-6
        # Constants
        delta = 0.6108 * np.exp((17.27 * model.temp) / (model.temp + 237.3))
        gamma = 0.665 * (1.0 / (model.relHum + epsilon)) * (model.temp + 273.15)
        # Calculate potential evaporation
        numerator = (delta * model.rad) + (gamma * model.wind)
        denominator = delta + gamma
        model.evap = numerator / (denominator + epsilon)
        # Ensure evaporation is non-negative using ReLU
        model.evap = np.relu(model.evap)
        return 



main()
