import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt

class HBV_Model_LSTM:
    def __init__(self, params, channels, height, width):
        # Input shape: (batch_size, seq_length, channels, height, width)
        batch_size, hidden_size = params.size()

        # Variables 
        self.temp       = 0
        self.prec       = 0
        self.rad        = 0
        self.hyd        = 0
        self.relHum     = 0
        self.wind       = 0
        self.evap       = 0
        self.snow_pack  = 0
        self.Q_0        = 0 
        self.SM         = 5   #  5 mm 
        self.recharge   = 0
        self.SUZ        = 0 
        self.SLZ        = 0 
        self.Q_1        = 0 
        self.snow_acc   = 0
        
        # Parameters
        self.max_snow = 500 
        

        # Flatten spatial dimensions (channels, height, width) into a single feature vector
       
        params = params.view(batch_size, channels, height, width)  # Shape: (batch_size, seq_length, input_size)
        # Initialize parameter tensors
        self.TT     = torch.zeros((batch_size, height, width))
        self.CFR    = torch.zeros((batch_size, height, width))
        self.CFMAX  = torch.zeros((batch_size, height, width))
        self.FC     = torch.zeros((batch_size, height, width))
        self.UZL    = torch.zeros((batch_size, height, width))
        self.K0     = torch.zeros((batch_size, height, width))
        self.K1     = torch.zeros((batch_size, height, width))
        self.K2     = torch.zeros((batch_size, height, width))
        self.PERC   = torch.zeros((batch_size, height, width))

        # Extract and assign parameter values from the LSTM output
        self.TT     = params[:, 0, :, :]   # Channel 0: TT
        self.CFR    = params[:, 1, :, :]   # Channel 1: CFR
        self.CFMAX  = params[:, 2, :, :]   # Channel 2: CFMAX
        self.FC     = params[:, 3, :, :]   # Channel 3: FC
        self.UZL    = params[:, 4, :, :]   # Channel 4: UZL
        self.K0     = params[:, 5, :, :]   # Channel 5: K0
        self.K1     = params[:, 6, :, :]   # Channel 6: K1
        self.K2     = params[:, 7, :, :]   # Channel 7: K2
        self.PERC   = params[:, 8, :, :]   # Channel 8: PERC
        self.CN     = params[:, 9, :, :]   # Channel 9: CN

    def curve_number_runoff(self, batch, y, x):
        """
        Calculate surface runoff using the SCS Curve Number method.
        :param precipitation: Daily precipitation (mm).
        :param soil_moisture: Current soil moisture (mm).
        :return: Surface runoff (mm).
        """
        
        S = 25.4 * (1000 / self.CN[batch, y, x] - 10)  # Potential maximum retention (mm)
        Ia = 0.2 * S  # Initial abstraction (mm)
        
        if self.prec > Ia:
            runoff = ((self.prec - Ia) ** 2) / (self.prec - Ia + S)
        else:
            runoff = 0
        
        return runoff

    def snow_routine(self, batch, y, x):
        """
            The function of this routine is to find out how much water is being stored and released
            in the snowpack.

            * The only function that is lacking right now is the infiltration
        """
        # Store
        if self.temp <= self.TT[batch, y, x]: # If the temperature is below the threshold
            if self.snow_acc >= self.max_snow:
                self.snow_acc = self.max_snow
            else:
                self.snow_acc += self.prec     
        # Release
        else: 
            if self.snow_acc > 0:
                melt =  self.CFMAX[batch, y, x] * (self.temp - self.TT[batch, y, x]) 
                refreezing = self.CFR[batch, y, x] * melt
                self.Q_0 = melt - refreezing

            self.snow_acc -= self.Q_0 # Remove all snow from storage
            if self.snow_acc < 0: # The accumulated snow can't be negative
                self.snow_acc = 0
                
        self.snow_pack = self.snow_acc
        return
    
    def soil_moisture_routine(self, batch, y, x):
        """ 
        Function to calculate the soil moisture routine.

        recharge: Water from the moist soil entering the storage in the upper reservoir (UZ)

        """
        #1. Infiltration + Rain -> SM -> Recharge          
        soil_resistance = torch.exp(self.SM/self.FC[batch, y, x] - 1)
        Q_SM = (1 - soil_resistance) * (self.Q_0 + self.prec) # If the soil moisture is high then more of the water reaches SUZ -> LOOK INTO A SCS CURVE IMPLEMENTATION
        self.recharge = (self.Q_0 + self.prec - Q_SM)

        self.SM = self.SM + Q_SM - self.evap
        
        if self.SM >= self.FC[batch, y, x]:
            #recharge[day] = SM[day] - params['FC']
            self.SM = self.FC[batch, y, x]
        elif self.SM <= 5:
            self.SM = 5

        #2. Recharge -> SUZ -> q0, q1 
        self.SUZ = self.SUZ + self.recharge 
        overflow = 0
        if self.SUZ >= self.FC[batch, y, x]:
            overflow = self.SUZ - self.FC[batch, y, x]
            self.SUZ = self.FC[batch, y, x]
            #print(f'Overflow: q = {overflow}')

        q0, q1, q2 = 0, 0, 0
        if self.SUZ > 0:
            if self.SUZ > self.UZL[batch, y, x]:
                q0 = self.K0[batch, y, x] * (self.SUZ - self.UZL[batch, y, x]) 
            q1 = self.K1[batch, y, x]*self.SUZ
            self.SUZ -= (q0 + q1)
        else:
            self.SUZ = 0
        
        #3. Percolation -> SLZ -> q2 
        self.SLZ = self.SLZ 
        if self.SLZ > 0:
            q2 = self.K2[batch, y, x]*self.SLZ 
            self.SLZ -= q2
        else:
            self.SLZ = 0

        self.Q_1 = q0 + q1 + q2 + overflow
        return
    
    def forward(self, input_x, out_channels, height,width, extent):
        # Input shape: (batch_size, seq_length, channels, height, width)
        batch_size, seq_length, channels, height, width = input_x.size()
       
        # Create a grid of points
        grid_x, grid_y = np.meshgrid(np.linspace(extent[0], extent[1], width), np.linspace(extent[2], extent[3], height))
        result_matrix = np.zeros((batch_size, height, width))
        # Initialize parameter tensors        
        zs = []
        for batch in range(batch_size):
            # for each cell in the grid find the discharge
            for y in range(height):
                for x in range(width):
                    for seq in range(seq_length):
                        # Extract the input values for the current cell
                        self.prec    = input_x[batch, seq, 0, y, x]
                        self.temp    = input_x[batch, seq, 1, y, x]
                        self.rad     = input_x[batch, seq, 2, y, x]
                        self.hyd     = input_x[batch, seq, 3, y, x]
                        self.relHum  = input_x[batch, seq, 4, y, x]
                        self.wind    = input_x[batch, seq, 5, y, x]

                        # Pass through HBV model
                        # Step 1. Calculate the flow from the snow pack 
                        self.snow_routine(batch, y, x)
                        # Step 2-3. Run the soil moisture routine and routing function
                        self.soil_moisture_routine(batch, y, x)
                    zs.append(self.Q_1)
        # Convert the list to a numpy array
        zs = torch.tensor(zs)
        # Reshape the result to match the output shape
        result_matrix = zs.reshape(batch_size, height, width)
        return result_matrix

def additional():
    # 1. Preprocessing the data and creating sequences
    # Load your data
    """
    df_ptq  = pd.read_csv('/Users/SverreB/Github_Repo/sverrbey_project/src/data/Lookout/data/ptq.txt', header=1, delimiter='\t')
    df_EVAP = pd.read_csv('/Users/SverreB/Github_Repo/sverrbey_project/src/data/Lookout/data/EVAP.txt',header=0)
    tree    = ET.parse('/Users/SverreB/Github_Repo/sverrbey_project/src/data/Lookout/data/Parameter.xml')
    root    = tree.getroot()

    df_ptq['Dag'] = pd.to_datetime(df_ptq['Dag'], format='%Y%m%d')
    df_ptq['month'] = df_ptq['Dag'].dt.month  # Extract the month

    data_0 = df_ptq[["P[mm/d]", "T[C]",	"Q[mm/d]"]]
    """

    df_ptq  = pd.read_csv('src/data/HBV-land/Data/ptq.txt', header=1, delimiter='\t')
    df_EVAP = pd.read_csv('src/data/HBV-land/Data/EVAP.txt',header=0)
    tree    = ET.parse('src/data/HBV-land/Data/Parameter.xml')
    root    = tree.getroot()

    df_ptq['date'] = pd.to_datetime(df_ptq['date'], format='%Y%m%d')
    df_ptq['month'] = df_ptq['date'].dt.month  # Extract the month
    date_month = df_ptq['month']

    data_0 = df_ptq[['Prec.','Temp','Qobs']]
    #
    data_1 = df_EVAP

    
    # Extract information

    ## Parameter_HBVland_start.xml
    # Extract information from XML file
    tree = ET.parse('/Users/SverreB/Github_Repo/sverrbey_project/src/data/HBV-land/Data/Parameter_HBVland_start.xml')
    root = tree.getroot()

    # Extract information
    catchmentParameters = root.find('CatchmentParameters')
    parameter_perc      = int(catchmentParameters.find('PERC').text)

    parameter_uzl       = float(catchmentParameters.find('UZL').text)
    parameter_k0        = float(catchmentParameters.find('K0').text)
    parameter_k1        = float(catchmentParameters.find('K1').text)
    parameter_k2        = float(catchmentParameters.find('K2').text)

    vegetationZoneParamteres    = root.find('VegetationZone/VegetationZoneParameters')
    v_parameter_tt              = int(vegetationZoneParamteres.find('TT').text)
    v_parameter_cfmax           = int(vegetationZoneParamteres.find('CFMAX').text)
    v_parameter_cfr             = float(vegetationZoneParamteres.find('CFR').text)
    v_parameter_fc              = int(vegetationZoneParamteres.find('FC').text)


    params = {
        'TT':   v_parameter_tt,
        'CFR':  v_parameter_cfr,
        'CFMAX':    v_parameter_cfmax,
        'FC':   v_parameter_fc,
        'UZL':  parameter_uzl,
        'K0':   parameter_k0,
        'K1':   parameter_k1,
        'K2':   parameter_k2,
        'PERC': parameter_perc
    }

    # HBV-land
    prec = data_0['Prec.']
    temp = data_0['Temp']
    evaporation = data_1['Nopex_evap']

    #prec = data_0['P[mm/d]']
    #temp = data_0["T[C]"]
    #evaporation = data_1['Cascades_evap']  
    """
    model = HBV_Model(temp, prec, evaporation, date_month, params)
    
    Q = model.run()
    ys = data_0['Qobs']
    #ys = data_0["Q[mm/d]"]
    xs = np.linspace(0, len(ys), len(ys))
  
    # First plot
    plt.step(xs, ys, label='Observed Runoff')
    plt.step(xs, Q, label='Simulated Runoff')
    #axs[0, 0].plot(xs, result, label='error', color='r')
    plt.xlabel('Days')
    plt.ylabel('Runoff')
    plt.legend()
    plt.title('Runoff Comparison')

    plt.show()
    
    
    # Second plot
    plt.plot(xs, temp, label='Temperature')
    plt.xlabel('Days')
    plt.ylabel('Temperature')
    plt.axhline(y=0, color='r', linestyle='--')  # Add horizontal line at y=0
    plt.legend()
    plt.title('Temperature Over Time')
    
    plt.show()
   
    # Third plot
    plt.plot(xs, rain, label='Precipitation')
    plt.xlabel('Days')
    plt.ylabel('Precipitation')
    plt.axhline(y=0, color='r', linestyle='--')  # Add horizontal line at y=0
    plt.legend()
    plt.title('Rainfall')

    plt.show()
   
    # Fourth plot
    plt.step(xs, snow_pack, label='Snow')
    plt.xlabel('Days')
    plt.ylabel('Snow')
    plt.axhline(y=0, color='r', linestyle='--')  # Add horizontal line at y=0
    plt.legend()
    plt.title('Snowpack')

    plt.show()
    """