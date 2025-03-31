import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt

class HBV_Model:
    def __init__(self, temp, prec, evaporation, dateMonth, params):
        self.n = len(prec) # amount of days
        # Time-series
        self.temp = temp
        self.prec = prec
        self.evap = evaporation
        self.dateMonth = dateMonth
        self.snow = np.zeros(len(prec))
        self.rain = np.zeros(len(prec))
        self.snow_pack = np.zeros(self.n)
        self.Q_0    = np.zeros(self.n) 
        self.SM     = np.zeros(self.n) 
        self.recharge = np.zeros(self.n)
        self.SUZ    = np.zeros(self.n) 
        self.SLZ    = np.zeros(self.n) 
        self.Q_1    = np.zeros(self.n) 
        # Parameters
        self.TT     = params['TT']
        self.CFR    = params['CFR']
        self.CFMAX  = params['CFMAX']
        self.FC     = params['FC']
        self.UZL    = params['UZL']
        self.K0     = params['K0']
        self.K1     = params['K1']
        self.K2     = params['K2']
        self.PERC   = params['PERC']

        for i in range(self.n):
            if self.temp[i] > self.TT:
                self.rain[i] = prec[i]
            else:
                self.snow[i] = prec[i]

    def curve_number_runoff(self, precipitation, soil_moisture):
        """
        Calculate surface runoff using the SCS Curve Number method.
        :param precipitation: Daily precipitation (mm).
        :param soil_moisture: Current soil moisture (mm).
        :return: Surface runoff (mm).
        """
        CN = self.params['curve_number']
        S = 25.4 * (1000 / CN - 10)  # Potential maximum retention (mm)
        Ia = 0.2 * S  # Initial abstraction (mm)
        
        if precipitation > Ia:
            runoff = ((precipitation - Ia) ** 2) / (precipitation - Ia + S)
        else:
            runoff = 0
        
        return runoff

    def snow_routine(self):
        """
            The function of this routine is to find out how much water is being stored and released
            in the snowpack.

            * The only function that is lacking right now is the infiltration
        """
        snow_acc = 0
        max_snow = 500 
        for day in range(self.n):
            # Store
            if self.temp[day] < self.TT:
                if snow_acc >= max_snow:
                    snow_acc = max_snow
                else:
                    snow_acc += self.snow[day]     
            # Release
            else: 
                if snow_acc > 0:
                    melt =  self.CFMAX * (self.temp[day] - self.TT) 
                    refreezing = self.CFR * melt
                    self.Q_0[day] = melt - refreezing

                snow_acc -= self.Q_0[day] # Remove all snow from storage
                if snow_acc < 0: # The accumulated snow can't be negative
                    snow_acc = 0
                    
            self.snow_pack[day] = snow_acc
        return
    
    def soil_moisture_routine(self):
        """ 
        Function to calculate the soil moisture routine.

        recharge: Water from the moist soil entering the storage in the upper reservoir (UZ)

        """

        # Initial conditions
        self.SM[0]  = 5   #  5 mm
        self.SUZ[0] = 0
        self.SLZ[0] = 0

        #Update storage
        for day in range(1, self.n):
            
            #1. Infiltration + Rain -> SM -> Recharge          
            soil_resistance = np.exp(self.SM[day-1]/self.FC - 1)
            Q_SM = (1 - soil_resistance) * (self.Q_0[day] + self.rain[day]) # If the soil moisture is high then more of the water reaches SUZ -> LOOK INTO A SCS CURVE IMPLEMENTATION
            self.recharge[day] = (self.Q_0[day] + self.rain[day] - Q_SM)

            self.SM[day] = self.SM[day-1] + Q_SM - self.evap[self.dateMonth[day]-1]
            
            if self.SM[day] >= self.FC:
                #recharge[day] = SM[day] - params['FC']
                self.SM[day] = self.FC
            elif self.SM[day] <= 5:
                self.SM[day] = 5

            #2. Recharge -> SUZ -> q0, q1 
            self.SUZ[day] = self.SUZ[day-1] + self.recharge[day] 
            overflow = 0
            if self.SUZ[day] >= self.FC:
                overflow = self.SUZ[day] - self.FC
                self.SUZ[day] = self.FC
                print(f'Overflow on day {day}: q = {overflow}')

            q0, q1, q2 = 0, 0, 0
            if self.SUZ[day] > 0:
                if self.SUZ[day] > self.UZL:
                    q0 = self.K0 * (self.SUZ[day] - self.UZL) 
                q1 = self.K1*self.SUZ[day]
                self.SUZ[day] -= (q0 + q1)
            else:
                self.SUZ[day] = 0
            
            #3. Percolation -> SLZ -> q2 
            self.SLZ[day] = self.SLZ[day-1] 
            if self.SLZ[day] > 0:
                q2 = self.K2*self.SLZ[day] 
                self.SLZ[day] -= q2
            else:
                self.SLZ[day] = 0

            self.Q_1[day] = q0 + q1 + q2 + overflow
        return
    
    def run(self):
        # Step 1. Calculate the flow from the snow pack 
        self.snow_routine()
        # Step 2-3. Run the soil moisture routine and routing function
        self.soil_moisture_routine()

        return self.Q_1

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