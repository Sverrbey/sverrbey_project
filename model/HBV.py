import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt

# 1. Preprocessing the data and creating sequences
# Load your data

## ptq.txt 
# Precipitation, Temperature, Runoff (10 years: 1981 01 01 - 1991 12 31)
df = pd.read_csv('/Users/SverreB/Github_Repo/sverrbey_project/src/data/HBV-land/Data/ptq.txt', header=1, delimiter='\t')
df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
# Extract the month
df['month'] = df['date'].dt.month
date_month = df['month']

data_0 = df[['Prec.','Temp','Qobs']]

## EVAP.txt 
# Evaporation (1 year: Jan - Dec)
df = pd.read_csv('/Users/SverreB/Github_Repo/sverrbey_project/src/data/HBV-land/Data/EVAP.txt',header=0)
data_1 = df

## T_MEAN.txt
# Mean temperature (1 year: Jan - Dec)
df = pd.read_csv('/Users/SverreB/Github_Repo/sverrbey_project/src/data/HBV-land/Data/T_MEAN.txt')   
data_2 = df

## clarea.xml
# Extract information from XML file
tree = ET.parse('/Users/SverreB/Github_Repo/sverrbey_project/src/data/HBV-land/Data/clarea.xml')
root = tree.getroot()

# Extract information
vegetation_zone_count = int(root.find('VegetationZoneCount').text)
elevation_zone_height = float(root.find('ElevationZoneHeight/double').text)

sub_catchment = root.find('SubCatchment')
vegetation_zone = float(sub_catchment.find('VegetationZone/EVU/Area').text)
lake_area = float(sub_catchment.find('Lake/Area').text)
lake_elevation = float(sub_catchment.find('Lake/Elevation').text)
lake_tt = float(sub_catchment.find('Lake/TT').text)
lake_sfcf = float(sub_catchment.find('Lake/SFCF').text)
absolute_area = float(sub_catchment.find('AbsoluteArea').text)

## Parameter_HBVland_start.xml
# Extract information from XML file
tree = ET.parse('/Users/SverreB/Github_Repo/sverrbey_project/src/data/HBV-land/Data/Parameter_HBVland_start.xml')
root = tree.getroot()

# Extract information
catchmentParameters = root.find('CatchmentParameters')
parameter_perc = int(catchmentParameters.find('PERC').text)
parameter_alpha = int(catchmentParameters.find('Alpha').text)
parameter_uzl = float(catchmentParameters.find('UZL').text)
parameter_k0 = float(catchmentParameters.find('K0').text)
parameter_k1 = float(catchmentParameters.find('K1').text)
parameter_k2 = float(catchmentParameters.find('K2').text)
parameter_maxbas = float(catchmentParameters.find('MAXBAS').text)
parameter_cet = float(catchmentParameters.find('Cet').text)
parameter_pcalt = float(catchmentParameters.find('PCALT').text)
parameter_tcalt = float(catchmentParameters.find('TCALT').text)
parameter_pelev = float(catchmentParameters.find('Pelev').text)
parameter_telev = float(catchmentParameters.find('Telev').text)
parameter_part = float(catchmentParameters.find('PART').text)
parameter_delay = float(catchmentParameters.find('DELAY').text)

vegetationZoneParamteres = root.find('VegetationZone/VegetationZoneParameters')
v_parameter_tt = int(vegetationZoneParamteres.find('TT').text)
v_parameter_cfmax = int(vegetationZoneParamteres.find('CFMAX').text)
v_parameter_sp= int(vegetationZoneParamteres.find('SP').text)
v_parameter_sfcf= int(vegetationZoneParamteres.find('SFCF').text)
v_parameter_cfr= float(vegetationZoneParamteres.find('CFR').text)
v_parameter_cwh= float(vegetationZoneParamteres.find('CWH').text)
v_parameter_cfglacier= int(vegetationZoneParamteres.find('CFGlacier').text)
v_parameter_cfslope= int(vegetationZoneParamteres.find('CFSlope').text)
v_parameter_fc= int(vegetationZoneParamteres.find('FC').text)
v_parameter_lp= int(vegetationZoneParamteres.find('LP').text)
v_parameter_beta= int(vegetationZoneParamteres.find('BETA').text)

'''
subcatchmentParameters = root.find('SubCatchment/SubCatchmentParameters')
parameter_perc = int(subcatchmentParameters.find('PERC').text)
parameter_alpha = int(subcatchmentParameters.find('Alpha').text)
parameter_uzl = float(subcatchmentParameters.find('UZL').text)
parameter_k0 = float(subcatchmentParameters.find('K0').text)
parameter_k1 = float(subcatchmentParameters.find('K1').text)
parameter_k2 = float(subcatchmentParameters.find('K2').text)
parameter_maxbas = float(subcatchmentParameters.find('MAXBAS').text)
parameter_cet = float(subcatchmentParameters.find('Cet').text)
parameter_pcalt = float(subcatchmentParameters.find('PCALT').text)
parameter_tcalt = float(subcatchmentParameters.find('TCALT').text)
parameter_pelev = float(subcatchmentParameters.find('Pelev').text)
parameter_telev = float(subcatchmentParameters.find('Telev').text)
parameter_part = float(subcatchmentParameters.find('PART').text)
parameter_delay = float(subcatchmentParameters.find('DELAY').text)

subcatchmentvegetationZoneParamteres = root.find('SubCatchment/SubCatchmentVegetationZone/SubCatchmentVegetationZoneParameters')
v_parameter_tt = int(subcatchmentvegetationZoneParamteres.find('TT').text)
v_parameter_cfmax = int(subcatchmentvegetationZoneParamteres.find('CFMAX').text)
v_parameter_sp= int(subcatchmentvegetationZoneParamteres.find('SP').text)
v_parameter_sfcf= int(subcatchmentvegetationZoneParamteres.find('SFCF').text)
v_parameter_cfr= float(subcatchmentvegetationZoneParamteres.find('CFR').text)
v_parameter_cwh= float(subcatchmentvegetationZoneParamteres.find('CWH').text)
v_parameter_cfglacier= int(subcatchmentvegetationZoneParamteres.find('CFGlacier').text)
v_parameter_cfslope= int(subcatchmentvegetationZoneParamteres.find('CFSlope').text)
v_parameter_fc= int(subcatchmentvegetationZoneParamteres.find('FC').text)
v_parameter_lp= int(subcatchmentvegetationZoneParamteres.find('LP').text)
v_parameter_beta= int(subcatchmentvegetationZoneParamteres.find('BETA').text)
'''

def snow_routine(temp, snow):
    Q_0 = np.zeros(len(temp)) 
    snow_pack = np.zeros(len(temp))
    snow_acc = 0
    max_snow = 500 
    for day in range(len(temp)):
        # Store
        if temp[day] < v_parameter_tt:
            if snow_acc >= max_snow:
                snow_acc = max_snow
            else:
                snow_acc += snow[day]
        # Release
        else: 
            if snow_acc > 0:

                melt =  v_parameter_cfmax * (temp[day] - v_parameter_tt) 
                refreezing = v_parameter_cfr * melt
                Q_0[day] = melt - refreezing
                snow_acc -= Q_0[day] # Remove all snow from storage
        snow_pack[day] = snow_acc
    return Q_0, snow_pack

def soil_moisture_routine(recharge, rain, evaporation):
    """ 
    Function to calculate the soil moisture routine.
    """

    #Variables: soil_moisture_storage
    SUZ = np.zeros(len(rain)) 
    SLZ = np.zeros(len(rain)) 
    Q_1 = np.zeros(len(rain)) 
    SUZ[0], SLZ[0] = rain[0], rain[0]
    #Update storage
    for day in range(1, len(rain)):
        
        SUZ[day] = SUZ[day-1] + recharge[day] 
        
        if SUZ[day] >= v_parameter_fc:
            SUZ[day] = v_parameter_fc
  
        SLZ[day] = SLZ[day-1] + rain[day] - evaporation['Nopex_evap'][date_month[day]-1]
        
        q0, q1, q2 = 0, 0, 0
        if SUZ[day] > 0:
            if SUZ[day] > parameter_uzl:
                q0 = parameter_k0 * (SUZ[day] - parameter_uzl) 
            q1 = parameter_k1*SUZ[day]
            SUZ[day] -= (q0 + q1)
        else:
            SUZ[day] = 0
        if SLZ[day] > 0:
            q2 = parameter_k2*SLZ[day] 
            SLZ[day] -= q2
        else:
            SLZ[day] = 0

        Q_1[day] = q0 + q1 + q2
    return Q_1


prec = data_0['Prec.']
temp = data_0['Temp']
evaporation = data_1

snow = np.zeros(len(prec))
rain = np.zeros(len(prec))
for i in range(len(temp)):
    if temp[i] > v_parameter_tt:
        rain[i] = prec[i]
    else:
        snow[i] = prec[i]
    

Q_0, snow_pack = snow_routine(temp, snow)
Q_1 = soil_moisture_routine(Q_0, rain, evaporation)


ys = data_0['Qobs']
xs = np.linspace(0, len(ys), len(ys))
fig, axs = plt.subplots(2, 2, figsize=(10, 8))

result = ys - Q_1

# First plot
#axs[0, 0].step(xs, ys, label='Observed Runoff')
#axs[0, 0].step(xs, Q_1, label='Simulated Runoff')
axs[0, 0].plot(xs, result, label='error', color='r')
axs[0, 0].set_xlabel('Days')
axs[0, 0].set_ylabel('Runoff')
axs[0, 0].legend()
axs[0, 0].set_title('Runoff Comparison')

# Second plot
axs[0, 1].plot(xs, temp, label='Temperature')
axs[0, 1].set_xlabel('Days')
axs[0, 1].set_ylabel('Temperature')
axs[0, 1].axhline(y=0, color='r', linestyle='--')  # Add horizontal line at y=0
axs[0, 1].legend()
axs[0, 1].set_title('Temperature Over Time')

# Third plot
axs[1, 0].plot(xs, rain, label='Precipitation')
axs[1, 0].set_xlabel('Days')
axs[1, 0].set_ylabel('Precipitation')
axs[1, 0].axhline(y=0, color='r', linestyle='--')  # Add horizontal line at y=0
axs[1, 0].legend()
axs[1, 0].set_title('Rainfall')

# Fourth plot
axs[1, 1].step(xs, snow_pack, label='Snow')
axs[1, 1].set_xlabel('Days')
axs[1, 1].set_ylabel('Snow')
axs[1, 1].axhline(y=0, color='r', linestyle='--')  # Add horizontal line at y=0
axs[1, 1].legend()
axs[1, 1].set_title('Snowpack')

plt.tight_layout()
plt.show()
