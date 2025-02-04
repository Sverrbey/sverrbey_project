"""
author: @Sverrbey

This file contains only the necessary code to run the application: Intelligent Streamflow Monitoring 
system. It is written in a general way so that it can be used for any reservoir one chooses to monitor.
Underneath instructions in the form of comments are provided to guide the user on how to use the code.

In order to use the application the following data is needed:
    - Precipitation data: (real-time, historical)
    - Temperature data (real-time, historical)
    - Altitude data (DEM)

**    
If multiple temporal scales are to be used, the code should include the steps where certain scales 
are chosen.
    - I think that if the script also includes a visualization that can be interpreted by the operator
    this specification would relate to what is shown by the program.

?- If the data includes some time series data that is updated with regular frequency this updating interval
needs to be specified in the code.
"""
# Import necessary models & libraries
from model.LSTM import LSTM_Model

#1. What is the research area in question? (data)
# - The file path to the data should be written here.
# - Update frequecy (time series data)(temporal scale(s))

## Input data
data_precipitation = "src/..."
data_temperature = "src/..."

## Parameter data
# Define the file path of the altitude data (.tif file)
data_altitude = "src/data/dtm50/dtm50_7002_50m_33.tif"
data_historical_precipitation = "src/..."
data_historial_temperature = "src/..."

#2. What model(s) is being used? (temporal scale(s))

## Short-term temporal scale (minutes to hours)


## Medium-term temporal scale (days to weeks)


## Long-term temporal scale (months to years)

#3. 