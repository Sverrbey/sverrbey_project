import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors
import seaborn as sns
from pysheds.grid import Grid

import rasterio
from rasterio.enums import Resampling

import timeit

import pandas as pd
import geopandas as gpd
from shapely import geometry, ops

import rasterio
import fiona
from fiona.crs import from_epsg
import json
from pysheds.grid import Grid

""" Description of how the two classes interact
    The SWAT model is a hydrological model that simulates the hydrological cycle in a watershed. The model is based on the water balance equation, 
    which states that the change in soil moisture is equal to the sum of precipitation, runoff, evapotranspiration, and percolation. The SWAT model 
    is a complex model that requires a large amount of input data, including climate data, soil properties, land use, and topography. The model is 
    typically used to simulate the effects of land use changes, climate change, and other factors on the hydrological cycle.


    # Parent: Map_Unit
    # Child SWAT_Model  

    THE SWAT_Model class inherents the Map_Unit class. Because, for each of the Map_Unit objects the SWAT model is run. And when we want to run and
    display the model, it would be practical to have the information stored in the Map_Unit class. Especially when we want to model the snow store

    The Map_Unit class is meant to model an area in the catchment. For a small spatial scale resolution, the model
    only uses 1 of these units. Then when the model is expanded to a larger area a collection of these units are 
    used.
"""

def Main ():
    # Define model parameters
    # We are looking at a theroetical catchment area of 1 square meter
    parameters = {
        "CN":     50,  # SCS Curve Number (30-100)
        "soil_capacity":        400,  # mm
        "percolation_rate":     2,  # mm/day
        "initial_soil_moisture":5,  # mm
        "elevation":        'src/data/lookout_creek_simplified_10.tif',
        "soils":            None,
        "days":             365,
        "precipitation":    None,
        "temperature":      None,
        "wind_speed":       None,
        "solar_radiation":  None,
        "relative_humidity": None,
        "Evapotranspiration": None,
        "SW_initial":       0,
        "SM_initial":       0,
        "seepage":          None
        }

   
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

    # Generate climate data

    days = 365  # Simulate for one year
    precipitation = data_0['Prec.'][0:days]
    temperature = data_0['Temp'][0:days]
    potential_et = np.zeros(days)
    for day in range(0, days):
        potential_et[day] = data_1['Nopex_evap'][date_month[day]-1]

    parameters['Evapotranspiration'] = potential_et
    parameters['precipitation'] = precipitation
    parameters['temperature'] = temperature
    qobs = data_0['Qobs'][0:days]
    # Run the model
    swat = SWAT_Model(params=parameters)
    
    swat.run(qobs)

    #swat = SimplifiedSWAT(params=parameters, time_steps=days)
    #swat.simulate(climate_data)

    

    # Plot results
    #swat.plot_results(ys)
    
    #simplify_tiff(inn_file, out_file, scale_factor=10)
    

class Map_Unit:
    def __init__(self, params):
        self.latitude   = None
        self.longitude  = None
        self.elevation  = params['elevation']   # .tiff file containing the Lidar points
        self.soils      = params['soils']       # .shp file containing the soil types
        self.area       = None
        self.land_use       = None
        self.soil_taxonomy  = None
        self.lai        = None # Leaf Area Index
        self.grid           = None
        self.branches       = None

class SWAT_Model(Map_Unit):
    def __init__(self, params):
        super().__init__(params)
        # Sets
        self.days = params['days']

        # Input
        self.precipitation  = params['precipitation']   # mm/day
        self.temperature    = params['temperature']    # C/day

        # Parameters
        self.CN                 = params['CN']
        self.soil_capacity      = params['soil_capacity']
        self.percolation_rate   = params['percolation_rate']
        self.wind_speed         = params['wind_speed']
        self.solar_radiation    = params['solar_radiation']
        self.relative_humidity  = params['relative_humidity']
        self.et                 = params['Evapotranspiration']
        self.soil_water         = np.zeros(self.days)
        self.soil_water[0]      = params['SW_initial']
        self.soil_moisture      = np.zeros(self.days)   
        self.soil_moisture[0]   = params['SM_initial']  
        self.percolation        = np.zeros(self.days)

        self.runoff_surface = self.curve_number_runoff(self.precipitation)
        self.soil_moisture, self.percolation = self.soil_water_balance(self.precipitation, self.runoff_surface, self.et) 
        self.runoff_ground  = self.calculate_groundwater_runoff()  

    def __str__(self):
        return f"PRINTOUT SWAT"
    

    def water_balance(self):
        SW_t = self.soil_water
        for t in range(1, self.days):
            if self.temperature[t] > 0:
                #water_balance[t] = self.soil_water_0 + sum(self.precipitation[i] - self.runoff_surface[i] - self.et[i] - self.seepage[i] - self.runoff_ground[i] for i in range(1,t)) 
                SW_t[t] = SW_t[0] + sum(self.precipitation[i] - self.runoff_surface[i] - self.runoff_ground[i] - self.et[i] for i in range(1,t))
            else:
                SW_t[t] = SW_t[t-1]
        self.soil_water = SW_t
    
    def soil_water_balance(self, precipitation, runoff, evapotranspiration):
        """
        Update soil moisture based on precipitation, runoff, and ET.
        :param precipitation: Daily precipitation (mm).
        :param runoff: Surface runoff (mm).
        :param evapotranspiration: Actual evapotranspiration (mm).
        :return: Updated soil moisture (mm) and percolation (mm).
        """    
        # Update soil moisture
        net_precipitation = precipitation - runoff - evapotranspiration
        soil_moisture = self.soil_moisture
        percolation = self.percolation
        for t in range(1, self.days):
            soil_moisture[t] = soil_moisture[t-1] + net_precipitation[t]
        
            # Percolation
            if soil_moisture[t] > self.soil_capacity:
                percolation[t] = min(soil_moisture[t] - self.soil_capacity, self.percolation_rate)
                soil_moisture[t] = self.soil_capacity 
            else:
                percolation[t] = 0

            
        
        return soil_moisture, percolation

    def calculate_evapotranspiration(self, temperature, solar_radiation, wind_speed, relative_humidity):
        """
        Calculate potential evapotranspiration using the Penman-Monteith equation.
        :param temperature: Daily temperature (C).
        :param solar_radiation: Daily solar radiation (MJ/m^2).
        :param wind_speed: Daily wind speed (m/s).
        :param relative_humidity: Daily relative humidity (%).
        :return: Potential evapotranspiration (mm).
        """
        # Constants
        G = 0  # Soil heat flux density (MJ/m^2/day)
        gamma = 0.665 * 10**-3 * 101.3  # Psychrometric constant (kPa/C)
        delta = 4098 * (0.6108 * np.exp((17.27 * temperature) / (temperature + 237.3))) / (temperature + 237.3)**2  # Slope of vapor pressure curve (kPa/C)
        Rn = solar_radiation  # Net radiation (MJ/m^2/day)
        es = 0.6108 * np.exp((17.27 * temperature) / (temperature + 237.3))  # Saturation vapor pressure (kPa)
        ea = es * (relative_humidity / 100)  # Actual vapor pressure (kPa)
        
        # Penman-Monteith equation
        et = (0.408 * delta * (Rn - G) + gamma * (900 / (temperature + 273)) * wind_speed * (es - ea)) / (delta + gamma * (1 + 0.34 * wind_speed))
        
        return et

    def hydrologic_response_unit(self):
        return

    def run(self, qobs):
        """ When called upon this function runs the SWAT model, and is made accordring to the flow model of the SWAT. 
        
            For each day t the model uses the water balance
        """
        # 1. Water Delineation
        self.watershed_delineation()

        # 2. Water Balance
        self.water_balance()

        # Calculate surface runoff
        runoff = self.runoff_surface
        
        # Calculate evapotranspiration
        #actual_et = self.evapotranspiration(potential_et, soil_moisture)
        
        # Update soil water balance
        #soil_moisture, percolation = self.soil_water_balance(self.precipitation, self.runoff, actual_et)
        
        # Simulate lateral flow and streamflow (simplified)
        streamflow = abs(self.runoff_surface - self.soil_water)
        xs = np.linspace(0, self.days, self.days)
        plt.step(xs, streamflow, label='Simulated Streamflow')
        plt.step(xs, qobs, label='Observed Streamflow')
        plt.legend(loc='upper right')
        plt.title('Streamflow')
        plt.xlabel('Days')
        plt.ylabel('Streamflow (mm/day)')
        plt.show()
        

        return

    def watershed_delineation(self):
        """
            Watershed delineation is the process of identifying the boundary of a watershed, also reffered to as a catchment, drainage basin, or river basin. 
            It's an important step in many areas of environmental science, engineering, and management, for example to study flooding, aquatic habitat, or water
            pollution.
            Source: https://mattbartos.com/pysheds/
        
        """
        # Create a grid from the DEM
        grid = Grid.from_raster(self.elevation, data_name='dem')
        dem = grid.read_raster(self.elevation)
        
        # Condition DEM
        # ----------------

        # Fill pits in DEM
        pit_filled_dem = grid.fill_pits(dem)
        # Fill depressions in DEM
        flooded_dem = grid.fill_depressions(pit_filled_dem)
        #Resolve flats in DEM
        inflated_dem = grid.resolve_flats(flooded_dem)

        # Determine flow directions from DEM 
        # ----------------

        # Specify directional mapping, (By default, the ESRI scheme is used)
        dirmap = (64, 128, 1, 2, 4, 8, 16, 32)
        # Compute D8 flow directions (Options: D-infinity & MFD)
        fdir = grid.flowdir(inflated_dem, dirmap=dirmap)
    
        # Calculate flow accumulation
        # ----------------

        acc = grid.accumulation(fdir, dirmap=dirmap)
        
        # Delineate a catchment
        # ----------------

        # Specify pour point                     
        x, y = 5.5893e+05, 4.89489e+06  # River Inlet
        # Snap pour point to high accumulation cell
        x_snap, y_snap = grid.snap_to_mask(acc > 1000, (x,y))
        # Delineate the catchment
        catch = grid.catchment(x=x_snap, y=y_snap, fdir=fdir, dirmap=dirmap, xytype='coordinate')

        # Crop and plot the catchment
        # ----------------

        # Clip the bounding box to the catchment
        grid.clip_to(catch)
        clipped_catch = grid.view(catch)

        #Extract river network
        # ----------------

        branches = grid.extract_river_network(fdir, acc > 50, dirmap=dirmap)
        
        # Calculate distance to outlet from each cell
        # ----------------  

        dist = grid.distance_to_outlet(x=x_snap, y=y_snap, fdir=fdir, dirmap=dirmap, xytype='coordinate')
                
        # Combine with land cover data
        #self.soil_raster(grid)
        
        #soil_polygons = zip(catchment_soils.geometry.values, catchment_soils[soil_id].values)
        #soil_raster = grid.rasterize(soil_polygons, fill=np.nan)

        # Plotting 
        #self.plot_DEM(grid, dem) # the digital elevation model
        #self.plot_flow_direction(grid, fdir, dirmap) # the flow direction
        #self.plot_flow_accumulation(grid, acc) # the flow accumulation
        #self.plot_surface(acc)
        #self.plot_catchment(grid, catch, clipped_catch) # the catchment
        #self.plot_branches(grid, branches) #  the river network
        #self.plot_flow_distance(self, grid, dist) # the flow distance
    
    def curve_number_runoff(self, precipitation):
        """
        Calculate surface runoff using the SCS Curve Number method.
        :param precipitation: Daily precipitation (mm).
        :param soil_moisture: Current soil moisture (mm).
        :return: Surface runoff (mm).
        """
        CN = self.CN
        S = 25.4 * (1000 / CN - 10)  # Potential maximum retention (mm)
        Ia = 0.2 * S  # Initial abstraction (mm)
        
        runoff = np.zeros(self.days)
        for t in range(0, self.days):
            if precipitation[t] > Ia:
                runoff[t] = ((precipitation[t] - Ia) ** 2) / (precipitation[t] - Ia + S)
            else:
                pass
        return runoff
    
    def calculate_groundwater_runoff(self):
        """
        Calculate groundwater runoff based on soil moisture and percolation.
        :param soil_moisture: Daily soil moisture (mm).
        :param percolation: Daily percolation (mm).
        :return: Groundwater runoff (mm).
        """
        groundwater_runoff = np.zeros(self.days)
        for t in range(self.days):
            if self.soil_moisture[t] > self.soil_capacity:
                groundwater_runoff[t] = self.percolation[t]
            else:
                groundwater_runoff[t] = 0
        return groundwater_runoff
    
    def plot_DEM(self, grid, dem):
        fig, ax = plt.subplots(figsize=(8,6))
        fig.patch.set_alpha(0)

        plt.imshow(dem, extent=grid.extent, cmap='terrain', zorder=1)
        plt.colorbar(label='Elevation (m)')
        plt.grid(zorder=0)
        plt.title('Digital elevation map', size=14)
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.tight_layout()
        plt.show()

    def plot_flow_direction(self, grid, fdir, dirmap):
        fig = plt.figure(figsize=(8,6))
        fig.patch.set_alpha(0)

        plt.imshow(fdir, extent=grid.extent, cmap='viridis', zorder=2)
        boundaries = ([0] + sorted(list(dirmap)))
        plt.colorbar(boundaries= boundaries,
                    values=sorted(dirmap))
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.title('Flow direction grid', size=14)
        plt.grid(zorder=-1)
        plt.tight_layout()
        plt.show()

    def plot_flow_accumulation(self, grid, acc):
        fig, ax = plt.subplots(figsize=(8,6))
        fig.patch.set_alpha(0)
        plt.grid('on', zorder=0)
        im = ax.imshow(acc, extent=grid.extent, zorder=2,
                    cmap='cubehelix',
                    norm=colors.LogNorm(1, acc.max()),
                    interpolation='bilinear')
        plt.colorbar(im, ax=ax, label='Upstream Cells')
        plt.title('Flow Accumulation', size=14)
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.tight_layout()
        plt.show()

    def plot_catchment(self, grid, catch, clipped_catch):
        fig, ax = plt.subplots(figsize=(8,6))
        fig.patch.set_alpha(0)

        plt.grid('on', zorder=0)
        im = ax.imshow(np.where(clipped_catch, clipped_catch, np.nan), extent=grid.extent,
                    zorder=1, cmap='Greys_r')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.title('Delineated Catchment', size=14)
        plt.show()

    def plot_branches(self, grid, branches):
        sns.set_palette('husl')
        fig, ax = plt.subplots(figsize=(8.5,6.5))
        plt.xlim(grid.bbox[0], grid.bbox[2])
        plt.ylim(grid.bbox[1], grid.bbox[3])
        ax.set_aspect('equal')

        for branch in branches['features']:
            line = np.asarray(branch['geometry']['coordinates'])
            plt.plot(line[:, 0], line[:, 1],0)
            
        plt.title('D8 channels', size=14)
        plt.show()

    def plot_flow_distance(self, grid, dist):
        fig, ax = plt.subplots(figsize=(8,6))
        fig.patch.set_alpha(0)
        plt.grid('on', zorder=0)
        im = ax.imshow(dist, extent=grid.extent, zorder=2,
                    cmap='cubehelix_r')
        plt.colorbar(im, ax=ax, label='Distance to outlet (cells)')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.title('Flow Distance', size=14)
        plt.show()
    
    def plot_surface(self, data, plane_z = 0):
        # Create a meshgrid for the x and y coordinates
        x = np.arange(data.shape[1])
        y = np.arange(data.shape[0])
        x, y = np.meshgrid(x, y)

        # Create a figure and a 3D axis
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Plot the surface
        ax.plot_surface(x, y, data, cmap='viridis')

        # Plot the plane if plane_z is specified
        if plane_z is not None:
            # Create a plane at z = plane_z
            plane = np.full_like(data, plane_z)
            ax.plot_surface(x, y, plane, color='r', alpha=0.5)
        
        # Add labels
        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')
        ax.set_zlabel('Z axis')
        ax.set_title('3D Surface Plot')
        
        # Show the plot
        plt.show()
    
    def soil_raster(self, grid):
        # ---------------------
        terrain = grid.read_raster(self.elevation, window=grid.bbox,
                                window_crs=grid.crs, nodata=0)
        # Reproject data to grid's coordinate reference system
        projected_terrain = terrain.to_crs(grid.crs)
        # View data in catchment's spatial extent
        catchment_terrain = grid.view(projected_terrain, nodata=np.nan)

        # Convert catchment raster to vector and combine with soils shapefile
        # ---------------------
        # Read soils shapefile
        soils = gpd.read_file(self.soils)
        soil_id = 'MUKEY'
        # Convert catchment raster to vector geometry and find intersection
        shapes = grid.polygonize()
        catchment_polygon = ops.unary_union([geometry.shape(shape)
                                            for shape, value in shapes])
        soils = soils[soils.intersects(catchment_polygon)]
        catchment_soils = gpd.GeoDataFrame(soils[soil_id], 
                                        geometry=soils.intersection(catchment_polygon))
        # Convert soil types to simple integer values
        soil_types = np.unique(catchment_soils[soil_id])
        soil_types = pd.Series(np.arange(soil_types.size), index=soil_types)
        catchment_soils[soil_id] = catchment_soils[soil_id].map(soil_types)

        fig, ax = plt.subplots(figsize=(8, 6))
        catchment_soils.plot(ax=ax, column=soil_id, categorical=True, cmap='terrain',
                            linewidth=0.5, edgecolor='k', alpha=1, aspect='equal')
        ax.set_xlim(grid.bbox[0], grid.bbox[2])
        ax.set_ylim(grid.bbox[1], grid.bbox[3])
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        ax.set_title('Soil types (vector)', size=14)
        plt.show()
    

Main()