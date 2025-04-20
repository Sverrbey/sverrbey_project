# -*- coding: utf-8 -*-
""" 
Reading satelite data

Objective: Translating EUMENTSAT's .nat files to GTiff
In order for these packages to work the following steps need to be done
Sufficient space needs to be available on the computer to install the packages
Can take a long time (it took me over 12 hours)
1. brew install grpc :
2. brew install gdal
"""
import os
import numpy as np
import pyresample as pr
import matplotlib.pyplot as plt

from osgeo import gdal
from osgeo import osr
from satpy import Scene

file = "data/graphs/MHSx_xxx_1B_M02_20070318164551Z_20070318182450Z_N_O_20070318182500Z/MHSx_xxx_1B_M02_20070318164551Z_20070318182450Z_N_O_20070318182500Z.nat"

# define reader
reader = "avhrr_l1b_eps"
# read the file
scn = Scene(filenames = {reader:[file]})
# extract data set names
dataset_names = scn.all_dataset_names()
# print available datasets
print('\n'.join(map(str, dataset_names)))


def nat2tif(file, calibration, area_def, dataset, reader, outdir, label, dtype, radius, epsilon, nodata):
  # open the file
  scn = Scene(filenames = {reader: [file]})
  # let us check that the specified data set is actually available
  scn_names = scn.all_dataset_names()
  # raise exception if dataset is not present in available names
  if dataset not in scn_names:
    raise Exception("Specified dataset is not available.")
  # we need to load the data, different calibration can be chosen
  scn.load([dataset], calibration=calibration)
  # let us extract the longitude and latitude data
  lons, lats = scn[dataset].area.get_lonlats()
  # now we can apply a swath definition for our output raster
  swath_def = pr.geometry.SwathDefinition(lons=lons, lats=lats)
  # and finally we also extract the data
  values = scn[dataset].values
  # we will now change the datatype of the arrays
  # depending on the present data this can be changed
  lons = lons.astype(dtype)
  lats = lats.astype(dtype)
  values = values.astype(dtype)
  # now we can already resample our data to the area of interest
  values = pr.kd_tree.resample_nearest(swath_def, values,
                                             area_def,
                                             radius_of_influence=radius, # in meters
                                             epsilon=epsilon,
                                             fill_value=False)
  # we are going to check if the outdir exists and create it if it doesnt
  if not os.path.exists(outdir):
   os.makedirs(outdir)
  # let us join our filename based on the input file's basename                                           
  outname = os.path.join(outdir, os.path.basename(file)[:-4] + "_" + str(label) + ".tif")
  # now we define some metadata for our raster file
  cols = values.shape[1]
  rows = values.shape[0]
  pixelWidth = (area_def.area_extent[2] - area_def.area_extent[0]) / cols
  pixelHeight = (area_def.area_extent[1] - area_def.area_extent[3]) / rows
  originX = area_def.area_extent[0]
  originY = area_def.area_extent[3] 
  # here we actually create the file
  driver = gdal.GetDriverByName("GTiff")
  outRaster = driver.Create(outname, cols, rows, 1)
  # writing the metadata
  outRaster.SetGeoTransform((originX, pixelWidth, 0, originY, 0, pixelHeight))
  # creating a new band and writting the data
  outband = outRaster.GetRasterBand(1)
  outband.SetNoDataValue(nodata) #specified no data value by user
  outband.WriteArray(np.array(values)) # writting the values
  outRasterSRS = osr.SpatialReference() # create CRS instance
  outRasterSRS.ImportFromEPSG(4326) # get info for EPSG 4326
  outRaster.SetProjection(outRasterSRS.ExportToWkt()) # set CRS as WKT
  # clean up
  outband.FlushCache()
  outband = None
  outRaster = None

# create some information on the reference system
area_id = "Stryn"
description = "Geographical Coordinate System clipped on Stryn"
proj_id = "Stryn"
# specifing some parameters of the projection
proj_dict = {"proj": "longlat", "ellps": "WGS84", "datum": "WGS84"}
# calculate the width and height of the aoi in pixels
llx = 6.45811333 # lower left x coordinate in degrees )
lly = 61.75932639 # lower left y coordinate in degrees 
urx = 7.33417667 # upper right x coordinate in degrees 
ury = 62.06398778 # upper right y coordinate in degrees 
resolution = 0.005 # target resolution in degrees (1 km)
# calculating the number of pixels
width = int((urx - llx) / resolution)
height = int((ury - lly) / resolution)
area_extent = (llx,lly,urx,ury)
# defining the area
area_def = pr.geometry.AreaDefinition(area_id, proj_id, description, proj_dict, width, height, area_extent)
print(area_def)



nat2tif(file = file, 
        calibration = "radiance",  
        area_def = area_def,  
        dataset = "MHS", 
        reader = reader, 
        outdir = "data/graphs/MHSx_xxx_1B_M02_20070318164551Z_20070318182450Z_N_O_20070318182500Z/output",  
        label = "AVHRR_L1B_MHS", 
        dtype = "float32", 
        radius = 16000, 
        epsilon = .5, 
        nodata = -3.4E+38)

file = "data/graphs/MHSx_xxx_1B_M02_20070318164551Z_20070318182450Z_N_O_20070318182500Z/output/MHSx_xxx_1B_M02_20070318164551Z_20070318182450Z_N_O_20070318182500Z_AVHRR_L1B_MHS.tif"
ds = gdal.Open(file)
band = ds.GetRasterBand(1)
data = band.ReadAsArray()
plt.imshow(data)