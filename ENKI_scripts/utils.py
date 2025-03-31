import os
from datetime import datetime, timedelta
from osgeo import gdal, osr
import numpy as np
from netCDF4 import Dataset, Variable
from osgeo.gdalconst import CE_None
import pandas as pd


''' CF convention summary; 
see http://cfconventions.org/Data/cf-conventions/cf-conventions-1.8/cf-conventions.pdf

The general description of a file’s contents should be contained in the following attributes: title ,
history , institution , source , comment and references ( Section 2.6.2, "Description of file contents" ).

Each variable in a netCDF file has an associated description which is provided by the attributes
units , long_name , and standard_name . The units , and long_name attributes are defined in the NUG
and the standard_name attribute is defined in this document.

Standard names:
http://cfconventions.org/Data/cf-standard-names/current/src/cf-standard-name-table.xml

'''

timetypes = {"excel":   [datetime(1899,12,30),  timedelta(days=1), "days"], 
             "seNorge": [datetime(1900,1,1),    timedelta(hours=1), "hours"],
             "unix":    [datetime(1970,1,1),    timedelta(seconds=1), "seconds"]   }


def get_utm_wkt(zone):
    """Return the WKT of UTM with the supplied zone"""
    srs = osr.SpatialReference()
    srs.SetUTM(int(zone),True)
    srs.SetWellKnownGeogCS('WGS84')
    return srs.ExportToWkt()



def get_projstring_wkt(projstring):
    """Return the WKT of the supplied projstring"""
    srs = osr.SpatialReference()
    srs.ImportFromProj4(str(projstring))
    return srs.ExportToWkt()


def GetXYcoords(GDALDS):

    if GDALDS.RasterCount > 0:              # This is a raster
        '''Returns lists of cell center xcoord and ycoord, from GDAL cell edge '''
        GT = GDALDS.GetGeoTransform()       # We assume NSEW-aligned (unrotated) grid.
        nx = GDALDS.RasterXSize
        ny = GDALDS.RasterYSize
        WestC  = GT[0]         + GT[1]/2    # Cell center Xcoords along western column
        EastE  = GT[0] + nx*GT[1]           # Cell edge Xcoords along eastern column
        NorthC = GT[3]         + GT[5]/2    # Cell center Ycoords along northern row
        SouthE = GT[3] + ny*GT[5]           # Cell edge Ycoords along southern row
        Xcoords = list(np.arange(WestC, EastE, GT[1]))     # EastE is not included
        Ycoords = list(np.arange(NorthC, SouthE, GT[5]))   # SouthE is not included
    else:                                   # This is a vector data set (only points supported for now)
        Xcoords = [feat.GetGeometryRef().GetPoint()[0] for feat in GDALDS.GetLayer()]
        Ycoords = [feat.GetGeometryRef().GetPoint()[1] for feat in GDALDS.GetLayer()]

    return Xcoords, Ycoords



def SetGDALMetadata(gdalDS, varname, stdname, unit, flagcode):
    gdalDS.SetMetadataItem('name',varname)
    gdalDS.SetMetadataItem('standard_name',stdname)
    gdalDS.SetMetadataItem('units',unit)
    gdalDS.SetMetadataItem('missing_value',str(flagcode))



def SetNcMetaData(ncVar, stdname, units, grd_mapping_vname):
    '''Set the most common attributes for NcVar metfields'''
    ncVar.standard_name = stdname
    ncVar.units = units
    ncVar.grid_mapping = grd_mapping_vname
    ncVar.coordinates = "longitude latitude"

def GetNcMetaData(ncVar):
    '''Get the most common attributes for NcVar metfields'''
    return [ncVar.name, ncVar.standard_name, ncVar.units, ncVar.grid_mapping]


def MemDSfromGISfile(fname, nbands):
    """Create a GDAL memory data set (ratser only) with geometry and projection as in fname"""

    FileDS = gdal.Open(fname)
    assert FileDS is not None, f"MemDSfromGISfile could not open GDALDataset from {fname}"
    FileWKT = FileDS.GetProjectionRef() # Should be GetSpatialRef for vector data
    FileGT = FileDS.GetGeoTransform()
    nCols = FileDS.RasterXSize
    nRows = FileDS.RasterYSize
    
    MemDriver = gdal.GetDriverByName('MEM')
    memDS = MemDriver.Create("", nCols, nRows, nbands, gdal.GDT_Float32)
    memDS.SetProjection(FileWKT)              # Same CS and cell size
    memDS.SetGeoTransform(FileGT)
    return memDS


def MemDSfromInfo(nRows, nCols, nbands, xylim, wktstr="", projstr=""):
    """Create a GDAL memory data set with geometry and projection as given"""
    # xylim order is West, East, North, South; cell center coordinates
    # projstring may be a proj.4 string, or a WKT (recognised by containing 'PARAMETER')
    # In the output memory GDALDataset, the geotransform has edge coordinates.

    if wktstr=="" and projstr=="":
        return None
    dx = (xylim[1]-xylim[0]) / (nCols-1)    # (CCxmax-CCxmin) / (nx-1) 
    dy = (xylim[3]-xylim[2]) / (nRows-1)    # (CCymin-CCymax) / (ny-1) 
    MemDriver = gdal.GetDriverByName('MEM')
    memDS = MemDriver.Create("", nCols, nRows, nbands, gdal.GDT_Float32)
    FileGT = [  xylim[0] - dx/2.,           # xmin, west edge of NW pixel
                dx,                         # dx/dcol, usually positive
                0.0,                        # dx/drow, zero for aligned grid
                xylim[2] - dy/2.,           # ymax, norh edge of NW pixel
                0.0,                        # dy/dcol, zero for aligned grid
                dy                ]         # dy/drow, usually negative
    memDS.SetGeoTransform(FileGT)

    if wktstr != "":                        # WKT supplied
        memDS.SetProjection(wktstr)         
    else:
        SrcSRS = osr.SpatialReference()     # WKT not supplied, use projstr.
        SrcSRS.ImportFromProj4(str(projstr))
        memDS.SetProjection(SrcSRS.ExportToWkt())
    return memDS


def CreateLongLatMatrices(GDALDS):      # (nRows, nCols, GT, SrcSRS):
    ''' Return Wlon and Wlat; np-arrays from raster GDALDS, np-vectors from a point-cover GDALDS'''

    assert GDALDS is not None, f"utils::CreateLongLatMatrices received an empty GDALDataset"

    raster = GDALDS.RasterCount > 0

    LLSRS  = osr.SpatialReference()     
    LLSRS.ImportFromEPSG(4326)          # Long-Lat geodetic; ellipsiod and datum both WGS84.
    SrcSRS = osr.SpatialReference()
    if raster:
        SrcSRS.ImportFromWkt(GDALDS.GetProjectionRef()) 
    else:
        SrcSRS = GDALDS.GetLayer().GetSpatialRef()

    WndLLtrans = osr.CoordinateTransformation(SrcSRS, LLSRS)

    nx = GDALDS.RasterXSize    if raster else GDALDS.GetLayer().GetFeatureCount()
    ny = GDALDS.RasterYSize    if raster else 1

    z = 0.0     # Elevation of non-reprojected data

    if raster:
        WndCCxcoords, WndCCycoords = GetXYcoords(GDALDS)
        WLon = np.ones((ny,nx),np.float32)
        WLat = np.ones((ny,nx),np.float32)
        for yindex, y in enumerate(WndCCycoords):
            for xindex, x in enumerate(WndCCxcoords):
                (ptlat, ptlon, ptelev) = WndLLtrans.TransformPoint(float(x),float(y),z)
                WLon[yindex][xindex], WLat[yindex][xindex] = ptlon, ptlat
    else:
        WndCCxcoords = [feat.GetGeometryRef().GetPoint()[0] for feat in GDALDS.GetLayer()]
        WndCCycoords = [feat.GetGeometryRef().GetPoint()[1] for feat in GDALDS.GetLayer()]
        WLon = np.ones(nx,np.float32)
        WLat = np.ones(nx,np.float32)
        for index, x in enumerate(WndCCxcoords):
            y = WndCCycoords[index]
            (ptlat, ptlon, ptelev) = WndLLtrans.TransformPoint(float(x),float(y),z)
            WLon[index], WLat[index] = ptlon, ptlat

    return WLon, WLat


def createMappingVar(ncfile, SR):
    ''' Add to ncfile an ncvar with attributes holding the info in SR
    
    ncfile:         NetCDF file. Must exist
    SR:             OSR SpatialReference
    return value:   None
    '''
    mapping_name = SR.GetAttrValue('projection').lower()
    ncfile.grid_mapping = mapping_name   # Making grid_mapping global is not CF, but eases lookup.
    rsVar = ncfile.createVariable(mapping_name,'S1')
    rsVar.grid_mapping_name = mapping_name
    rsVar.longitude_of_central_meridian = SR.GetProjParm('central_meridian')
    rsVar.false_easting = SR.GetProjParm('false_easting') # 500000.0
    rsVar.false_northing = SR.GetProjParm('false_northing') # 0.0
    rsVar.latitude_of_projection_origin = SR.GetProjParm('latitude_of_projection_origin')
    rsVar.scale_factor_at_central_meridian = SR.GetProjParm('scale_factor')
    rsVar.longitude_of_prime_meridian = SR.GetProjParm('longitude_of_projection_origin')
    rsVar.semi_major_axis = SR.GetSemiMajor() # 6378137.0
    rsVar.inverse_flattening = SR.GetInvFlattening() # 298.257223563
    rsVar.crs_wkt = SR.ExportToWkt()




def CreatePtnetworkVar(NcFile, ds):
    mapping_name = NcFile.grid_mapping
    assert ds.GetLayerCount() > 0
    name = ds.GetMetadataItem('name')
    layer = ds.GetLayer()
    IDs = [int(feat.GetField('ID')) for feat in layer]
    try:
        ptnames = [feat.GetField('Navn').encode('utf-8', 'surrogateescape').decode('ISO-8859-1') for feat in layer]
    except:
        ptnames = [f"ID{id}" for id in IDS]
    xcoords = [feat.GetGeometryRef().GetPoint()[0] for feat in layer]
    ycoords = [feat.GetGeometryRef().GetPoint()[1] for feat in layer]
    assert len(IDs) > 0

    flag = float(ds.GetMetadataItem('missing_value'))

    if 'parset' in NcFile.variables:            # The NetCDF file is an Enki time series database
        dimnames = ("time", "parset", "ldtime", "member", f"{name}Indx")  
    else: dimnames = ("time", f"{name}Indx")    # A general NetCDF file, skip version dimensions

    ntwdim      = NcFile.createDimension(f"{name}Indx", len(IDs))
    dsVar       = NcFile.createVariable(name, 'f4', dimnames, zlib=True, 
                                        least_significant_digit=1, fill_value=flag)
    pVidx	    = NcFile.createVariable(f"{name}Indx",'i4', (f"{name}Indx"))     # PrecipitationIndx
    pVid        = NcFile.createVariable(f"{name}ID", 'i4', (f"{name}Indx"))	    # PrecipitationID
    ncXcoord    = NcFile.createVariable(f"{name}X", 'f8', (f"{name}Indx"))	    # PrecipitationX
    ncYcoord    = NcFile.createVariable(f"{name}Y", 'f8', (f"{name}Indx"))      # PrecipitationY
    lonVar      = NcFile.createVariable(f"{name}_lon", 'f8', (f"{name}Indx"))	# Precipitation_lon
    latVar      = NcFile.createVariable(f"{name}_lat", 'f8', (f"{name}Indx"))	# Precipitation_lat
    nameVar     = NcFile.createVariable(f"{name}Names",str, (f"{name}Indx"))	# PrecipitationNames

    dsVar.units = ds.GetMetadataItem('units')
    dsVar.standard_name = ds.GetMetadataItem('standard_name')
    dsVar.grid_mapping = mapping_name 
    ncXcoord.units = 'm'
    ncXcoord.standard_name = 'projection_x_coordinate'
    ncXcoord[:] = xcoords
    ncYcoord.units = 'm'
    ncYcoord.standard_name = 'projection_y_coordinate'
    ncYcoord[:] = ycoords
    pVidx[:] = range(len(IDs))
    pVid[:] = IDs
    WLon, WLat = CreateLongLatMatrices(ds)
    lonVar.grid_mapping = mapping_name
    lonVar[:] = WLon
    latVar.grid_mapping = mapping_name
    latVar[:] = WLat
    npnamearr = np.empty(len(ptnames),"O")
    for i in range(len(npnamearr)):
        npnamearr[i] = ptnames[i]
    nameVar[:] = npnamearr
    print(nameVar)
    for i in range(nameVar.shape[0]):
        print (nameVar[i])

    return dsVar


def CreateNewEnkiNcFile(fname, first, dt, nt, npar, nldt, nmem, ds, ttype="unix"):
    '''
    fname:  str. Path to the new netCDF file
    first:  datetime.datetime. Rescales to the first time coordinate
    dt:     datetime.timedelta. Rescales to time step length
    nt:     int. Number of temporal steps in the supplied data
    npar:   int. Size of parset dimension, possibly 0
    nldt:   int. Size of ldtime dimension, possibly 0
    nmem:   int. Size of member dimension, possibly 0
    ds:     GDALDataset. Contains either raster or vector data
    ttype:  timetypes key ('unix', 'seNorge', 'excel') 
    '''

    if ttype not in timetypes:
        return f"Unsupported time type {ttype} (must be 'excel', 'seNorge', or 'unix')"
    # Expand the temporal coordinate range into a numpy array according to ttype
    print("type(first) = ", type(first), "type(timetypes[ttype][0] = ", type(timetypes[ttype][0]))
    dtfirst = first   - timetypes[ttype][0]     # A timedelta from ttype's datum to first
    tstart  = dtfirst / timetypes[ttype][1]     # The same duration in days, hours or seconds 
    deltat  = dt      / timetypes[ttype][1]     # Time resolution of data
    tcoords = tstart  + np.arange(nt) * deltat  # Time coords in ttype's formulation
    nx = ny = 0
    IDs = None

    varname = ds.GetMetadataItem('name') 

    if ds.RasterCount > 0:
        testSR = osr.SpatialReference()
        testSR.ImportFromWkt(ds.GetProjectionRef()) # ...since this is a raster data set
    elif ds.GetLayerCount() > 0:
        testSR = ds.GetLayer().GetSpatialRef()      # ...since this is a vector data set

    mapping_name = testSR.GetAttrValue('projection').lower()

    # Open the NetCDF output file for writing, replacing if it already exists
    TrgNcFile = Dataset(fname,'w')
    TrgNcFile.Conventions = "CF-1.6"        # Some global parametters
    TrgNcFile.institution = "Enki hydrologi"
    TrgNcFile.grid_mapping = mapping_name   # This global attribute is not CF, but eases lookup.
    createMappingVar(TrgNcFile, testSR)     # Create a CF-conforming netcdf refsyst variable 

    # Create temporal dimension and coordinate variable
    TrgNcFile.createDimension('time',None)      # Unlimited dimension
    ncTcoord = TrgNcFile.createVariable('time','f8',('time'))
    ncTcoord.units = f"{timetypes[ttype][2]} since {timetypes[ttype][0].strftime('%Y-%m-%d %H:%M:%S')}"
    ncTcoord.standard_name = 'time'
    ncTcoord[:] = tcoords

    if npar > 0:     # Create the parset dimension and coordinate variable
        TrgNcFile.createDimension('parset', npar)
        ncPcoord = TrgNcFile.createVariable('parset','i4',('parset'))
        ncPcoord.units = '-'
        ncPcoord.standard_name = 'parset'
        ncPcoord[:] = np.arange(npar)
    if nldt > 0:    # Create the ldtime dimension and coordinate variable
        TrgNcFile.createDimension('ldtime', nldt)
        ncLcoord = TrgNcFile.createVariable('ldtime','i4',('ldtime'))
        ncLcoord.units = '-'
        ncLcoord.standard_name = 'lead_time'
        ncLcoord[:] = np.arange(nldt)
    if nmem > 0:    # Create the member dimension and coordinate variable
        TrgNcFile.createDimension('member', nmem)
        ncEcoord = TrgNcFile.createVariable('member','i4',('member'))
        ncEcoord.units = '-'
        ncEcoord.standard_name = 'enseble_member'
        ncEcoord[:] = np.arange(nmem)



    # Create spatial dimensions and coordinate variables
    # The raster variable itself is not created here, but in ExportEnkiNcVar
    if ds.RasterCount > 0:                  # Raster. Assume that the geometry will be re-used for many 
        TrgNcFile.createDimension('x', nx)  # other rasters, hence making it global to the ncfile.
        TrgNcFile.createDimension('y', ny)
        ncYcoord = TrgNcFile.createVariable('y','f8',('y'))
        ncXcoord = TrgNcFile.createVariable('x','f8',('x'))
        lonVar = TrgNcFile.createVariable('longitude', 'f8',  ('y', 'x'), zlib=True)
        latVar = TrgNcFile.createVariable('latitude', 'f8',  ('y', 'x'), zlib=True)
        nx = ds.RasterXSize                         # Expand spatial coordinate ranges 
        ny = ds.RasterYSize                         # into two numpy arrays
        GT = ds.GetGeoTransform()
        xcoords = np.arange(nx)*GT[1]+GT[0]+GT[1]/2 # Adjust a half-pixel to
        ycoords = np.arange(ny)*GT[5]+GT[3]+GT[5]/2 # get cell-center coords
        ncXcoord.units = 'm'
        ncXcoord.standard_name = 'projection_x_coordinate'
        ncXcoord[:] = xcoords
        ncYcoord.units = 'm'
        ncYcoord.standard_name = 'projection_y_coordinate'
        ncYcoord[:] = ycoords
        WLon, WLat = CreateLongLatMatrices(ds)
        lonVar.grid_mapping = mapping_name
        lonVar[:] = WLon
        latVar.grid_mapping = mapping_name
        latVar[:] = WLat
    # else:       # Vector geometry is variable-specific and not made global at ncfile level

    return TrgNcFile





# Ligner veldig på ResampleSeNorgeSeries.py::ExportNetCDF()
def ExportEnkiNcVar(ds, fname, first, deltat, ttype):
    '''
    Export a multiband GDAL Dataset to a raster time series in a NetCDF file. No return value.

    The NetCDF file is initiated if necessary, following CF conventions and Enki format with 
    dimensions time (unlimited), parset(1), ldtime(1), member(1), y(N to S), x(W to E). The
    x and y coordinates mark cell centers, and are computed from ds' GeoTransform object.
    ds: Either: A gdal.Dataset with time steps represented as RasterBands.
        Or ?? : A NetCDF4.Variable with time, y, x as dimensions (not implemented).
    fname:      string holding the file name to export to (extension .nc).
    first:      datetime.datetime for the first time step
    deltat:     datetime.timedelta for the time step length
    No return value.
    '''

    TrgNcFile = ncvars = GT = None
    nbands = 0
    name = ""

    if type(ds) != gdal.Dataset:
        print("Unsupported variable type in call to utils.ExportEnkiNcVar", flush=True)
    elif ds.RasterCount < 1 or ds.GetRasterBand(1) is None:
        print(f"There are no RasterBands ({ds.RasterCount}) in GDALDataset {ds.GetMetadataItem('name')}", flush=True)
    else:
        assert ds.GetMetadataItem('name') is not None, "Missing name"
        assert ds.GetMetadataItem('units') is not None, "Missing units"
        assert ds.GetMetadataItem('standard_name') is not None, "Missing standard_name"
        assert ds.GetMetadataItem('missing_value') is not None, "Missing missing_value"
        name = ds.GetMetadataItem('name')   # Assumes ds contains only one variable
        nbands = ds.RasterCount
        GT = ds.GetGeoTransform()       # GT[0] and GT[3] locate the corner of the NW cell
        nx = ds.RasterXSize
        ny = ds.RasterYSize
        flag = float(ds.GetMetadataItem('missing_value'))
        units = ds.GetMetadataItem('units')
        standard_name = ds.GetMetadataItem('standard_name')
        scale, offset = ds.GetRasterBand(1).GetScale(), ds.GetRasterBand(1).GetOffset()
        inputvars = [ds]
    '''
    elif type(ds) == Dataset:
        inputvars = ds
        for var in ds:
            [name, standard_name, units, grid_mapping] = GetNcMetaData(var)
            assert(var.dimensions in [('time', 'y', 'x'), ('time', 'Y', 'X')]) , "Invalid Nc dimensions"
            nband, ny, nx = var.shape
            GT = var.GetGeoTransform()       # GT[0] and GT[3] locate the corner of the NW cell
            if '_FillValue' in var.ncattrs():    flag = var._FillValue
            scale = 1 if not 'scale_factor' in var.ncattrs() else var.scale_factor
            offset = 0 if not 'add_offset' in var.ncattrs() else var.add_offset
    '''





    if not os.path.isfile(fname):       # Create a new NetCDF output file for writing
        print("Creating new NcFile", fname)
        if ttype == "excel":
            npar = nldt = nmem = 1      # Default the version dimensions' lengths to 1.
        else:
            npar = nldt = nmem = 0      # Default the version dimensions' lengths to 1.
        TrgNcFile = CreateNewEnkiNcFile(fname, first, deltat, nbands, npar, nldt, nmem, ds, ttype) # 

        ncvars = TrgNcFile.variables
        xvar = ncvars['x']
        yvar = ncvars['y']
    else:
        print(f"Appending {nbands} time steps of {name} to existing NcFile", fname)
        TrgNcFile = Dataset(fname,'a')  # Open the NetCDF output file for appending
        ncvars = TrgNcFile.variables
        # npar, nldt, nmem = ncvars['parset'].shape[0], ncvars['ldtime'].shape[0], ncvars['member'].shape[0]

        if 'X' in ncvars and 'Y' in ncvars:        
            xvar, yvar = ncvars['X'], ncvars['Y']
        else:                                       # SeNorge_2018 uses capital X and Y 
            xvar, yvar = ncvars['x'], ncvars['y']   # for spatial dimensions/coords.

        # Check match of raster dims, NW origins, cell sizes, axis orientation, center/edge.
        dx, dy = GT[1], GT[5]       # dy is usually negative. 
        assert dx == xvar[1]-xvar[0] and dy == yvar[1]-yvar[0], "dx, dy error"
        assert len(xvar) == nx and xvar[0] == GT[0] + dx/2., "Xsize error"
        assert len(yvar) == ny and yvar[0] == GT[3] + dy/2., "Ysize error"

    nctvar = ncvars['time']

    if "seconds since 1970-01-01" in nctvar.units:      # Scale the data resolution from 
        ttype = 'unix'                                  # deltat (python timedelta) to 
    elif "hours since 1900-01-01" in nctvar.units:      # dt, which is the time unit in the
        ttype = 'seNorge'                               # NetCDF file being exported to. 
    elif "days since 1899-12-30"  in nctvar.units:      # This may be days, hours or seconds, 
        ttype = 'excel'                                 # given by timetypes[ttype][1].
    dt = deltat / timetypes[ttype][1]                   # timetypes[ttype][0] is the datum.
    
    ncfirst = timetypes[ttype][0] + float(nctvar[0]) * timetypes[ttype][1]  # datetime

    # if (__name__ == "__main__"):
    print (f"First date/time in NetCDF file: {ncfirst} (in NcTunits: {nctvar[0]})")
    print (f"First date/time in new dataset: {first}")

    assert ( first >= ncfirst ) , f"New data cannot start before existing ({name}, {fname}, {first}, {ncfirst})"
    if nctvar.shape[0] > 1:       # NcFile has a time resolution - check it
        assert ( abs(dt - float(nctvar[-1] - nctvar[0]) / float(nctvar.shape[0] - 1) ) < 0.00001), "dt error"
    firstidx = int((first-ncfirst) / deltat) # The NcFile time index of 'first'
    # if (__name__ == "__main__"):
    print (f'New data starting on {first} are inserted from firstidx = {firstidx}')
    ntsteps = max(firstidx+nbands,nctvar.shape[0])  # Number of tsteps after appending
    print(f"After insertion of {nbands} bands, the netCDF file increases",
            f"from {nctvar.shape[0]} to {ntsteps} time steps")
    tcoords = np.arange(ntsteps) * dt + nctvar[0]
    last = tcoords[-1]
    newNclast = timetypes[ttype][0] + float(tcoords[-1]) * timetypes[ttype][1]
    print (f"Last date/time after insertion: {newNclast} (in NcTunits: {last})")

    nctvar[0:ntsteps] = tcoords     # Extend the time coordinates

    if ttype=='seNorge' and 'time_bnds' in ncvars:  # Averaging period delineation
        tbnds = tcoords - 24.0                      # One day less than time coord 
        ncvars['time_bnds'][0:ntsteps,0] = tbnds    # Start of tstep (Yesterday 06)
        ncvars['time_bnds'][0:ntsteps,1] = tcoords  # End of tstep (today 06:00utc)

    mapping_name = [nv for nv in ncvars if "grid_mapping_name" in ncvars[nv].__dict__][0] #ncattrs()][0]

    if name in ncvars:
        dsVar = ncvars[name]
    else:                           # The variable is new and needs to be created
        if 'X' in ncvars and 'Y' in ncvars:     spatdimnames = ('Y', 'X')
        else:                                   spatdimnames = ('y', 'x')
        if 'parset' in ncvars:                  verdimnames = ('parset', 'ldtime', 'member')
        else:                                   verdimnames = ()
        dsVar = TrgNcFile.createVariable(name, 'f4',  ('time',) + verdimnames + spatdimnames, 
                                         zlib=True, least_significant_digit=1, fill_value=flag)
        #     dsVar = TrgNcFile.createVariable(name, 'f4',  ('time', 'parset', 'ldtime', 'member', 'Y', 'X'), 
        #                                     zlib=True,fill_value=flag)
        #     dsVar = TrgNcFile.createVariable(name, 'f4',  ('time', 'parset', 'ldtime', 'member', 'y', 'x'), 
        #                                    zlib=True,fill_value=flag)
        dsVar.units = units
        if scale != None:   dsVar.scale_factor = scale
        if offset != None:  dsVar.add_offset = offset
        dsVar.standard_name = standard_name
        dsVar.grid_mapping = mapping_name 
        dsVar.GeoTransform = format('%i %i %i %i %i %i ') % \
                                (GT[0], GT[1], GT[2], GT[3], GT[4], GT[5])


    
    # http://gis.stackexchange.com/questions/70458/convert-timeseries-stack-of-gtiff-raster-to-single-netcdf
    # Copy each input RasterBand to the NetCDF object via a numpy array.
    if type(ds) == gdal.Dataset:
        for i in range (0,nbands):
            data_map = ds.GetRasterBand(i+1).ReadAsArray()
            if 'parset' in ncvars:
                dsVar[firstidx+i,0,0,0,:,:] = data_map
            else:
                dsVar[firstidx+i,:,:] = data_map
    else:
        print("Noe gikk galt i ExportEnkiNcVar", type(ds), flush=True)
    
    TrgNcFile.close()



def ExportEnkiNcTser(ogrds, df, ncfname, ttype):
    '''
    Export a OGR dataset with a time-indexed Pandas data frame to a netCDF file. No return value.

    ExportEnkiNcVar exports a time-indexed Pandas DataFrame to a network time
    series in a NetCDF file. Unlike ExportEnkiNcVar(), the input time series (df)
    is separate from the geodataset (ogrds). The NetCDF file is initiated if
    necessary, following CF conventions and Enki format.
    ogrds:      An OGR point data set providing variable name and point order
    df:         Time-indexed Pandas DataFrame with point IDs as column headers
    ncfname:    Target netCDF file to export to (extension .nc).
    ttype:      Time type (unit/datum); excel(d/1899), unix(s/1970), seNorge(h/1900)

    If the target netCDF file (ncfname) already exists and contains the ogrds variable, 
    we check that the netCDF IDs are ordered as in ogrds. If required, we rearrange 
    the df so its IDs appear in the same order as in ogrds (and ncfname). 

    If ncfname does not exist or does not contain the ogrds variable, we add it.
    If the resulting netCDF database is to be used in Enki, ogrds must be correctly
    represented in the tpx file required by Enki's Python interface. Enki cannot
    create GIS variables in the region from info in a NetCDF file.
    '''

    assert ogrds.GetMetadataItem('standard_name') is not None, "Missing standard_name"
    assert ogrds.GetMetadataItem('missing_value') is not None, "Missing missing_value"
    name = ogrds.GetMetadataItem('name')

    first = df.index[0]                 # First datetime in the Pandas DataFrame
    deltat = df.index[1] - df.index[0]  # Time step length in the Pandas DataFrame
    nbands = df.shape[0]                # Number of time steps in the Pandas DataFrame

                                        # Number of columns in the Pandas DataFrame
                                        # must match the number of points in ogrds,
                                        # and dimension length in 

    # Check that gdalIDs matches pdIDs in counts and ordered values
    layer = ogrds.GetLayer()
    ogrIDs = [int(feat.GetField('ID')) for feat in layer]
    dfcols = list(df.columns.values) 
    assert len(dfcols) == len(ogrIDs)
    for i, oID in enumerate(ogrIDs):
        if oID != dfcols[i]:
            print (i, oID, dfcols[i])
            print (ogrIDs)
            print (dfcols)
            assert oID == dfcols[i]     # Just to create a stop


    TrgNcFile = ncvars = nctvar = None

    if not os.path.isfile(ncfname):     # Create a new NetCDF output file for writing
        print("Creating new NcFile", ncfname)
        if ttype == "excel":            # This is typical for Enki's databases.
            npar = nldt = nmem = 1      # Default the version dimensions' lengths to 1.
        else:
            npar = nldt = nmem = 0      # Default the version dimensions' lengths to 1.

        TrgNcFile = CreateNewEnkiNcFile(ncfname, first, deltat, nbands, npar, nldt, nmem, ogrds, ttype) # 
        ncvars = TrgNcFile.variables
        nctvar = ncvars['time']
    else:
        print(f"ncfile {ncfname} already exists, appending")
        TrgNcFile = Dataset(ncfname,'a')# Open the NetCDF output file for appending
        ncvars = TrgNcFile.variables
        npar, nldt, nmem = ncvars['parset'].shape[0], ncvars['ldtime'].shape[0], ncvars['member'].shape[0]

        if name in ncvars:                      # If the NcFile already contains this 
            ncIDs = ncvars[f"{name}ID"][:]      # variable, verify that the IDs in the
            for i, nID in enumerate(ncIDs):     # NcFile and df are the same, and in 
                if nID != dfcols[i]:            # the same order. For variables not 
                    print (i, nID, dfcols[i])   # already in the NcFile, this check is 
                    print (dfcols)              # neither possible nor needed.
                    print (ncIDs)
                    assert nID == dfcols[i]     # Just to create a stop

        nctvar = ncvars['time']

        if "seconds since 1970-01-01" in nctvar.units:    
            ttype = 'unix'
        elif "hours since 1900-01-01" in nctvar.units:    
            ttype = 'seNorge'
        elif "days since 1899-12-30"  in nctvar.units:    
            ttype = 'excel'

    # nbands = nctvar.shape[0]

    dt = deltat / timetypes[ttype][1] # Scaling from timedelta to NetCDF time

    ncfirst = timetypes[ttype][0] + float(nctvar[0]) * timetypes[ttype][1]
    if (__name__ == "__main__"):
        print ("First day/time in existing NetCDF file: ", ncfirst)
        print ("First day/time in new data set: ", first)

    assert ( first >= ncfirst ) , f"New data cannot start before existing ({name}, {ncfname})"
    # TODO: Could alternatively skip the start of the df if it start too early
    # TODO: Should probably also report error if there is a gap between NcPeriod and df_start.
    if nctvar.shape[0] > 1:       # NcFile has a time resolution - check it
        assert ( abs(dt - float(nctvar[-1] - nctvar[0]) / float(nctvar.shape[0] - 1) ) < 0.00001), "dt error"
    firstidx = int((first-ncfirst).total_seconds() / deltat.total_seconds()) # The NcFile time index of 'first'
    if (__name__ == "__main__"):
        print (f'New data starting on {first} are inserted from firstidx = {firstidx}')
    ntsteps = max(firstidx+nbands,nctvar.shape[0])  # The period of df may fully, partially or not at all 
    tcoords = np.arange(ntsteps) * dt + nctvar[0]   # cover the period already in the existing NcFile, but
    last = tcoords[-1]                              # cannot start prior to existing NcFile start.

    if (__name__ == "__main__"):
        print ("Last day/time in new data set: ", last)

    nctvar[0:ntsteps] = tcoords     # Extend the time coordinates

    mapping_name = TrgNcFile.grid_mapping

    if name in ncvars:
        dsVar = ncvars[name]
    else:                           # The variable is new and need to be created
        dsVar = CreatePtnetworkVar(TrgNcFile, ogrds)

    # Check that ncIDs matches pdIDs throughout

    # http://gis.stackexchange.com/questions/70458/convert-timeseries-stack-of-gtiff-raster-to-single-netcdf
    # Copy each input RasterBand to the NetCDF object via a numpy array.

    ll = df.values.tolist()             # Creates a list of lists
    assert len(ll)==nbands, f"Length mismatch: len(ll) == {len(ll)}, nbands == {nbands}"
    for i in range (0,nbands):
        dsVar[firstidx+i,0,0,0,:] = ll[i]
    
    TrgNcFile.close()



def GetNcInfo(fname):
    # if not os.path.isfile(fname):
    #     return None
    # print("")
    # print(f"starting on GetNcInfo() with file name {fname}")
    # print("")
    try:
        NcDS = Dataset(fname)
    except:
        return None
    ncv = NcDS.variables
    gridmapping = wkt = sdim0 = sdim1 = None
    res = dict()
    gridmapping = [ncv[v] for v in ncv if 'grid_mapping_name' in ncv[v].__dict__]
    # print("grid_mapping list = ", gridmapping)
    if 'X' in ncv and 'Y' in ncv:        
        xvar, yvar = ncv['X'], ncv['Y']
    else:                                       # SeNorge_2018 uses capital X and Y 
        xvar, yvar = ncv['x'], ncv['y']         # for spatial dimensions/coords.
    lonvar = [ncv[v] for v in ncv if v in ['lon','longitude']]
    latvar = [ncv[v] for v in ncv if v in ['lat','latitude']]
    tvar = ncv['time']
    if 'member' in ncv:
        mvar = ncv['member']
        res['nm'] = mvar.shape[0]
    spatialvars = [ncv[v] for v in ncv if 'x' in ncv[v].dimensions or 'X' in ncv[v].dimensions]
    nonspatialvars = [ncv[v] for v in ncv if ncv[v] not in spatialvars]
    if len(lonvar) > 0:
        res['longitude'] = lonvar[0].name
    if len(latvar) > 0:
        res['latitude'] = latvar[0].name
    if len(gridmapping) > 0:
        res['grid_mapping'] = gridmapping[0].name
        print(f"Selected grid_mapping = {res['grid_mapping']}")
        if 'crs_wkt' in gridmapping[0].__dict__:    #ncattrs():
            res['wkt'] = gridmapping[0].crs_wkt
            print(f"grid_mapping.wkt = {res['wkt']}")
        if 'spatial_ref' in gridmapping[0].__dict__: #ncattrs():
            res['spatial_ref'] = gridmapping[0].spatial_ref
            print(f"grid_mapping.spatial_ref = {res['spatial_ref']}")
        if 'proj4' in gridmapping[0].__dict__: #ncattrs():
            res['proj4'] = gridmapping[0].proj4
            print(f"grid_mapping.proj4 = {res['proj4']}")
        
        if gridmapping[0].name == 'transverse_mercator' and gridmapping[0].false_easting == 500000.0:
            res['UTM zone'] = int(30 + (gridmapping[0].longitude_of_central_meridian+3)/6)
    res['nt'] = tvar.shape[0]
    res['xvar'] = xvar.name
    res['nx']   = xvar.shape[0]
    res['xmin'] = min(xvar[0], xvar[-1])      # These are cell center coordinates
    res['xmax'] = max(xvar[0], xvar[-1])      # from the NcFile dimension coords
    dx = (res['xmax']-res['xmin'])/(res['nx']-1)
    res['yvar'] = yvar.name
    res['ny']   = yvar.shape[0]
    res['ymin'] = min(yvar[0], yvar[-1])
    res['ymax'] = max(yvar[0], yvar[-1])
    dy = (res['ymin']-res['ymax'])/(res['ny']-1)    # Ensuring n->S convention in GT,
    res['mapvars'] = [v.name for v in spatialvars]      # giving negative dy (north positive)
    res['nomapvarsmapfields'] = [v.name for v in nonspatialvars]
    res['GeoTransform'] = [res['xmin'], dx, 0, res['ymax'],0, dy]   # Edge coords
    
    NcDS.close()
    return res




def find_minimal_window(srcDS,trgDS):
    """ Return UL indices and size of a window from srcDS which cover trgDS

        Find the smallest window of the source data set which will  
        cover the target data set (which has another coordinate system).
        Return [firstC, firstR, nC, nR] """

    assert srcDS is not None, "utils::find_minimal_window received an empty GDALDataset"
    srcGT = srcDS.GetGeoTransform()
    # print 'srcGT = ', srcGT
    # Check out a reverse warp to find a source window, thus supplying the
    # target data set trgDS as "source", and the source coordsystem as "target"
    # projection info.
    tmpDS = gdal.AutoCreateWarpedVRT(trgDS, None, srcDS.GetProjectionRef(), 
                                     gdal.GRA_NearestNeighbour, 0 )
    tmpGT = tmpDS.GetGeoTransform() # [ULX,dxdc,dxdr=0,ULN,dydc=0,dydr]
    # print 'tmpGT = ', tmpGT
    
    # Calculate the four region borders, in source coordinate system
    trgXmin = tmpGT[0]
    trgXmax = tmpGT[0] + tmpDS.RasterXSize*tmpGT[1] + tmpDS.RasterYSize*tmpGT[2]
    trgYmax = tmpGT[3]
    trgYmin = tmpGT[3] + tmpDS.RasterXSize*tmpGT[4] + tmpDS.RasterYSize*tmpGT[5]
    # print Xmin, Xmax, Ymin, Ymax
    
    # ...and the pixel size...
    srcDX = srcGT[1]
    srcDY = - srcGT[5]
    
    # Find the source window, in pixel index terms, that will cover the
    # target Region of interest when reprojected. (plus 2 at each side).
    # Cmin and Rmin are from the Western and Northern map edge, respectively

    # Region West border left-right index in src:
    Cmin    = int ( max ( 0, ( trgXmin - srcGT[0] )  / srcDX - 2 ) ) 
    # Number of columns in src-window covering trg:
    trg_nC  = int ( min ( srcDS.RasterXSize, (trgXmax - trgXmin) / srcDX + 4 ) )

    # Region North border top-down index in src
    Rmin    = int ( max ( 0, ( srcGT[3] - trgYmax )  / srcDY - 2 ) )    
    # Number of rows in src-window covering trg
    trg_nR  = int ( min ( srcDS.RasterYSize, (trgYmax - trgYmin) / srcDY + 4 ) )
    
    W, E, S, N   =   Cmin, Cmin + trg_nC, srcDS.RasterYSize - Rmin - trg_nR, srcDS.RasterYSize - Rmin, 

    # return [Cmin, Rmin, trg_nC, trg_nR]      # FirstCol, FirstRow, Ncols, Nrows
    return W, E, S, N
 




'''
    Input:          WndTserDS: A memory GDALDataset with nbands spanning 
                    a time dimension or a time-member combined dimension.
                    WndTserDS is in the supplier's projection, and assumed
                    to cover the area of RstTmplFname.
    Return value:   RegHourlyTserDS: A memory GDALDataset with hourly bands,
                    resampled to the geometry/coordsystem of RstTmplFname.
    '''
def Resample(WndTserDS, RstTmplfname):

    nbands      = WndTserDS.RasterCount
    varname     = WndTserDS.GetMetadataItem('name')
    stdname     = WndTserDS.GetMetadataItem('standard_name')
    unit        = WndTserDS.GetMetadataItem('units')
    flagcode    = float(WndTserDS.GetMetadataItem('missing_value'))

    # RstTmplFname only supplies its geometry/coordsystem to a new memory data  
    # set RegTrgDS, the values in RstTmplFname are neither used nor modified. 
    RegTrgDS    = MemDSfromGISfile(RstTmplfname,nbands)
    SetGDALMetadata(RegTrgDS, varname, stdname, unit, flagcode)

    RegRows     = RegTrgDS.RasterYSize
    RegCols     = RegTrgDS.RasterXSize
    RegNodataTable = np.ndarray(shape=(RegRows,RegCols))
    RegNodataTable.fill(flagcode)  

    for tidx in range(0,nbands):                    # Pre-fill RegTrgDS with 
        Rband = RegTrgDS.GetRasterBand(tidx+1)      # missing-value tables
        Rband.SetNoDataValue(flagcode)              # for all bands.
        Rband.WriteArray(RegNodataTable)
    
    # Now reproject and clip to RegTrgDS's extent and coordinate system.
    result = gdal.ReprojectImage(WndTserDS,RegTrgDS,None, None, gdal.GRA_Bilinear)
    if result != 0: 
        print (result, gdal.GetLastErrorMsg())

    return RegTrgDS
    






if __name__ == "__main__":
    # Perform a series of tests
    fname = 'G:\\Robot\\Regions\\seNorge2\\TEMP1h\\seNorge_v2_0_TEMP1h_grid_2021032713.nc'
    res = GetNcInfo(fname)
    print (res)

    utm33n_zonewkt = get_utm_wkt(33)
    print(utm33n_zonewkt)

    utm33n_projwkt = get_projstring_wkt("+proj=utm +zone=33 +datum=WGS84 +units=m +no_defs +ellps=WGS84 +towgs84=0,0,0")
    print(utm33n_projwkt)
    print()

    fname = "G:\\Robot\\Regions\\TEK\\Elevation.sdat"
    rststack = MemDSfromGISfile(fname, 4)
    varname = rststack.GetMetadataItem('name')
    print(f"Imported {varname} from {fname}")
    SetGDALMetadata(rststack, "testrst", "testing_stdname", "nauticalmiles", -99) # No return, called from Resample, which is not called internally in utils
    x, y = GetXYcoords(rststack)
    print (f"nx={len(x)} ({x[0]} to {x[-1]}), ny={len(y)} ({y[0]} to {y[-1]})")
    lon, lat = CreateLongLatMatrices(rststack)
    fname = "G:\\Robot\\Regions\\testrasternc.nc"
    if os.path.isfile(fname): os.remove(fname)
    ExportEnkiNcVar(rststack, fname, datetime(2021,4,1), timedelta(hours=1), "excel")   # No return, not called internally in utils
    print (f"Check output at {fname}")
    print()
    rst = None

    # See http://pcjericks.github.io/py-gdalogr-cookbook/index.html for ideas
    fname = "G:\\Robot\\Regions\\TEK\\TEKPstats.shp"
    points = gdal.OpenEx(fname, gdal.OF_VECTOR)
    SetGDALMetadata(points, "TEKPstats", "testing_pstats_stdname", "mm", -99) # No return, called from Resample, which is not called internally in utils
    varname = points.GetMetadataItem('name')
    print(f"Imported {varname} from {fname}")
    x, y = GetXYcoords(points)
    print (f"npoints={len(x)}, first at {x[0]},{y[0]}; last at {x[-1]},{y[-1]}")
    print (f"The 'points' data set has reference system {points.GetLayer().GetSpatialRef().ExportToWkt()}")
    lon, lat = CreateLongLatMatrices(points)
    print (f"npoints={len(lon)}, first at {lon[0]},{lat[0]}; last at {lon[-1]},{lat[-1]}")
    layer = points.GetLayer()
    IDs = [int(feat.GetField('ID')) for feat in layer]
    print (f"npoints={len(IDs)}, first ID is {IDs[0]}, last is {IDs[-1]}")
    ttags = [datetime(2021,4,1) + timedelta(hours=i) for i in range(len(IDs))]
    arr = np.add.outer([i*100 for i in IDs],[h.hour for h in ttags]).T
    df = pd.DataFrame(arr, columns=IDs, index=ttags)
    print(df)
    fname = "G:\\Robot\\Regions\\testpointnc.nc"
    if os.path.isfile(fname): os.remove(fname)      # Ensures that CreateNewEnkiNcFile is called
    ExportEnkiNcTser(points, df, fname, "excel")  
    print (f"Check output at {fname}")
    points = None
    ncf = Dataset(fname)
    ncvars = ncf.variables
    print(f"[name, standard_name, units, grid_mapping] for {varname} are: {GetNcMetaData(ncvars[varname])}")
    ncf.close()




    # NOt tested: SetNcMetaData(ncVar, stdname, units, grd_mapping_vname) # No return, not called internally in utils




# See http://pcjericks.github.io/py-gdalogr-cookbook/index.html for ideas
# layer = ds.GetLayer()
# IDs = [int(feat.GetField('ID')) for feat in layer]
# xcoords = [feat.GetGeometryRef().GetPoint()[0] for feat in layer]
# ycoords = [feat.GetGeometryRef().GetPoint()[1] for feat in layer]

# featureCount = layer.GetFeatureCount()  # Sjekke wbkFlatten(poGeometry->getGeometryType()) == wbkPOint?
# for feat in layer:
#     geom = feat.GetGeometryRef()
#     print (type(geom), len(geom))
# print(layer)
#    <osgeo.ogr.Layer; proxy of <Swig Object of type 'OGRLayerShadow *' at 0x000001BEA6573A20> >
# print(layer.GetName())
#    TEKpstats
# print(layer.GetGeomType())     
#    1
# layer.GetLayerDefn().GetFieldIndex("ID")
#    0
# layer.GetLayerDefn().GetFieldDefn(0).name  
#   'id'
# layer.GetLayerDefn().GetFieldDefn(1).name 
#   'TEKPstats'
# layer.GetLayerDefn().GetFieldDefn(2).name 
# for feature in layer:
#     print(feature.GetField("ID"))
#   3.0
#   4.0
#   2.0
#   9.0
#   8.0
#   10.0
#   14.0
#   5.0
#   21.0
# layer[0].GetField("ID")          
#   3.0
# >>> for feat in layer: print(feat.GetGeometryRef())
# POINT (580045.9747 6939483.997)
# POINT (560196.6062 6930430.306)
# POINT (565668.0367 6940009.994)
# POINT (535590.2715 6966511.874)
# POINT (551892.9607 6967393.118)
# POINT (541327.7842 7001714.749)
# POINT (504935.1319 6951561.525)
# POINT (528216.7657 6945538.056)
# POINT (517428.8511 7011327.84)
