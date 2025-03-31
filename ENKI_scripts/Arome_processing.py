# -*- coding: utf-8 -*-
"""
Created on Tue Apr 10 08:52:57 2018

@author: Sjur Kolberg, Enki hydrologi



Data sets are  'AromePP', 'AromeMEPS'

Call chain:
if __name__ == "__main__"
    # The following two functions are called with each of the datasets above.
    GetNewData(dataset, regionfld): called only from if __name__ == "__main__"
        check_new(urlsite, datafld):
        download_thredds(urlfnamelist, datafld):
    ResampleExport(dataset, regionfld, ENKIbase = ""): called only from if __name__ == "__main__"
        CreateTemporalGDALstacks(datafld, obsfilelist):
        utils.Resample(WndTserDS, RstTmplfname):
        GetFileList(datafld, fromtime, totime):

TODO: 

"""


import os
import requests
import json
import traceback
from netCDF4 import Dataset
from osgeo import gdal, osr, ogr
from datetime import datetime, timedelta
import numpy as np
if __name__ == "__main__":
    from utils import ExportEnkiNcVar, SetGDALMetadata, GetXYcoords, GetNcInfo, MemDSfromInfo, SetNcMetaData, Resample, find_minimal_window
else:
    from Tasks.utils import ExportEnkiNcVar, SetGDALMetadata, GetXYcoords, GetNcInfo, MemDSfromInfo, GetNcMetaData, Resample, find_minimal_window



def AromeQC(vname, time, val):
    ''' Check quality of values nparray(val) for variable str(vname) at datetime(time)
        Return "" if passed, one-line error description of not.
    '''
    # For now, a super simple check on value sanity
    if not np.any(np.isfinite(val)):
        return f"No values for {vname} are finite on {time}\n"
    if np.all(val < -50):
        return f"All values for {vname} is less than -50 on {time}\n"
    if np.all(val > 2000):
        return f"All values for {vname} is greater than 2000 on {time}\n"
    if not np.all(np.isfinite(val)):
        return f"Some values for {vname} are non-finite on {time}\n"
    if np.any(val < -50):
        return f"Some values for {vname} is less than -50 on {time}\n"
    if np.any(val > 2000):
        return f"Some values for {vname} is greater than 2000 on {time}\n"
    return ""


# Determine if there are new forcasts available at the download site,
# checking for files less than one day old and not already downloaded.
# Return a list of file names to download.
# For multiple regions, call separately with different datafld.
# Data type is inferred from the supplied URL
def check_new(urlsite, datafld, start=None, end=None):
    """ Check urlsite for new files to download to datafld, return list of fnames"""
    if not os.path.isdir(datafld):
        return 'Arome.check_new() could not find folder ' + datafld 
    if not datafld.endswith(os.sep):
        datafld += os.sep

    # For routine downloads of fresh data, just leave start and stop as None;
    # check_new will fetch any file not more than nhours old existing at the
    # URL and not already in regionfld. Forecasts older than 1 day do not
    # exist on the URL server. For forecasts, time is the reference time, 
    # not the valid_time which could be several days into the future.
    nhours = 72
    if end==None:                   # No end: Proceed until present time.
        end = datetime.utcnow()
    if start==None:                 # No start: Aim at 3 days before end.
        start = end - timedelta(hours=nhours)
    nhours = int ((end-start).total_seconds() / 3600)
    if nhours < 1:
        return f"Invalid time period in Arome:check_new()."

    var = ""

    if "metpplatest/" in urlsite:  # Site for 1-hourly postprocessed forecasts
        fprefix  = f"met_forecast_1_0km_nordic_" 
                    #datetime.today().strftime("met_forecast_1_0km_nordic_") #
        ttags = ['T{:02d}Z.nc'.format(hr) for hr in list(range(0,24,1))]
    elif "mepslatest/" in urlsite:  # Site for 3-hourly ensemble forecasts
        fprefix  = f"meps_lagged_6_h_subset_2_5km_"
                    #datetime.today().strftime("meps_lagged_6_h_subset_2_5km_") #
        ttags = ['T{:02d}Z.ncml'.format(hr) for hr in list(range(0,24,3))]
        # Include forecasts up to 24 hours old (max 8 or 24) unless already downloaded

    s = requests.session()

    try:
        flist = []
        dtag  = datetime.today().strftime("%Y%m%d")                      # Today
        ydtag = (datetime.today()+timedelta(days=-1)).strftime("%Y%m%d") # Yesterday
        for ttag in ttags:
            urlfname = fprefix + dtag + ttag                        # ttag includes 
            localfname = urlfname.replace('.ncml','.nc')            # '.nc' or '.ncml'
            print("check_new is looking for",urlfname,end=':', flush=True)
            if requests.head(urlsite + urlfname + '.html').status_code == 200:
                print(" Found",end=',', flush=True)
            else:
                urlfname = fprefix + ydtag + ttag                        # Yesterday
                print("  Not found; trying",urlfname,end=':', flush=True)
                if requests.head(urlsite + urlfname + '.html').status_code == 200:
                    print(" Found", end=',', flush=True)
                    localfname = urlfname.replace('.ncml','.nc')            # '.nc' or '.ncml'
                else:
                    print(" Not found.", flush=True)
                    continue

            if os.path.isfile(datafld + localfname):                      # Already downloaded
                print(localfname, "exists already.", flush=True)
            else:
                flist.append(urlsite + urlfname)
                print('added.', flush=True)

        s.cookies.clear() 
        return flist

    except Exception as e:
        msg = format("%s: %s") % (type(e),str(e))
        traceback.print_exc()
        return 'Arome.check_new error: ' + msg


def download_thredds(urlfnamelist, datafld):
    """ Downloads the listed files to datafld, returning local-name list of downloaded files"""
    # Download the data files listed in urlfnamelist and store in datafld.
    # Name datafld's nearest parent as regionfld, and look here for a file
    # 'RegionConfig.json' holding necessary metadata for region clipping.
    # Return list of downloaded files with their local path to datafld
    # During production the list is usually short, but can increase after interruptions

    if not datafld.endswith(os.sep):
        datafld += os.sep                               # include the last sep
    regionfld = datafld[0:datafld[0:-1].rfind(os.sep)+1]      

    with open(regionfld+'RegionConfig.json') as f:
        Cfg = json.load(f)
    RegionElevDS = gdal.Open(regionfld + Cfg["elev_GIS_dataset"])
    s = requests.session()

    DSfnames = []

    for urlfname in urlfnamelist:
        # The urlfname likely use '/' as folder separator, but we check both.
        fttl = urlfname[max(urlfname.rfind('\\'),urlfname.rfind('/'))+1:]
        outfname = datafld + fttl.replace('.ncml','.nc')

        # if __name__ == "__main__":
        print(f"\ndownload_thredds starts working on file {fttl} exporting to {outfname}")
        urlfileinfo = GetNcInfo(urlfname)
        # print(f"Calling GetNcInfo with argument {urlfname} yielded: {urlfileinfo}")

        urlDS = Dataset(urlfname, 'r')      # Open the remote NetCDF Dataset
        ncv = urlDS.variables
        lonvar, latvar = "longitude", "latitude"
        inNcX, inNcY = ncv['x'], ncv['y']
        inNcTair = ncv['air_temperature_2m']
        inNcRefT  = ncv['forecast_reference_time']
        DTref = datetime.fromtimestamp(int(inNcRefT.getValue()))
        utcDTref = datetime.utcfromtimestamp(int(inNcRefT.getValue()))
        inNcProjvar = inNcTair.grid_mapping # Just a random value, but with important attributes.
        inSCS  = ncv[inNcProjvar]
        if __name__ == "__main__":
            print("Remote data set proj string = ", inSCS.proj4)
            print(f"DTref forecast refererence time in postprocessed file: {DTref} {utcDTref} ({inNcRefT.getValue()})")
            # print(urlfileinfo)
        
        xylim = [urlfileinfo['xmin'], urlfileinfo['xmax'],urlfileinfo['ymax'], urlfileinfo['ymin']]
        tmpSrcDS = MemDSfromInfo(urlfileinfo['ny'], urlfileinfo['nx'], 1, xylim, "", inSCS.proj4)    # GDALDataset 
        windidx = find_minimal_window(tmpSrcDS,RegionElevDS)  # windidx = Cfg["arome_ens_wndidx"].copy()

        inNcPvar = None
        if "met_forecast_1_0km_nordic_" in urlfname:
            inNcPvar = ncv['precipitation_amount']
            forecast_type = "Arome_pp"
            emin, emax = 0, 1   
        elif "meps_lagged_6_h_subset_2_5km_" in urlfname:
            inNcPvar = ncv['precipitation_amount_acc']
            forecast_type = "Arome_ens"
            emin, emax = 0, 3  # inNcTair.shape[1] # The MEPS ensemble has 30 members


        inTc = ncv['time'][:]                           # Time coordinate
        tmin, tmax = 0, len(inTc)
        nbands = (tmax-tmin)*(emax-emin)
        assert(nbands > 0)
        inEc = None
        if forecast_type == "Arome_ens":
            inEc = ncv['ensemble_member'][emin:emax]    # Ensemble coordinate


        NcNtoS = inNcY[0] > inNcY[-1]   # North-to-south line order. True for SeNorge, False for Arome.
        if __name__ == "__main__":
            print (f"Flipping {forecast_type}? NcNtoS is {NcNtoS}, Cfg_windidx = {windidx}")
        if NcNtoS:                                      # The W-E-S-N window definition from Cfg must
            tmp = inNcY.shape[0] - windidx[3]           # be flipped to a W-E-N-S window to frame the
            windidx[3] = inNcY.shape[0] - windidx[2]    # desired subregion from the source map. Still,
            windidx[2] = tmp                            # windidx[3] needs to be larger than windidx[2].
            assert(windidx[3] > windidx[2])
        wndRows = windidx[3] - windidx[2]
        wndCols = windidx[1] - windidx[0]
        inYc = inNcY[windidx[2]:windidx[3]]             # Y coordinates in window, in NcFile order
        inXc = inNcX[windidx[0]:windidx[1]]             # X coordinates in window, in NcFile order
        #inLat = ncv[latvar][windidx[2]:windidx[3],windidx[0]:windidx[1]]  # Latitude field
        #inLon = ncv[lonvar][windidx[2]:windidx[3],windidx[0]:windidx[1]]  # Longitude field
        if NcNtoS:
            if __name__ == "__main__":
                print (f"Window has {wndRows} rows and {wndCols} cols, indexed ",windidx)
        else:
            inYc = np.flip(inYc)
            if __name__ == "__main__":
                print (f"Flipped window has {wndRows} rows and {wndCols} cols, indexed ",windidx)
        assert(inXc[0] < inXc[-1] and wndCols > 0)      # X coordinates in window, ascending eastward
        assert(inYc[0] > inYc[-1] and wndRows > 0)      # Y coordinates in window, descending southward

        inNcElev = None
        if 'altitude' in ncv:
            if __name__ == "__main__":
                print ("Downloading and clipping altitude", flush=True)
            inNcElev = ncv['altitude']
            E_Meta = inNcElev.__dict__.copy()
            E_Meta['name'] = "altitude"
            if '_FillValue' not in E_Meta:
                E_Meta['_FillValue'] = -999.99
            inElev = inNcElev[windidx[2]:windidx[3],windidx[0]:windidx[1]]
            if not NcNtoS:  inElev = np.flipud(inElev)

        # The input variables defined above are masked_arrays, not NetCDF variables.

        cube = (slice(tmin,tmax),0,slice(emin,emax),
                slice(windidx[2],windidx[3]),slice(windidx[0],windidx[1]))
        if forecast_type == "Arome_pp":
            cube = (slice(tmin,tmax),slice(windidx[2],windidx[3]),slice(windidx[0],windidx[1]))

        # Read meteorological fields from remote source, clipped to window, derive if necessary
        inTair = inRH = inSWrad = inWS = inPrec = None  # Masked arrays to hold input window met fields

        if inNcTair is not None:
            if __name__ == "__main__":
                print ("Downloading and clipping air_temperature_2m", flush=True)
            T_Meta = inNcTair.__dict__.copy()
            T_Meta['name'] = "air_temperature"
            if '_FillValue' not in T_Meta:
                T_Meta['_FillValue'] = -999.99
            inTair = inNcTair[cube]
            if inNcTair.units not in ["DegC", "Celsius"]:
                inTair[:] -= 273.15
                T_Meta['units'] = "DegC"
            if not NcNtoS:  inTair = np.flip(inTair,-2)

        if 'relative_humidity_2m' in ncv:
            if __name__ == "__main__":
                print ("Downloading and clipping relative_humidity_2m", flush=True)
            inNcRH = ncv['relative_humidity_2m']
            RH_Meta = inNcRH.__dict__.copy()
            RH_Meta['name'] = "relative_humidity"
            if '_FillValue' not in RH_Meta:
                RH_Meta['_FillValue'] = -999.99
            inRH   = inNcRH[cube]
            inRH[:] *= 100
            if not NcNtoS:  inRH = np.flip(inRH,-2)

        if 'integral_of_surface_downwelling_shortwave_flux_in_air_wrt_time' in ncv: 
            if __name__ == "__main__":
                print ("Downloading and clipping integral_of_shortwave_flux", flush=True)
            inNcSWR = ncv['integral_of_surface_downwelling_shortwave_flux_in_air_wrt_time']
            SWR_Meta = inNcSWR.__dict__.copy()
            SWR_Meta['name'] = "shortwave_flux"
            if '_FillValue' not in SWR_Meta:
                SWR_Meta['_FillValue'] = -999.99
            inSWradAcc = inNcSWR[cube]
            # Derive SWrad by temporal differentiation
            inSWrad = np.insert(np.diff(inSWradAcc.filled(),axis=0),0,0,axis=0) / 3600.0
            if not NcNtoS:  inSWrad = np.flip(inSWrad,-2)

        if 'wind_speed_10m' in ncv:     # Typical for the postprocessed forecast
            if __name__ == "__main__":
                print ("Downloading and clipping wind_speed_10m", flush=True)
            inNcWS = ncv['wind_speed_10m'] 
            inWS  = inNcWS[cube]
            WS_Meta = inNcWS.__dict__.copy()
            WS_Meta['name'] = "wind_speed_10m"
            if '_FillValue' not in WS_Meta:
                WS_Meta['_FillValue'] = -999.99
            if not NcNtoS:  inWS = np.flip(inWS,-2)
        elif 'x_wind_10m' in ncv:       # The standard for the MEPS ensemble. 
            if __name__ == "__main__":
                print ("Downloading and clipping x_wind_10m and y_wind_10m", flush=True)
            inWsX  = ncv['x_wind_10m'][cube]    # Derive speed from components.
            inWsY = ncv['y_wind_10m'][cube]
            inWS = np.sqrt(np.power(inWsX[:],2) + np.power(inWsY[:],2))
            WS_Meta = ncv['x_wind_10m'].__dict__.copy()
            WS_Meta['name'] = "wind_speed_10m"
            if '_FillValue' not in WS_Meta:
                WS_Meta['_FillValue'] = -999.99
            # NB: Adjust the metadata from x-component to speed!
            if not NcNtoS:  inWS = np.flip(inWS,-2)

        if 'precipitation_amount' in ncv:   # Typical for postprocessed Arome
            if __name__ == "__main__":
                print ("Downloading and clipping precipitation_amount", flush=True)
            inPrec = inNcPvar[cube]
        elif 'precipitation_amount_acc' in ncv:  # Standard for the MEPS ensemble.
            if __name__ == "__main__":
                print ("Downloading and clipping precipitation_amount_acc", flush=True)
            inPrecAcc = inNcPvar[cube]
            # Derive hourly precip by temporal differentiation, first map being all zeros.
            inPrec = np.insert(np.diff(inPrecAcc.filled(),axis=0),0,0,axis=0)

        if inNcPvar is not None:
            P_Meta = inNcPvar.__dict__.copy()
            P_Meta['name'] = "precipitation_amount"
            if '_FillValue' not in P_Meta:
                P_Meta['_FillValue'] = -999.99
            if not NcNtoS:  inPrec = np.flip(inPrec,-2)

        invars = [inTair, inRH, inSWrad, inWS, inPrec]                  # masked_arrays
        metadata = [T_Meta, RH_Meta, SWR_Meta, WS_Meta, P_Meta]


        # Now the data are clipped to cover the target region and stored in numpy arrays. 
        # To create and fill output NetCDF files from these we need a source GDAL MEM data 
        # set with WKT and GT as basis 

        xylim = [inXc[0], inXc[wndCols-1], inYc[0], inYc[wndRows-1]]        # Window coords
        SrcMemDS = MemDSfromInfo(wndRows, wndCols, nbands, xylim, "", inSCS.proj4)    # GDALDataset 
        first = datetime(1970,1,1,0) + timedelta(seconds = inTc[0])

        for invar, meta in zip(invars, metadata): # Re-using SrcMemDS for each variable, copy
            # TODO: Consider adding a parameter metadata(dict) to ExportEnkiNcVar, 
            # instead of transporting all metadata through the preliminary GDAL var.
            if 'standard_name' not in meta:
                if 'long_name' in meta:
                    meta['standard_name'] = meta['long_name']
                else:
                    meta['standard_name'] = 'none'
            SetGDALMetadata(SrcMemDS, meta['name'], meta['standard_name'], meta['units'], float(meta['_FillValue']))
            for t in range(tmin, tmax):     # data from the 3D or 4D numpy arrays into the 
                for e in range(emin,emax):  # 3D SrcMemDS, combining time and member dims.
                    bandno = t*(emax-emin)+e + 1
                    rband = SrcMemDS.GetRasterBand(bandno)
                    rband.SetNoDataValue(float(meta['_FillValue']))
                    if forecast_type == "Arome_ens":  rband.WriteArray(invar[t,e,:,:])
                    else:                             rband.WriteArray(invar[t,:,:])
                    # print(f'Copied input Ncvar to GDALDataset, t = {t}, e = {e},\
                    #         bandno = {bandno}, mean = {np.mean(invar)}, shape = {invar.shape}',\
                    #         end='\r', flush=True)

            # Export SrcMemDS to a NetCDF file
            band1 = SrcMemDS.GetRasterBand(1)
            name = band1.GetMetadataItem('name')
            ExportEnkiNcVar(SrcMemDS, outfname, first, timedelta(hours=1),"unix")

        if __name__ == "__main__":
            print ("Done downloading and clipping", flush=True)


        DSfnames.append(outfname)

    urlDS.close()
    # s.cookies.clear() 


    return DSfnames



# From a single Arome ncfile residing in datafld, create a multi-band memory 
# GDALDataset for each variable. Only a single file is processed; call
# repeatedly if successive inclusion of forecasts is required. If ensemble,
# the bands span a time major, member minor combined dimension
# Return a list of five multi-band GDALDataset sets: (T, RH, R, WS, P), 
# assuming no gaps.
# TODO: Seems to miss the ensemble dimension of the MEPS files.
def CreateTemporalGDALstacks(datafld, obsfilelist):
    if not datafld.endswith(os.sep):
        datafld += os.sep                   

    if not os.path.isdir(datafld):
        return None, 'CreateTemporalGDALstacks: Could not find folder ' + datafld 
    if type(obsfilelist) is not list or len(obsfilelist) < 1:
        return None, 'CreateTemporalGDALstacks: Invalid type or length of obsfilelist'
    if not os.path.isfile(datafld + obsfilelist[0]):
        return None, f'CreateTemporalGDALstacks: {obsfilelist[0]} not found in {datafld}'
    nfiles = len(obsfilelist)

    # Get some time info from the file names
    dt, dtsub, frmt = timedelta(hours=1), slice(-15,-4), "%Y%m%dT%H"
    assert(nfiles == 1)                     # Arome postprocessed or ensemble
    firstdt = datetime.strptime(obsfilelist[0][dtsub],frmt)
    lastdt = datetime.strptime(obsfilelist[-1][dtsub],frmt)
    nbands = int((lastdt-firstdt).total_seconds() / dt.total_seconds()) + 1
    # For single-file Arome data, lastdt==firstdt, and nbands will be replaced 

    testncfname = datafld + obsfilelist[0]
    ncfileinfo = GetNcInfo(testncfname)
    nt = ncfileinfo['nt']                   # GetNcInfo reads metadata from the
    ne = 1 if not 'nm' in ncfileinfo\
        else ncfileinfo['nm']               # (first) ncfile into a dict, and 
    nRows = ncfileinfo['ny']                # closes the file after use. These 
    nCols = ncfileinfo['nx']                # metadata should be the same for
    wktstring = ncfileinfo['wkt']           # all files in obsfilelist.
    xylim = [ncfileinfo['xmin'], ncfileinfo['xmax'],\
             ncfileinfo['ymax'], ncfileinfo['ymin']] # WENS
    mapvarnames = ncfileinfo['mapvars']     # metfields, lat/longitude, elevation
    # if nfiles == 1: 
    nbands = int(nt * ne)
    assert (nbands > 1)
    # else: assert(nt==1 and ne==1)

    SrcNcDS = Dataset(testncfname)          # Again open testncfname to get info
    ncv = SrcNcDS.variables                 # on each individual variable for 
    MemDataSets = []                        # creating GDALDatasets

    flag = -999.99                          # Will be updated

    for varname in mapvarnames:             # Construct an empty GDALDataset for 
        mapvar = ncv[varname]               # each metfield variable in the ncfile
        if mapvar.ndim < 3:  continue       # (Skip lat/longitude, elevation etc)

        MemDataSets.append(MemDSfromInfo(nRows, nCols, nbands, xylim, wktstring, ""))
        stdname = mapvar.__dict__['standard_name']
        units = mapvar.__dict__['units']
        flag = float(mapvar.__dict__['_FillValue'])
        SetGDALMetadata(MemDataSets[-1], varname, stdname, units, flag)

    # if nbands > 1 and nfiles == 1:        # We have an Arome file. Condition removed,
    for MemDS in MemDataSets:               # since this script is now Arome specific.
        varname = MemDS.GetMetadata()['name']
        ncvar = ncv[varname]

        for t in range(0, nt):              # Load the numpy arrays into the 3D GDALRasterBands
            validt = firstdt + t * dt
            for e in range(0, ne):          # ne==1 for AromePP files, > 1 for AromeMEPS files
                assert ncvar.ndim==3+(ne>1), f"Wrong number of dims {ncvar.ndim} in Arome PP file"
                ncmap = ncvar[t,:,:] if ne==1 else ncvar[t,e,:,:]
                QCresult = AromeQC(varname, validt, ncmap)
                if QCresult != "":
                    print(f"Quality Control Issue in {testncfname}: {QCresult}", flush=True)
                elif varname == "precipitation_amount":
                    print(f"Quality Control for {testncfname}::{varname} at {validt}: Passed", flush=True)
                bandno = t * ne + e + 1     # Equals t+1 when ne==1 and e consequently 0.
                rband = MemDS.GetRasterBand(bandno) # GDAL band numbers are 1-based.
                rband.SetNoDataValue(flag)
                rband.WriteArray(ncmap)

        # for t in range(0, nt):              # Load the 3/4D numpy arrays into the 3D GDALRasterBands
        #     if not 'nm' in ncfileinfo:      # No member dimension, this is an AromePP file
        #         assert ncvar.ndim==3, f"Wrong number of dims {ncvar.ndim} in Arome PP file"
        #         bandno = t + 1
        #         rband = MemDS.GetRasterBand(bandno)
        #         rband.SetNoDataValue(flag)
        #         rband.WriteArray(ncvar[t,:,:])
        #     else:
        #         for e in range(0, ne):      # This is an Arome MEPS ensemble file
        #             assert ncvar.ndim==4, f"Wrong number of dims {ncvar.ndim} in Arome MEPS file"
        #             bandno = t * ne + e + 1
        #             rband = MemDS.GetRasterBand(bandno)
        #             rband.SetNoDataValue(flag)
        #             rband.WriteArray(ncvar[t,e,:,:])
    SrcNcDS.close()
    return MemDataSets, firstdt

    # Rest of function written for SeNorge files (refactored out to its own script).

    # SrcNcDS.close() 
    # lastgoodidx = -1

    # We now know we are working with seNorge files, bandno iterates even time
    # steps, and nbands is the expected number of files, if all are present.
    # Most of this block is to handle temporal interpolation
    # for tidx, fname in enumerate(obsfilelist):
    #     if fname.startswith("Gap from: "):  # Detect missing files
    #         if lastgoodidx == -1:
    #             lastgoodidx = tidx-1      # Found the start of a gap
    #         continue                        # Do nothing until the gap is closed

    #     fullname = datafld + fname          # This file exists
    #     SrcNcDS = Dataset(fullname)         
    #     ncv = SrcNcDS.variables

    #     if lastgoodidx >= 0:                # This file closes a gap, fill up all
    #         for MemDS in MemDataSets:       # the missed tsteps by interpolation
    #             varname = MemDS.GetMetadata()['name']
    #             ncvar = ncv[varname]
    #             lastgoodmap = MemDS.GetRasterBand(lastgoodidx+1).ReadAsArray()
    #             for tmptidx in range(lastgoodidx+1,tidx,1):
    #                 lastgoodwgt = (tidx-tmptidx) / (tidx-lastgoodidx)
    #                 nextgoodwgt = (tmptidx-lastgoodidx) / (tidx-lastgoodidx)
    #                 rband = MemDS.GetRasterBand(tmptidx+1)
    #                 rband.SetNoDataValue(flag)
    #                 rband.WriteArray(   lastgoodmap        * lastgoodwgt \
    #                                   + ncvar[0,0,0,0,:,:] * nextgoodwgt    )
    #         lastgoodidx = -1
    #         # End of dataset loop
    #     for MemDS in MemDataSets:           # We still haven't written tidx itself
    #         varname = MemDS.GetMetadata()['name']
    #         ncvar = ncv[varname]
    #         rband = MemDS.GetRasterBand(tidx+1)
    #         rband.SetNoDataValue(flag)
    #         rband.WriteArray(ncvar[0,0,0,0,:,:])
        # End of dataset loop
    # End of tidx / file list loop  
    # 
    # return MemDataSets, firstdt





    # Return a list of <dataset> files in regionfld/dataset/, for the specified
    # period. A None in fromtime or totime means no limit in that direction.
    # For single-time PREC1h and TEMP1h files, any missing file is represented
    # by a flag string, keeping the list at fixed time interval. The first
    # element in the returned list is always an existing file, thus the
    # returned list is not guaranteed to start at fromtime.
def GetFileList(datafld, fromtime, totime):
    if not datafld.endswith(os.sep):
        datafld += os.sep

    regionfld = datafld[0:datafld[0:-1].rfind(os.sep)+1]
    product = datafld.split(os.sep)[-2]     # [-1] would be the blank after the sep

    if product == "AromePP":      # met_forecast_1_0km_nordic_20210203T13Z.nc
        fileprefix = "met_forecast_1_0km_nordic_"
        dtsub, frmt = slice(-15,-4), "%Y%m%dT%H"
    elif product == "AromeMEPS":    # meps_lagged_6_h_subset_2_5km_20210203T00Z.nc
        fileprefix = "meps_lagged_6_h_subset_2_5km_"
        dtsub, frmt = slice(-15,-4), "%Y%m%dT%H"
    else:   return f"Unsupported subfolder {datafld} supplied to Arome::GetFileList()"

    filelist = [fn for fn in os.listdir(datafld) if fn.startswith(fileprefix)] 
    filelist.sort()                 # Puts files in temporal order

    SrcTstep = timedelta(minutes=60)
    prevTime = None
    hrfilelist = list()
                            # Copy file names from filelist to obsfilelist.
    for zitem in filelist:  # Insert a "Gap from..." string for any missing
                            # file when product is 'PREC1h' or 'TEMP1h'.
        now = datetime.strptime(zitem[dtsub],frmt)
        if fromtime is not None and now < fromtime:  continue
        if totime is not None and now > totime:    break

        # if product not in ['PREC1h','TEMP1h']:
        hrfilelist.append(zitem)        # Skip temporal interpolation for 
        continue                        # forecasts and SeNorge2018

        '''                         # Temporal interploation for TEMP1h and PREC1h:
        if prevTime==None:              # First temporal file
            prevTime = now - SrcTstep
            assert(len(hrfilelist)==0)
        else:                           # Not the first file, prevTime is valid
            assert(len(hrfilelist)>0)
            dt = now - prevTime         # Validate time sequence of files
            if dt > SrcTstep:           # Time interval longer than expected
                flag = "Gap from: " + prevTime.strftime("%d.%m.%Y %H:%M")
                flag +=     " to: " + now.strftime("%d.%m.%Y %H:%M")
                # print (now, dt, SrcTstep, zitem, flag)

            while dt > SrcTstep:        # For each missing file, insert the 
                hrfilelist.append(flag) # "Gap from ..." message in hrfilelist
                prevTime += SrcTstep    # where the missing file name would have 
                dt = now - prevTime     # been, so hrfilelist remains hourly spaced.

        hrfilelist.append(zitem)    # The only real file name added in this iter.
        prevTime = now              # Time of the file just read, always existing.
        '''        

    return hrfilelist


def run_analysis(fnames):
    # Prepare the derived science data sets from the input and store in
    # the proper disk location.
    try:
        resultmsg = ""
        return resultmsg
    except Exception as e:
        msg = format("%s: %s") % (type(e),str(e))
        traceback.print_exc()
        return 'Arome.run_analysis error: ' + msg


# From https://pypi.org/project/netcdf4_pydap/
#import matplotlib.pyplot as plt
#import numpy as np
#import netcdf4_pydap
#
#credentials={'username': YOURUSERNAME,
#             'password': YOURPASSWORD,
#             'authentication_url':'https://urs.earthdata.nasa.gov/'}
#url = ('http://goldsmr3.gesdisc.eosdis.nasa.gov:80/opendap/'
#       'MERRA_MONTHLY/MAIMCPASM.5.2.0/1979/MERRA100.prod.assim.instM_3d_asm_Cp.197901.hdf')
#
#with netcdf4_pydap.Dataset(url, **credentials) as dataset:
#    data = dataset.variables['SLP'][0,:,:]
#    plt.contourf(np.squeeze(data))
#    plt.show()


thredds_site = {'AromePP':  "https://thredds.met.no/thredds/dodsC/metpplatest/",
                'AromeMEPS':"https://thredds.met.no/thredds/dodsC/mepslatest/"}
                # 'SN2018':   "https://thredds.met.no/thredds/dodsC/senorge/seNorge_2018/Latest/",
                # 'TEMP1h':   "https://thredds.met.no/thredds/dodsC/senorge/seNorge2/provisional_archive/TEMP1h/gridded_dataset/",
                # 'PREC1h':   "https://thredds.met.no/thredds/dodsC/senorge/seNorge2/provisional_archive/PREC1h/gridded_dataset/"}
# TEMP1h extend back to October 2015, PREC1h to December 2015 in the /provisional/ folder.
# Older data (Jan 2010 - Dec 2016) can be found at 
# Interactive:  https://thredds.met.no/thredds/catalog/senorge/seNorge2/archive/PREC1h or TEMP1h/gridded_dataset/"
# Script:       https://thredds.met.no/thredds/dodsC/senorge/seNorge2/archive/PREC1h or TEMP1h/gridded_dataset/"
'''
The GetNewData() function calls check_new() and then download_thredds() for any
of five data sets: SeNorge 2018 Daily, SeNorge 2.0 PREC1h, SeNorge 2.0 TEMP1h, 
Arome Postprocessed forecasts, and Arome MEPS ensemble. Data are downloaded
for a spatial window covering a template grid, and stored in a subfolder named
as the dataset. Only recent files not already present in the target folder is downloaded.
The program will look for 'RegionConfig.json' in regionfld for metadata.
'''
def GetNewData(dataset, mainfolder, regiondirs, start=None, end=None):

    if not mainfolder.endswith(os.sep):
        mainfolder += os.sep
    if dataset in thredds_site:             # thredds_site is a global dict, 
        urlsite = thredds_site[dataset]     # defined just above.
    else:
        return "Usage: GetNewData(dataset, workfld, start=None, end=None) with:\n \
                    one of 'AromePP', 'AromeMEPS' as dataset"
    if end==None:
        end = datetime.utcnow()
    if start==None:
        start = datetime.utcnow() - timedelta(hours=48)
    period = end - start
    if period.total_seconds() < 0:
        return f"GetNewData() for {dataset} failed due to start-end inconsistency"

    regionhome = f"{mainfolder}Regions{os.sep}"

    logmsg = ""

    for regfld in regiondirs:
        regionfld = f"{regionhome}{regfld}{os.sep}"
        datafld = regionfld + dataset

        # Cookie file cannot be read and written: (null)
        chunk = timedelta(days=5)  # Break up long periods in smaller chunks
        tmp_start = start
        nfiles = 0

        while tmp_start < end:
            # Get a list of relevant files on urlsite which are not present in datafld
            # For now accept that this queries the url for the same files for each region
            tmp_end = tmp_start + chunk
            remote_fnames = check_new(urlsite, datafld, tmp_start, min(end, tmp_end))
            if type(remote_fnames) == str:
                return f"check_new() for {dataset} failed: {remote_fnames}\n"
            if len(remote_fnames) == 0:
                logmsg += f"{regfld}: No new {dataset} files found at {thredds_site[dataset]}\n"
            else:
                logmsg += f"{regfld}: {len(remote_fnames)} new {dataset} files found at {thredds_site[dataset]}\n"

            if len(remote_fnames) > 0:
                # Download using OpenDAP, clip and store relevant window in datafld
                download_fnames = download_thredds(remote_fnames, datafld) 
                if type(download_fnames)==str:      # download_thredds knows forecast_type 
                    return (download_fnames+"\n")   # from the remote file names
                else:
                    nfiles += len(download_fnames)
            tmp_start += chunk
        
        filespec = f"; the last being {download_fnames[-1]}." if nfiles > 0 else ""

        logmsg += f"{nfiles} new {dataset} files downloaded for region {regfld}{filespec}\n"
        #except Exception as e:
        #    msg = format("%s: %s") % (type(e),str(e))
        #    logmsg += f"Thredds_processing::GetNewData raised exception: {msg}"

    return logmsg



def ResampleExport(dataset, mainfolder, regiondirs, fromtime=None, totime=None):
    # ResampleExport is called from the scheduler.
    # Calls GetFileList to get a list of locally available 1-hr files.
    # Builds a temporal MEM GDALDataset from these, and geotransforms each
    # band into a new MEM GDALDataset, which is exported to the NetCDF
    # file supplied as ENKIbasefname. Returns logmsg.


    if dataset in thredds_site:             # thredds_site is a global dict, 
        urlsite = thredds_site[dataset]     # defined just above.
    else:
        return "Usage: ResampleExport(dataset, regionfld, ENKIbasefname = "",\
                                        fromtime=None, totime=None) with:\n \
                                        one of 'AromePP', 'AromeMEPS' as dataset"

    regionhome = f"{mainfolder}Regions{os.sep}"
    logmsg = ""

    # The Region loop. Unlike seNorge and LandSAF, Arome never imports the entire domain.
    for regfld in regiondirs:
        regionfld = f"{regionhome}{regfld}{os.sep}"
        if not os.path.isfile(f"{regionfld}RegionConfig.json"):
            logmsg += f"Region {regionfld} skipped due to missing RegionConfig.json"
            continue

        datafld = regionfld + dataset

        ENKIbase = f"{regionfld}HourlyForecastInput.nc"    # AromePP
        EnsembleBase = f"{regionfld}Ensembles.nc"          # AromeMEPS

        try:
            datafld = regionfld + dataset
            localfilelist = GetFileList(datafld, fromtime, totime)

            if __name__ == "__main__":
                print(f"GetFileList found {len(localfilelist)} files in {datafld}\n", flush = True)

            # From the list of files, Create a time-gap-filled GDALDataset from the 
            # files returned by GetFileList
            # TEMP1h and PREC1h result in one GDALDataset holding the time series.
            # SeNorge2018 results in two GDALDatasets (prec/temp) holding time series
            # For Arome forecasts, only the last will be used, giving five GDALDatasets.

            if dataset in ['AromePP', 'AromeMEPS']:
                if __name__ == "__main__":      # Restrict to latest forecast [-1]
                    print("Calling CreateTemporalGDALstacks with the last Arome file:", localfilelist[-1])
                WndTserDS, startTime = CreateTemporalGDALstacks(datafld, [localfilelist[-1]])
                logmsg += f"{datetime.utcnow().strftime('%Y.%m.%d %H:%M')}: "
                if WndTserDS==None or len(WndTserDS)==0:
                    logmsg += f"No GDAL stack created for {dataset} in {regfld} at {startTime}\n"    
                else:
                    logmsg += f"{len(WndTserDS)} temporal GDAL stacks created in region {regfld} "\
                            + f"for {dataset} with the last Arome file {localfilelist[-1]}\n"
            
            with open(regionfld+'RegionConfig.json') as f:
                Cfg = json.load(f)
            tmpl_rst_fname = regionfld + Cfg["elev_GIS_dataset"]


            if len(WndTserDS) > 0:
                logmsg += f"{dataset}: Resampling {len(WndTserDS)} data sets for region {regfld}\n"
            for origDS in WndTserDS:
                # logmsg += f"Resampling and exporting {dataset}:{origDS.GetMetadata()['name']} to {regfld}\n"
                regDS = Resample(origDS, tmpl_rst_fname)
                logmsg += f"{datetime.utcnow().strftime('%Y.%m.%d %H:%M')}: "
                if dataset=='AromePP':
                    ExportEnkiNcVar(regDS, ENKIbase, startTime, timedelta(hours=1),"excel")
                    logmsg += f"{origDS.GetMetadata()['name']} resampled and exported to {ENKIbase}\n"
                else:
                    ExportEnkiNcVar(regDS, EnsembleBase, startTime, timedelta(hours=1),"excel")
                    logmsg += f"{origDS.GetMetadata()['name']} resampled and exported to {EnsembleBase}\n"
        except Exception as e:
            msg = format("%s: %s") % (type(e),str(e))
            logmsg += f"Arome_processing::ResampleExport raised exception: {msg}"
            traceback.print_exc()
    return logmsg




if __name__ == "__main__":

    mainfolder = 'G:\\Robot\\'

    regionhome = f"{mainfolder}Regions{os.sep}"
    regiondirs = [d for d in os.listdir(regionhome) if os.path.isdir(f"{regionhome}{d}")] 
    # Unlike LandSAF and seNorge, Arome data is not downloaded nationally.
    # GetNewData thus runs for the whole region list, just as ResampleExport.

    logmsg = GetNewData('AromePP', mainfolder, regiondirs)
    print(logmsg)
    logmsg = GetNewData('AromeMEPS', mainfolder, regiondirs)
    print(logmsg)

    logmsg = ResampleExport('AromePP', mainfolder, regiondirs)
    print(logmsg)
    logmsg = ResampleExport('AromeMEPS', mainfolder, regiondirs)
    print(logmsg)








'''
    # SeNorge 2.1 er ikke ført lenger fram enn 2015.
    # SeNorge 2.0 times-punktdata er ikke ført lenger fram enn april 2020. Bruk Frost.
    # Alle SeNorge-data ligger på https://thredds.met.no/thredds/dodsC/senorge/:
    #   SeNorge2018: seNorge_2018/Latest/seNorge2018_20210205.nc (Nedbør og temperatur)
    # SeNorge 2.0:  seNorge2/provisional_archive/
    #   Nedbør:     PREC1h/gridded_dataset/202102/seNorge_v2_0_PREC1h_grid_2021020511_2021020511.nc
    #   Temp:       TEMP1h/gridded_dataset/202102/seNorge_v2_0_TEMP1h_grid_2021020511.nc
    # Filname og nærmeste katalog navngis fra tid. 202102/seNorge_v2_0_TEMP1h_grid_2021020511.nc

    sn18site = "https://thredds.met.no/thredds/dodsC/senorge/seNorge_2018/Latest/"
    sn18_fnames = check_new(sn18site, datadir)     # seNorge2018_20210205.nc
    if type(sn18_fnames) == str:
        print ("check_new() for SeNorge 2018 failed:", sn18_fnames)
    elif len(sn18_fnames) == 0:
        print ("No new SN2018 urlfile available")
    else:
        download_fnames = download_thredds(sn18_fnames, datadir) # download_thredds will infer
        if type(download_fnames)==str:                    # forecast_type from the remote
            print("Download SN2018 failed:\n")            # file names, and look for file 
        else:                                             # 'RegionConfig.json' in the local
            print ("New SN2018 files downloaded:\n")      # datadir/ folder.
        print(download_fnames)




    sn20site = "https://thredds.met.no/thredds/dodsC/senorge/seNorge2/provisional_archive/"
    sn20_pfnames = check_new(f"{sn20site}PREC1h/gridded_dataset/", datadir) 
    if type(sn20_pfnames) == str:
        print ("check_new() for SeNorge 2.0 PREC1h failed:", sn20_pfnames)
    elif len(sn20_pfnames) == 0:
        print ("No new PREC1h urlfile available")
    else:
        download_fnames = download_thredds(sn20_pfnames, datadir) # download_thredds will infer
        if type(download_fnames)==str:                    # forecast_type from the remote
            print("Download PREC1h failed:\n")            # file names, and look for file 
        else:                                             # 'RegionConfig.json' in the local
            print ("New PREC1h files downloaded:\n")      # datadir/ folder.
        print(download_fnames)




    sn20_tfnames = check_new(f"{sn20site}TEMP1h/gridded_dataset/", datadir) 
    if type(sn20_tfnames) == str:
        print ("check_new() for SeNorge 2.0 TEMP1h failed:", sn20_tfnames)
    elif len(sn20_tfnames) == 0:
        print ("No new TEMP1h urlfile available")
    else:
        download_fnames = download_thredds(sn20_tfnames, datadir) # download_thredds will infer
        if type(download_fnames)==str:                    # forecast_type from the remote
            print("Download TEMP1h failed:\n")            # file names, and look for file 
        else:                                             # 'RegionConfig.json' in the local
            print ("New TEMP1h files downloaded:\n")      # datadir/ folder.
        print(download_fnames)




    # Both check_new() and download_thredds() handles SeNorge maps as well as
    # forecasts (ensemble or postprocessed), but only one type at a time. 
    # Call repeatedly to fetch more than one type of data.

    arppsite = "https://thredds.met.no/thredds/dodsC/metpplatest/"
    pp_fnames = check_new(arppsite, datadir)
    if type(pp_fnames) == str:
        print ("check_new() for postprocessed forecasts failed:", pp_fnames)
    elif len(pp_fnames) == 0:
        print ("No new url file available")
    else:
        download_fnames = download_thredds(pp_fnames, datadir) # download_thredds will infer
        if type(download_fnames)==str:                    # forecast_type from the remote
            print("Download ensemble failed:\n")          # file names, and look for file 
        else:                                             # 'RegionConfig.json' in the local
            print ("New ensemble forecasts downloaded:\n")       # datadir/ folder.
        print(download_fnames)



    arensite = "https://thredds.met.no/thredds/dodsC/mepslatest/" #meps_lagged_6_h_subset_2_5km_20210210T18Z.ncml
    en_fnames = check_new(arensite, datadir)
    if type(en_fnames) == str:
        print ("check_new() for postprocessed forecasts failed:", en_fnames)
    elif len(en_fnames) == 0:
        print ("No new url file available")
    else:
        download_fnames = download_thredds(en_fnames, datadir) # download_thredds will infer
        if type(download_fnames)==str:                    # forecast_type from the remote
            print("Download postproc failed:\n")          # file names, and look for file 
        else:                                             # 'RegionConfig.json' in the local
            print ("New postprocessed forecasts downloaded:\n")       # datadir/ folder.
        print(download_fnames)


'''


#    NCO:  ncks -d time,0,8,2 -d time,10 -d lat,-20.0,20.0 -d lon,50.0,350.0  -d lev,,,4
#    PYNCO:
#        opt = [
#                c.Limit("time", 0, 8, 2),
#                c.LimitSingle("time", 10),
#                c.Limit("lat", -20.0, 20.0),
#                c.Limit(dmn_name="lon", srt=50.0, end=350.0),
#                c.Limit(dmn_name="lev", srd=4)
#              ]
#    nco.ncks(input="in.nc", output="out.nc", options=opt)
#
#    opt = [
#            custom.Limit('y', 57, 650),
#            custom.Limit('x', 5, 750)
#            ]
#
#    nco.ncks( input=urlfname, output=localfld+"cropped_file.nc", options=opt )
#


#    https://thredds.met.no/thredds/ncss/meps25files/meps_det_pp_2_5km_latest.nc?
#    var=altitude&var=latitude&var=longitude&var=cloud_area_fraction&
#    var=precipitation_amount&var=x_wind_10m&var=y_wind_10m&
#    var=precipitation_amount_acc&var=air_temperature_2m&
#    var=relative_humidity_2m&north=71&west=4&east=30&south=58&
#    horizStride=1&time_start=2018-04-09T18%3A00%3A00Z&
#    time_end=2018-04-12T12%3A00%3A00Z&timeStride=1&vertCoord=&addLatLon=true
#
#    Temperatur:
#    https://thredds.met.no/thredds/ncss/meps25files/meps_det_pp_2_5km_latest.nc?
#    var=air_temperature_2m&north=71&west=4&east=30&south=58&disableProjSubset=on&horizStride=1&
#    time_start=2018-04-09T18%3A00%3A00Z&time_end=2018-04-12T12%3A00%3A00Z&timeStride=1&vertCoord=1
#    Fuktighet:
#    https://thredds.met.no/thredds/ncss/meps25files/meps_det_pp_2_5km_latest.nc?
#    var=relative_humidity_2m&north=71&west=4&east=30&south=58&disableProjSubset=on&horizStride=1&
#    time_start=2018-04-09T18%3A00%3A00Z&time_end=2018-04-12T12%3A00%3A00Z&timeStride=1&vertCoord=1
#    Nedbør:
#    https://thredds.met.no/thredds/ncss/meps25files/meps_det_pp_2_5km_latest.nc?
#    var=precipitation_amount&north=71&west=4&east=30&south=58&disableProjSubset=on&horizStride=1&
#    time_start=2018-04-09T18%3A00%3A00Z&time_end=2018-04-12T12%3A00%3A00Z&timeStride=1&vertCoord=1
#    Skyfraksjon:
#    https://thredds.met.no/thredds/ncss/meps25files/meps_det_pp_2_5km_latest.nc?
#    var=cloud_area_fraction&north=71&west=4&east=30&south=58&disableProjSubset=on&horizStride=1&
#    time_start=2018-04-09T18%3A00%3A00Z&time_end=2018-04-12T12%3A00%3A00Z&timeStride=1&vertCoord=1
#    U10m:
#    https://thredds.met.no/thredds/ncss/meps25files/meps_det_pp_2_5km_latest.nc?
#    var=x_wind_10m&north=66&west=4&east=15&south=57&disableProjSubset=on&horizStride=1&
#    time_start=2018-04-09T18%3A00%3A00Z&time_end=2018-04-12T12%3A00%3A00Z&timeStride=1&vertCoord=1
#    V10m:
#    https://thredds.met.no/thredds/ncss/meps25files/meps_det_pp_2_5km_latest.nc?
#    var=y_wind_10m&north=66&west=4&east=15&south=57&disableProjSubset=on&horizStride=1&
#    time_start=2018-04-09T18%3A00%3A00Z&time_end=2018-04-12T12%3A00%3A00Z&timeStride=1&vertCoord=1
#    U10m, V10m:
#    https://thredds.met.no/thredds/ncss/meps25files/meps_det_pp_2_5km_latest.nc?
#    var=x_wind_10m&var=y_wind_10m&north=66&west=4&east=15&south=57&disableProjSubset=on&horizStride=1&
#    time_start=2018-04-09T18%3A00%3A00Z&time_end=2018-04-12T12%3A00%3A00Z&timeStride=1&vertCoord=1
#    precip, clouds, U, V: (4 - 117 MB)
#    https://thredds.met.no/thredds/ncss/meps25files/meps_det_pp_2_5km_latest.nc?
#    var=cloud_area_fraction&var=precipitation_amount&var=x_wind_10m&var=y_wind_10m&
#    north=66&west=4&east=15&south=57&disableProjSubset=on&horizStride=1&
#    time_start=2018-04-09T18%3A00%3A00Z&time_end=2018-04-12T12%3A00%3A00Z&timeStride=1&vertCoord=1
#    Temp, fukt: (5 - 59 MB)
#    https://thredds.met.no/thredds/ncss/meps25files/meps_det_pp_2_5km_latest.nc?
#    var=air_temperature_2m&var=relative_humidity_2m&north=66&west=4&east=15&south=57&
#    disableProjSubset=on&horizStride=1&
#    time_start=2018-04-09T18%3A00%3A00Z&time_end=2018-04-12T12%3A00%3A00Z&timeStride=1&vertCoord=1
#
#    Med ny nedlasting hver 6 time og 60 timer i varselet vil det være 10 varsel for hvert tidsskritt


# u'time',
# u'forecast_reference_time',
# u'height_above_msl',
# u'height0',
# u'height1',
# u'projection_lambert',
# u'x',
# u'y',
# u'p0',
# u'longitude',
# u'latitude',
# u'surface_air_pressure',
# u'relative_humidity_2m',
# u'air_pressure_at_sea_level',
# u'precipitation_amount_acc',
# u'fog_area_fraction',
# u'land_area_fraction',
# u'precipitation_amount',
# u'precipitation_amount_high_estimate',
# u'precipitation_amount_low_estimate',
# u'precipitation_amount_middle_estimate',
# u'precipitation_amount_prob_low',
# u'cloud_area_fraction',
# u'high_type_cloud_area_fraction',
# u'low_type_cloud_area_fraction',
# u'medium_type_cloud_area_fraction',
# u'helicopter_triggered_index',
# u'thunderstorm_index_combined',
# u'wind_speed_maxarea_10m',
# u'wind_speed_of_gust',
# u'x_wind_10m',
# u'y_wind_10m',
# u'air_temperature_2m',
# u'altitude']
#
#
    



# Before copying variable tables, we reproject the image. This is probably easiest done
# by using two GDAL MEM data sets, extracting tables from the remote NetCDF file into the
# first, reprojecting, and then exporting from the second to the local target NetCDF file.
# We create two functions returning WKT's from 1) the projection_lcc/projection_lambert
# variable and the

# NorwMapDS = MemDriver.Create("",NorwCols,NorwRows,1,gdal.GDT_Float32)
# NorwMapDS.SetProjection(NorwSRS.ExportToWkt())
# result = gdal.ReprojectImage(WndTserDS,NorwTrgDS,SrcSRS.ExportToWkt(),\
#                  NorwSRS.ExportToWkt(),gdal.GRA_NearestNeighbour)
# if result != 0: print result, gdal.GetLastErrorMsg()

# RegTrgDS = MemDriver.Create("",RegCols,RegRows,numtsteps,gdal.GDT_Float32)
# RegTrgDS.SetProjection(RegSRS.ExportToWkt())
# RegTrgDS.SetGeoTransform(RegGT)


# Copy variable tables
'''
if inTc is not None:   outNcTcoord[:]  = inTc[:]
else:                  outNcTcoord[:] = np.nan
if inEc is not None:   outNcEcoord[:]  = inEc[:]
else:                  outNcEcoord[:] = np.nan
if inYc is not None:   outNcYcoord[:]  = inYc[:]
else:                  outNcYcoord[:] = np.nan
if inXc is not None:   outNcXcoord[:]  = inXc[:]
else:                  outNcXcoord[:] = np.nan
if inLat is not None:  outNcLat[:] = inLat[:]
else:                  outNcLat[:] = np.nan
if inLon is not None:  outNcLon[:] = inLon[:]
else:                  outNcLon[:] = np.nan

if inTair is not None: outNcTair[:] = inTair[:]
else:                  outNcTair[:] = np.nan
if inRH  is not None:  outNcRH[:]   = inRH[:]
else:                  outNcRH[:] = np.nan
# if inClF is not None:  outClF[:]  = inClF[:]
# else:                  outClF[:] = np.nan
if inWS is not None:   
    outNcWS[:]  = inWS[:]
elif inWsX is not None and inWsY is not None:
    outNcWS[:]  = np.sqrt(np.power(inWsX[:],2) + np.power(inWsY[:],2))
else:                  
    outNcWS[:] = np.nan

if inPrec is not None: 
    outNcPrec[:] = inPrec[:]
elif inPrecAcc is not None:         # Must differentiate in time to convert 
    # outPrecAcc[:]  = inPrecAcc[:]   # the accumulated precipitation to hourly
    outNcPrec[:]  = np.insert(np.diff(inPrecAcc.filled(),axis=0),0,0,axis=0)
else:
    # outPrecAcc[:]  = np.nan
    outNcPrec[:]  = np.nan

# if inSnwFAcc is not None:   outSnwFAcc[:]  = inSnwFAcc[:]
# else:                       outSnwFAcc[:]  = np.nan

if inSWradAcc is not None:  
    # outSWradAcc[:] = inSWradAcc[:]
    outNcSWrad[:]  = np.insert(np.diff(inSWradAcc.filled(),axis=0),0,0,axis=0) / 3600.0
else:                       
    # outSWradAcc[:] = np.nan
    outNcSWrad[:] = np.nan

# if inLWradAcc is not None:  
#     outLWradAcc[:] = inLWradAcc[:]
#     outLWrad[:]  = np.insert(np.diff(inLWradAcc.filled(),axis=0),0,0,axis=0) / 3600.0
# else:                       
#     outLWradAcc[:] = np.nan
#     outLWrad[:] = np.nan

'''


'''

# Open and read from each ncfile in obsfilelist a window covering Norway.
# For Arome forecasts, obsfilelist may only contain one file name. Call
#   repeatedly if successive inclusion of forecasts is required.
# For TEMP1h and PREC1h, obsfilelist has flag entries for missing files
# From these maps build one or more temporal multi-band GDALDataset sets:
#   TEMP1h and PREC1h:  A single GDALdataset with gaps interpolated
#   SeNorge2018:        Two GDALDatasets (precip and temp)
#   Arome forecasts:    Five GDALDatasets (T, RH, R, WS, P)
# For TEMP1h and PREC1h, temporally interpolate any missing map
# For Arome ensembles, the bands span a combined time, member index
# Return list of MEM GDALDatasets, not yet geotransformed.
def CreateTemporalGDALstacks(datafld, obsfilelist):

    # If ENKIbasefname is specified, also append the downloaded data to it, 
    # creating it in regionfld if it doesn't already exist. 
    if not datafld.endswith(os.sep):
        datafld += os.sep                               # include the last sep
    regionfld = datafld[0:datafld[0:-1].rfind(os.sep)+1]      

    with open(regionfld+'RegionConfig.json') as f:
        Cfg = json.load(f)

    # For SeNorge files, nbands is length of obsfilelist, some of these may
    # not exist. For Arome forecasts, len(obsfilelist) should be 1, nbands 
    # is the length of that file's time dimension

    nbands = len(obsfilelist)               # Preliminary, will be set again 
    if nbands > 1:                          # if loading an Arome file.
        TrgMemDS = MemDSfromGISfile(regionfld + Cfg["elev_GIS_dataset"],nbands)
        TrgXcoords, TrgYcoords = GetXYcoords(TrgMemDS)
    else TrgMemDS = None

    MemDataSets = []


    for fname in obsfilelist:  
        fullname = datafld+fname            # obsfilelist contains only file 
        SrcNcDS = Dataset(fullname)         # names, not the entire paths.
        ncv = SrcNcDS.variables

        if ncv['time'].shape[0] > 1:        # This is an Arome file, it should be
            assert(len(obsfilelist)==1)     # the only one, and span the temporal 
            nbands = ncv['time'].shape[0]   # nbands internally. Having opened the 
            if TrgMemDS == None:            # SrcNcDS, we can create the TrgMemDS.
                TrgMemDS = MemDSfromGISfile(regionfld + Cfg["elev_GIS_dataset"],nbands)
                TrgXcoords, TrgYcoords = GetXYcoords(TrgMemDS)


        outDS = Dataset(outfname, "w", format="NETCDF4")  #output file

        outDS.createDimension('time', None)      # unlimited

        outDS.createDimension('y', ny)
        outDS.createDimension('x', nx)
        if forecast_type=='Arome_ens':
            outDS.createDimension('ensemble_member', emax-emin)  #
            dyndim = ('time', 'ensemble_member', 'y', 'x')       # Vertical dim was not extracted
        else:
            dyndim = ('time', 'y', 'x')              # No ensemble dim
        statdim = ('y', 'x')

        # print ("Creating and copying output ncvariables which do not need reprojection", flush=True)
        if "seNorge" not in urlfname:
            outNcRefTime = outDS.createVariable('forecast_reference_time','f4')
            outNcRefTime[...] = datetime.timestamp(DTref)
            outNcRefTime.timetag = DTref.strftime("%Y-%m-%d %H:%M")
            outNcRefTime.units = "seconds since 1970-01-01 00:00:00 +00:00"
            outNcRefTime.standard_name = "forecast_reference_time"
            print(f"DTref forecast refererence time written to result file: {DTref} ({datetime.timestamp(DTref)})")


        # outNcSCS  = outDS.createVariable('projection_lambert','i4')
        # outNcSCS  = inSCS
        outNcTcoord  = outDS.createVariable('time', 'float64', ('time'))
        outNcTcoord[:]  = inTc[:]
        if inEc is not None:
            outNcEcoord  = outDS.createVariable('ensemble_member', 'float64', ('ensemble_member'))
            outNcEcoord[:]  = inEc[:]
        outNcYcoord  = outDS.createVariable('y', 'float64', ('y'))
        outNcYcoord[:]  = TrgYcoords
        outNcYcoord.standard_name = "projection_y_coordinate"
        outNcYcoord.units = "m"
        outNcXcoord  = outDS.createVariable('x', 'float64', ('x'))
        outNcXcoord[:]  = TrgXcoords
        outNcXcoord.standard_name = "projection_x_coordinate"
        outNcXcoord.units = "m"


        # print ("Calculating target latlong fields")
        outNcLat = outDS.createVariable('latitude', 'float64', statdim)
        outNcLon = outDS.createVariable('longitude', 'float64', statdim)

        targSRS = osr.SpatialReference()
        targSRS.ImportFromWkt(TrgMemDS.GetProjectionRef())
        mapping_name = targSRS.GetAttrValue('projection').lower()
        # lower() works for some projections, incl. "transverse_mercator"
            # Create a netcdf variable holding all refsyst info as attributes
        rsVar = outDS.createVariable(mapping_name,'i4')
        rsVar.grid_mapping_name = mapping_name
        rsVar.longitude_of_central_meridian = targSRS.GetProjParm('central_meridian')
        rsVar.false_easting = targSRS.GetProjParm('false_easting') # 500000.0
        rsVar.false_northing = targSRS.GetProjParm('false_northing') # 0.0
        rsVar.latitude_of_projection_origin = targSRS.GetProjParm('latitude_of_projection_origin')
        rsVar.scale_factor_at_central_meridian = targSRS.GetProjParm('scale_factor')
        # rsVar.longitude_of_prime_meridian = targSRS.GetProjParm('longitude_of_projection_origin')
        rsVar.semi_major_axis = targSRS.GetSemiMajor() # 6378137.0
        rsVar.inverse_flattening = targSRS.GetInvFlattening() # 298.257223563
        rsVar.crs_wkt = TrgMemDS.GetProjectionRef()
        # :_FillValue = -1.0; // double
        # :utm_zone_number = 33; // int
        # :proj4 = "+proj=utm +zone=33 +datum=WGS84 +units=m +no_defs +ellps=WGS84 +towgs84=0,0,0";
        # :_CoordinateTransformType = "Projection";
        # :_CoordinateAxisType = "GeoX GeoY";
        rsVar.utm_zone_number = targSRS.GetUTMZone()
        rsVar.proj4 = targSRS.ExportToProj4()
        # print("Target coordinate system name = ",targSRS.GetName())


        if inNcElev is not None:
            outNcElev = outDS.createVariable('altitude', 'float32', statdim, fill_value=Eflag)
            SetNcMetaData(outNcElev, "altitude", "m", mapping_name)

        print ("Creating output ncvariables for reprojected metorological fields")
        outNcTair = outNcRH = outNcSWrad = outNcWS = outNcPrec = None
        outvars = []

        if "TEMP1h" in urlfname:    # Three different sources of temperature data
            outNcTair = outDS.createVariable('TEMP1h', 'float32', dyndim, fill_value=Tflag)
            outvars = [outNcTair]  # masked_array, NcVar
        elif "seNorge2018" in urlfname:
            outNcTair = outDS.createVariable('TEMP1d', 'float32', dyndim, fill_value=Tflag)
        elif inTair is not None:
            outNcTair = outDS.createVariable('air_temperature_2m', 'float32', dyndim, fill_value=Tflag)
        if outNcTair is not None:
            SetNcMetaData(outNcTair, "air_temperature_2m", "DegC", mapping_name)

        if inRH is not None:
            outNcRH   = outDS.createVariable('relative_humidity_2m',  'float32', dyndim, fill_value=Hflag)
            SetNcMetaData(outNcRH, "relative_humidity_2m", "%", mapping_name)
        if inSWrad is not None:
            outNcSWrad = outDS.createVariable('shortwave_flux', 'float32', dyndim, fill_value=Rflag)
            SetNcMetaData(outNcSWrad, "surface_downwelling_shortwave_flux_in_air", "W/m2", mapping_name)
        if inWS is not None:
            outNcWS   = outDS.createVariable('wind_speed_10m', 'float32', dyndim, fill_value=Wflag)
            SetNcMetaData(outNcWS, "wind_speed_10m", "m/s", mapping_name)
        
        if "PREC1h" in urlfname:
            outNcPrec = outDS.createVariable('PREC1h', 'float32', dyndim, fill_value=Pflag)
            outvars = [outNcPrec]  # masked_array, NcVar
        elif "seNorge2018" in urlfname:
            outNcPrec = outDS.createVariable('PREC1d', 'float32', dyndim, fill_value=Pflag)
        elif inPrec is not None:
            outNcPrec = outDS.createVariable('precipitation_amount', 'float32', dyndim, fill_value=Pflag)
        if outNcPrec is not None:
            SetNcMetaData(outNcPrec, "precipitation_amount", "mm", mapping_name)


        if "Arome_" in forecast_type:
            outvars = [outNcTair, outNcRH, outNcSWrad, outNcWS, outNcPrec]  # NcVars
        elif "seNorge2018" in urlfname:
            outvars = [outNcTair, outNcPrec]                                # NcVars
        # For the hourly SeNorge files, invars and outvars are already set.


        outNcTcoord.standard_name = "time"
        outNcTcoord.units = "seconds since 1970-01-01 00:00:00 +00:00"
        outNcTcoord.long_name = "time"
        outNcLon.standard_name = "longitude"
        outNcLon.units = "degrees_east"
        outNcLon.long_name = "longitude"
        outNcLon.grid_mapping = mapping_name
        outNcLat.standard_name = "latitude"
        outNcLat.units = "degrees_north"
        outNcLat.long_name = "latitude"
        outNcLat.grid_mapping = mapping_name


        latlon = osr.SpatialReference()
        latlon.ImportFromEPSG(4326)
        # srctransform = osr.CoordinateTransformation(source, latlon)
        trgtransform = osr.CoordinateTransformation(targSRS, latlon)

        for i, y in enumerate(TrgYcoords):
            for j, x in enumerate(TrgXcoords):
                point = ogr.Geometry(ogr.wkbPoint)
                point.AddPoint(float(x),float(y))
                point.ExportToWkt()
                point.Transform(trgtransform)
                outNcLat[i,j] = point.GetX()
                outNcLon[i,j] = point.GetY()

                # print(type(point.ExportToWkt()),point.ExportToWkt())
                # print(point.GetX(), point.GetY(), type(point.GetX()))

        # return "Test"

        if inNcElev is not None:
            SrcMemDS.GetRasterBand(1).WriteArray(inElev[:,:])
            result = gdal.ReprojectImage( SrcMemDS,TrgMemDS,SrcMemDS.GetProjection(),\
                                        TrgMemDS.GetProjection(),gdal.GRA_NearestNeighbour)
            outNcElev[:,:] = TrgMemDS.GetRasterBand(1).ReadAsArray(0,0,nx,ny)


        for (invar, outvar) in zip(invars, outvars):
            # Now load the 4D numpy arrays into the 3D GDALRasterBands
            SetGDALMetadata(TrgMemDS, outvar.name, outvar.standard_name, outvar.long_name, outvar.units,  float(outvar._FillValue))

            for t in range(tmin, tmax):
                for e in range(emin,emax):
                    bandno = t*(emax-emin)+e + 1
                    rband = SrcMemDS.GetRasterBand(bandno)
                    rband.SetNoDataValue(float(outvar._FillValue))
                    if forecast_type == "Arome_ens":   rband.WriteArray(invar[t,e,:,:])
                    else:                             rband.WriteArray(invar[t,:,:])
                    m = np.mean(invar)
                    # print(f'Copied input Ncvar to GDALDataset, t = {t}, e = {e},\
                    #         bandno = {bandno}, mean = {m}, shape = {invar.shape}',\
                    #         end='\r', flush=True)

            print('Reprojecting', outvar.name, '...', end = '', flush=True)

            result = gdal.ReprojectImage( SrcMemDS,TrgMemDS,SrcMemDS.GetProjection(),\
                                        TrgMemDS.GetProjection(),gdal.GRA_NearestNeighbour)

            if result != 0:
                print ("Failed.\n", result, gdal.GetLastErrorMsg(), flush=True)
            else:
                print ("Done", flush=True)


            for t in range(tmin, tmax):             # Copy the data tables from the 
                for e in range(emin,emax):          # geotransformed TrgMemDS to the
                    bandno = t*(emax-emin)+e + 1    # target NcVar; bands=time*ens.
                    rband = TrgMemDS.GetRasterBand(bandno)
                    if forecast_type == "Arome_ens": 
                        outvar[t,e,:,:] = rband.ReadAsArray(0,0,nx,ny)
                    else:
                        outvar[t,:,:] = rband.ReadAsArray(0,0,nx,ny)
                        
                    m = np.mean(outvar)

            if ENKIbasefname != "": # Check this
                if not os.path.isabs(ENKIbasefname):
                    ENKIbasefname = regionfld + ENKIbasefname.split(os.sep)[-1]
                print (f"Exporting {outvar.name} to ENKI database...", end='', flush=True)
                first = datetime(1970,1,1,0) + timedelta(seconds = inTc[0])
                ExportEnkiNcVar(TrgMemDS, ENKIbasefname, first, timedelta(hours=1))
                print ("Done.", flush=True)

        outDS.close()
        urlDS.close()
        print('\n Done reprojecting and exporting', flush=True)

        DSfnames.append(outfname)

    # End of file list loop

    return DSfnames

'''
